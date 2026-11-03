"""
Shared modelling, fast simulation engine, and plotting utilities for the
fluidic excitable neuron and half-centre oscillator (HCO) simulations.

Physical units are SI throughout (Pa, m, m^3, kg/s, s, W). Internal thermodynamic
states use absolute pressure; exported data and plots use gauge pressure
(p_gauge = p - P_ATM).
"""

from __future__ import annotations

import collections
import csv
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

for _var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import fonsim as fons  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# =============================================================================
# 1. Physical Constants & Default Model Parameters
# =============================================================================

P_ATM: float = fons.pressure_atmospheric  # Atmospheric pressure [Pa] (101 300 Pa)
BAR: float = 1e-5                         # Conversion factor Pa -> bar
MW: float = 1e3                           # Conversion factor W  -> mW

# Constant supply pressures (absolute [Pa])
SOURCES: dict[str, float] = {
    "I": 1.0 * P_ATM,  # Inhibitory sink (0.0 bar gauge)
    "L": 1.3 * P_ATM,  # Leak / resting supply (~0.30 bar gauge)
    "E": 2.0 * P_ATM,  # Excitatory supply (~1.01 bar gauge)
}

# Default EI fluidic neuron parameters
NEURON_PARAMS: dict[str, Any] = {
    "p_init": SOURCES["L"],   # Initial container pressure [Pa]
    "volume_f": 1e-6,         # Fast membrane chamber volume [m^3]
    "volume_s": 1e-6,         # Slow recovery chamber volume [m^3]
    "d_L": 1e-3, "L_L": 1.0,  # Passive leak restrictor (container_f <-> source_L)
    "d_s": 3e-4, "L_s": 1.0,  # Slow recovery restrictor (container_f <-> container_s)
    # Excitatory memductive valve (container_f <-> source_E, gated by e_m)
    "d_E": 2e-3, "L_E": 1.0, "g_max_E": 1.0, "a_E": 3e-4, "p_th_E": 1.45 * P_ATM,
    # Inhibitory memductive valve (container_f <-> source_I, gated by e_s)
    "d_I": 3e-3, "L_I": 1.0, "g_max_I": 1.0, "a_I": 2e-4, "p_th_I": 1.45 * P_ATM,
}

# Default inhibitory restrictor synapse parameters (post.f <-> source_I, gated by pre.s)
SYNAPSE_PARAMS: dict[str, Any] = {
    "d_syn_I": 6e-3,
    "L_syn_I": 1.0,
    "g_max_syn_I": 1.0,
    "a_syn_I": 8e-4,
    "p_th_syn_I": 1.35 * P_ATM,
}

# External stimulus coupling tube geometry
STIM_TUBE: dict[str, float] = {"d": 1e-3, "L": 1.0}


def to_gauge_pa(p_abs: np.ndarray | float) -> np.ndarray | float:
    """Convert absolute pressure [Pa] to gauge pressure above P_ATM [Pa]."""
    return p_abs - P_ATM


def to_gauge_bar(p_abs: np.ndarray | float) -> np.ndarray | float:
    """Convert absolute pressure [Pa] to gauge pressure [bar]."""
    return (p_abs - P_ATM) * BAR


# =============================================================================
# 2. Custom Component & Fast Vectorised Simulation Engine
# =============================================================================

def fast_custom_wave(points: list[tuple[float, float]], scale: float = 1.0):
    """Piecewise-constant (zero-order hold) waveform evaluation function."""
    pts = [(float(t), float(y) * scale) for t, y in points]
    t_first, y_first = pts[0]
    t_last, y_last = pts[-1]

    def evaluate_wave(t: float) -> float:
        if t < t_first:
            return y_first
        if t >= t_last:
            return y_last
        res = y_first
        for ti, yi in pts:
            if t >= ti:
                res = yi
            else:
                break
        return res

    return evaluate_wave


def _precompute_tube_constants(obj) -> None:
    """Precompute Hagen-Poiseuille laminar resistance constant."""
    obj._r_lam = 128.0 / math.pi * obj.length / (obj.d ** 4) * obj.fluid.mu


class FastModulatedRestrictor(fons.Component):
    """Laminar tube restrictor whose conductance is modulated by a sigmoid of p_c:

        g(p_c) = g_max / (1 + exp(-a * (p_c - p_th)))

    Terminals: 'a' (p0, mf0), 'b' (p1, mf1), 'c' (p_c, mf_c).
    Governing equations:
        0: g(p_c) * (p0 - p1) - (r_lam / rho) * mf0 = 0
        1: mf0 + mf1 = 0
        2: mf_c = 0
    """

    def __init__(
        self,
        label: str | None = None,
        fluid=None,
        length: float = 1.0,
        diameter: float = 2e-3,
        roughness: float = 1.5e-6,
        g_max: float = 1.0,
        a: float = 1e-3,
        p_th: float = 1.45 * P_ATM,
    ):
        super().__init__(label)
        self.set_terminals(
            fons.Terminal("a", fons.terminal_fluidic, {"pressure": "p0", "massflow": "mf0"}),
            fons.Terminal("b", fons.terminal_fluidic, {"pressure": "p1", "massflow": "mf1"}),
            fons.Terminal("c", fons.terminal_fluidic, {"pressure": "p_c", "massflow": "mf_c"}),
        )
        self.nb_equations = 3
        self.fluid = fluid
        self.length = length
        self.d = diameter
        self.e = roughness
        self.g_max = g_max
        self.a = a
        self.p_th = p_th
        self.Re_threshold_min = 1e12
        self.Re_threshold_max = 2e12
        _precompute_tube_constants(self)

    def get_gating(self, p_c=None) -> np.ndarray:
        """Evaluate dimensionless gating variable g(p_c) in [0, g_max]."""
        if p_c is None:
            p_c = self.get("pressure", "c")
        x = np.clip(self.a * (p_c - self.p_th), -500.0, 500.0)
        return self.g_max / (1.0 + np.exp(-x))


class FastCircularTube(fons.CircularTube):
    """Laminar circular tube with precomputed hydraulic resistance."""

    def __init__(self, label=None, fluid=None, length=1.0, diameter=1e-3, roughness=1.5e-6):
        super().__init__(label=label, fluid=fluid, length=length, diameter=diameter, roughness=roughness)
        self.nb_equations = 2
        self.Re_threshold_min = 1e12
        self.Re_threshold_max = 2e12
        _precompute_tube_constants(self)


class FastContainer(fons.Container):
    """Isothermal ideal-gas capacitive chamber without decorator overhead."""

    def __init__(self, label=None, fluid=None, volume=1e-6):
        super().__init__(label=label, fluid=fluid, volume=volume)
        self.nb_equations = 1


def _padded_index_table(col_lists: list[list[int] | np.ndarray]):
    width = max((len(c) for c in col_lists), default=1)
    idx = np.zeros((len(col_lists), width), dtype=np.intp)
    mask = np.zeros((len(col_lists), width), dtype=bool)
    for i, cols in enumerate(col_lists):
        idx[i, :len(cols)] = cols
        idx[i, len(cols):] = cols[0]
        mask[i, :len(cols)] = True
    return idx, mask


class FastSimulation(fons.Simulation):
    """Drop-in FONSim simulation engine with vectorised component evaluation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_fast_structures()

    def _init_fast_structures(self) -> None:
        n_net = self.nb_network_equations
        net_rows = [np.flatnonzero(self.H[i, :] != 0) for i in range(n_net)]
        self._net_col_idx, _ = _padded_index_table(net_rows)

        c = collections.defaultdict(list)
        ps = collections.defaultdict(list)
        ms = []
        t = collections.defaultdict(list)
        m = collections.defaultdict(list)
        t_comps, m_comps = [], []
        dyn_rows, dyn_cols, const_rows, const_cols = [], [], [], []

        irow = n_net
        for comp in self.system.components:
            k = comp.nb_equations
            ka = self.phi_index_slice_by_component[comp]
            ks = self.state_index_slice_by_component[comp]
            a0 = ka.start

            if isinstance(comp, fons.Container):
                c["rows"].append(irow)
                c["ks"].append(ks.start)
                c["mf"].append(a0)
                c["p"].append(a0 + 1)
                c["mass_stp"].append(comp.mass_stp)
                self.H[irow, a0 + 1] = -comp.mass_stp
                dyn_rows.append(irow)
                dyn_cols.append([a0, a0 + 1])

            elif isinstance(comp, fons.PressureSource) and not callable(comp.pressure):
                ps["rows"].append(irow)
                ps["p"].append(a0)
                ps["val"].append(float(comp.pressure))
                self.H[irow, a0] = 1.0
                const_rows.append(irow)
                const_cols.append([a0])

            elif isinstance(comp, fons.MassflowSource) and callable(comp.massflow):
                ms.append((irow, a0, comp.massflow))
                self.H[irow, a0] = 1.0
                const_rows.append(irow)
                const_cols.append([a0])

            elif isinstance(comp, FastCircularTube):
                t_comps.append(comp)
                t["row0"].append(irow)
                t["row1"].append(irow + 1)
                t["mf0"].append(a0)
                t["p0"].append(a0 + 1)
                t["mf1"].append(a0 + 2)
                t["p1"].append(a0 + 3)
                t["rlam"].append(comp._r_lam)
                self.H[irow + 1, a0] = 1.0
                self.H[irow + 1, a0 + 2] = 1.0
                dyn_rows.append(irow)
                dyn_cols.append([a0, a0 + 1, a0 + 3])
                const_rows.append(irow + 1)
                const_cols.append([a0, a0 + 2])

            elif isinstance(comp, FastModulatedRestrictor):
                m_comps.append(comp)
                for key, off in (("mf0", 0), ("p0", 1), ("mf1", 2), ("p1", 3), ("mfc", 4), ("pc", 5)):
                    m[key].append(a0 + off)
                m["row0"].append(irow)
                m["row1"].append(irow + 1)
                m["row2"].append(irow + 2)
                m["rlam"].append(comp._r_lam)
                m["gmax"].append(comp.g_max)
                m["a"].append(comp.a)
                m["pth"].append(comp.p_th)
                self.H[irow + 1, a0] = 1.0
                self.H[irow + 1, a0 + 2] = 1.0
                self.H[irow + 2, a0 + 4] = 1.0
                dyn_rows.append(irow)
                dyn_cols.append([a0, a0 + 1, a0 + 3, a0 + 5])
                const_rows += [irow + 1, irow + 2]
                const_cols += [[a0, a0 + 2], [a0 + 4]]

            irow += k

        def to_arr(d):
            float_keys = ("mass_stp", "val", "rlam", "gmax", "a", "pth")
            return {k: np.array(v, dtype=float if k in float_keys else np.intp) for k, v in d.items()}

        self._c, self._ps, self._t, self._m = to_arr(c), to_arr(ps), to_arr(t), to_arr(m)
        self._ms_list = ms
        rho_comps = t_comps + m_comps
        self._rho_factor = rho_comps[0].fluid.rho_stp / P_ATM if rho_comps else 0.0

        self._const_rows = np.array(const_rows, dtype=np.intp)
        self._const_col_idx, _ = _padded_index_table(const_cols)
        self._dyn_rows = np.array(dyn_rows, dtype=np.intp)
        self._dyn_col_idx, self._dyn_valid_mask = _padded_index_table(dyn_cols)
        self._dyn_row_idx = np.repeat(self._dyn_rows[:, None], self._dyn_col_idx.shape[1], axis=1)

    def _fill_const_tolerances(self, tol, tol_inv) -> None:
        if len(self._const_rows):
            c_tols = np.maximum(
                np.min(self.argument_tolerances[self._const_col_idx], axis=1),
                self.minimum_solving_tolerance,
            )
            tol[self._const_rows] = c_tols
            tol_inv[self._const_rows] = 1.0 / c_tols

    def fill_network_equation_tolerances(self, tol, tol_inv) -> None:
        n_net = self.nb_network_equations
        np.min(self.argument_tolerances[self._net_col_idx], axis=1, out=tol[:n_net])
        tol_inv[:n_net] = 1.0 / tol[:n_net]
        self._fill_const_tolerances(tol, tol_inv)

    def fill_component_equation_tolerances(self, tol, tol_inv) -> None:
        aa = np.abs(self.H[self._dyn_row_idx, self._dyn_col_idx]) * self.argument_tolerances[self._dyn_col_idx]
        aa = np.where(self._dyn_valid_mask & (aa > 0.0), aa, np.inf)
        a = np.maximum(np.min(aa, axis=1), self.minimum_solving_tolerance)
        tol[self._dyn_rows] = a
        tol_inv[self._dyn_rows] = 1.0 / a
        if self.simstep == 0:
            self._fill_const_tolerances(tol, tol_inv)

    def update_state(self, simstep_args, simstep_state, dt, discretization) -> None:
        tustin = discretization == "tustin"
        phi_cur = self.phi[simstep_args]
        phi_prev = self.phi[simstep_args - 1] if tustin else None
        if "ks" in self._c:
            ks, mf = self._c["ks"], self._c["mf"]
            mf_val = 0.5 * (phi_prev[mf] + phi_cur[mf]) if tustin else phi_cur[mf]
            self.state[simstep_state + 1, ks] = self.state[simstep_state, ks] + dt * mf_val

    def evaluate_equations(
        self, simstep, g, H, elapsed_time, dt, discretization, force_update_all_states=False
    ) -> None:
        simstep_state = simstep - 1 if discretization in ("backward", "tustin") else simstep
        propagate = dt != 0 and discretization in ("backward", "tustin")
        tustin = discretization == "tustin"
        phi_cur = self.phi[simstep]
        phi_prev = self.phi[simstep - 1] if tustin and propagate else None
        state_arr = self.state[simstep_state]
        p_min = 0.1 * P_ATM

        if "rows" in self._ps:
            g[self._ps["rows"]] = phi_cur[self._ps["p"]] - self._ps["val"]
        for row, mf_col, fn in self._ms_list:
            g[row] = phi_cur[mf_col] + fn(elapsed_time)

        # Passive tubes
        t = self._t
        if "row0" in t:
            mf0, p0, p1 = phi_cur[t["mf0"]], phi_cur[t["p0"]], phi_cur[t["p1"]]
            rho = self._rho_factor * np.maximum(0.5 * (p0 + p1), p_min)
            r_over_rho = t["rlam"] / rho
            dp_term = t["rlam"] / (rho ** 2) * (0.5 * self._rho_factor) * mf0
            g[t["row0"]] = (p0 - p1) - r_over_rho * mf0
            g[t["row1"]] = mf0 + phi_cur[t["mf1"]]
            H[t["row0"], t["mf0"]] = -r_over_rho
            H[t["row0"], t["p0"]] = 1.0 + dp_term
            H[t["row0"], t["p1"]] = -1.0 + dp_term

        # Active modulated restrictors
        m = self._m
        if "row0" in m:
            mf0, p0, p1, pc = phi_cur[m["mf0"]], phi_cur[m["p0"]], phi_cur[m["p1"]], phi_cur[m["pc"]]
            x = np.clip(m["a"] * (pc - m["pth"]), -500.0, 500.0)
            sig = 1.0 / (1.0 + np.exp(-x))
            s = m["gmax"] * sig
            ds_dpc = m["gmax"] * m["a"] * sig * (1.0 - sig)
            delta_p = p0 - p1
            rho = self._rho_factor * np.maximum(0.5 * (p0 + p1), p_min)
            r_over_rho = m["rlam"] / rho
            dp_term = m["rlam"] / (rho ** 2) * (0.5 * self._rho_factor) * mf0
            ja_pc = ds_dpc * delta_p
            g[m["row0"]] = s * delta_p - r_over_rho * mf0
            g[m["row1"]] = mf0 + phi_cur[m["mf1"]]
            g[m["row2"]] = phi_cur[m["mfc"]]
            H[m["row0"], m["mf0"]] = -r_over_rho
            H[m["row0"], m["p0"]] = s + dp_term
            H[m["row0"], m["p1"]] = -s + dp_term
            H[m["row0"], m["pc"]] = np.where(np.abs(ja_pc) > 1e-3, ja_pc, 0.0)

        # Capacitive chambers
        c = self._c
        if "rows" in c:
            if propagate:
                mf_val = 0.5 * (phi_prev[c["mf"]] + phi_cur[c["mf"]]) if tustin else phi_cur[c["mf"]]
                m_new = state_arr[c["ks"]] + dt * mf_val
                H[c["rows"], c["mf"]] = (0.5 * P_ATM * dt) if tustin else (P_ATM * dt)
            else:
                m_new = state_arr[c["ks"]]
                H[c["rows"], c["mf"]] = 0.0
            g[c["rows"]] = m_new * P_ATM - c["mass_stp"] * phi_cur[c["p"]]


# =============================================================================
# 3. Network Builders (System, Neuron, Synapse, Stimulus)
# =============================================================================

@dataclass
class Neuron:
    """Container for the fluidic components making up one EI neuron."""
    suffix: str
    parts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.parts[key]

    @property
    def f(self) -> Any:
        """Fast membrane pressure container (e_m)."""
        return self.parts["f"]

    @property
    def s(self) -> Any:
        """Slow recovery pressure container (e_s)."""
        return self.parts["s"]


@dataclass
class Synapse:
    """Container for an inhibitory restrictor synapse between two neurons."""
    suffix: str
    pre: Neuron
    post: Neuron
    parts: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.parts[key]


def make_base_system(fluid=fons.air) -> fons.System:
    """Create a FONSim system containing the shared pressure sources (I, L, E)."""
    system = fons.System()
    for name, p_val in SOURCES.items():
        system += fons.PressureSource(f"source_{name}", fluid, pressure=p_val)
    return system


def build_neuron(
    system: fons.System,
    suffix: str = "0",
    fluid=fons.air,
    **overrides,
) -> Neuron:
    """Instantiate and wire an EI fluidic neuron inside ``system``."""
    cfg = {**NEURON_PARAMS, **overrides}

    container_f = FastContainer(f"container_f_{suffix}", fluid=fluid, volume=cfg["volume_f"])
    container_s = FastContainer(f"container_s_{suffix}", fluid=fluid, volume=cfg["volume_s"])
    for container in (container_f, container_s):
        container.states[0].initial_value = container.mass_stp * cfg["p_init"] / P_ATM

    tube_L = FastCircularTube(f"tube_L_{suffix}", fluid=fluid, diameter=cfg["d_L"], length=cfg["L_L"])
    tube_s = FastCircularTube(f"tube_s_{suffix}", fluid=fluid, diameter=cfg["d_s"], length=cfg["L_s"])

    valve_E = FastModulatedRestrictor(
        f"gating_valve_E_{suffix}",
        fluid=fluid,
        diameter=cfg["d_E"],
        length=cfg["L_E"],
        g_max=cfg["g_max_E"],
        a=cfg["a_E"],
        p_th=cfg["p_th_E"],
    )
    valve_I = FastModulatedRestrictor(
        f"gating_valve_I_{suffix}",
        fluid=fluid,
        diameter=cfg["d_I"],
        length=cfg["L_I"],
        g_max=cfg["g_max_I"],
        a=cfg["a_I"],
        p_th=cfg["p_th_I"],
    )

    for comp in (container_f, container_s, tube_L, tube_s, valve_E, valve_I):
        system += comp

    # Leak pathway: container_f <-> tube_L <-> source_L
    system.connect(container_f, tube_L)
    system.connect(tube_L, "source_L")

    # Slow recovery pathway: container_f <-> tube_s <-> container_s
    system.connect(container_f, tube_s)
    system.connect(container_s, tube_s)

    # Excitatory pathway: container_f <-> valve_E <-> source_E (gated by e_m)
    system.connect(container_f, valve_E)
    system.connect("source_E", valve_E)
    system.connect((valve_E.label, "c"), container_f)

    # Inhibitory pathway: container_f <-> valve_I <-> source_I (gated by e_s)
    system.connect(container_f, valve_I)
    system.connect("source_I", valve_I)
    system.connect((valve_I.label, "c"), container_s)

    parts = {
        "f": container_f,
        "s": container_s,
        "tube_L": tube_L,
        "tube_s": tube_s,
        "E": valve_E,
        "I": valve_I,
    }
    return Neuron(suffix=suffix, parts=parts, config=cfg)


def build_synapse(
    system: fons.System,
    pre: Neuron,
    post: Neuron,
    suffix: str = "A_to_B",
    fluid=fons.air,
    **overrides,
) -> Synapse:
    """Wire an inhibitory restrictor synapse from ``pre.s`` (e_s) onto ``post.f`` (e_m)."""
    cfg = {**SYNAPSE_PARAMS, **overrides}

    synapse_I = FastModulatedRestrictor(
        f"synapse_I_{suffix}",
        fluid=fluid,
        diameter=cfg["d_syn_I"],
        length=cfg["L_syn_I"],
        g_max=cfg["g_max_syn_I"],
        a=cfg["a_syn_I"],
        p_th=cfg["p_th_syn_I"],
    )
    system += synapse_I
    system.connect(post.f, synapse_I)
    system.connect("source_I", synapse_I)
    system.connect((synapse_I.label, "c"), pre.s)

    return Synapse(suffix=suffix, pre=pre, post=post, parts={"I": synapse_I}, config=cfg)


def add_stimulus(
    system: fons.System,
    target: Neuron,
    steps: list[tuple[float, float]],
    amplitude: float = 1.0,
    label: str = "stim",
    fluid=fons.air,
) -> dict[str, Any]:
    """Attach a mass-flow stimulus source and coupling tube to ``target.f``."""
    wave_fn = fast_custom_wave(steps, amplitude)
    source_stim = fons.MassflowSource(f"source_{label}", fluid=fluid, massflow=wave_fn)
    tube_stim = FastCircularTube(f"tube_{label}", fluid, diameter=STIM_TUBE["d"], length=STIM_TUBE["L"])

    system += source_stim
    system += tube_stim
    system.connect(source_stim, tube_stim)
    system.connect(tube_stim, target.f)

    return {"source": source_stim, "tube": tube_stim, "steps": steps, "amplitude": amplitude}


def run_simulation(
    system: fons.System,
    duration: float,
    step: tuple[float, float] = (1e-5, 1e-2),
) -> tuple[FastSimulation, float]:
    """Execute the simulation and return (sim, runtime_s)."""
    sim = FastSimulation(system, duration=duration, step=step)
    t0 = time.perf_counter()
    sim.run()
    runtime_s = time.perf_counter() - t0
    return sim, runtime_s


def branch_signals(comp, rho_stp: float = fons.air.rho_stp) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pressure drop dP [Pa], volumetric flow Q [m^3/s], and dissipated power P [W]."""
    p_a = np.asarray(comp.get_all("pressure", "a")[0], dtype=float)
    p_b = np.asarray(comp.get_all("pressure", "b")[0], dtype=float)
    mf_a = np.asarray(comp.get_all("massflow", "a")[0], dtype=float)
    rho = rho_stp * np.maximum(0.5 * (p_a + p_b), 0.1 * P_ATM) / P_ATM
    dp = p_a - p_b
    q = mf_a / rho
    return dp, q, dp * q


# =============================================================================
# 4. Publication Plotting Style & Export Utilities
# =============================================================================

# Colorblind-safe / Matlab- & LaTeX-friendly palette matching the paper figures
COLORS: dict[str, str] = {
    "blue": "#0072BD",
    "orange": "#D95319",
    "green": "#009E73",
    "cyan": "#56B4E9",
    "purple": "#CC79A7",
    "gold": "#E69F00",
    "black": "#000000",
}


def apply_paper_style() -> None:
    """Configure Matplotlib to match the clean serif publication style."""
    mpl.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linewidth": 0.6,
        "axes.linewidth": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def save_csv(path: Path, headers: list[str], rows) -> Path:
    """Write rows to ``path`` (creating parent directories if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"Saved CSV:    {path.as_posix()}")
    return path


def save_fig(fig: plt.Figure, path: Path, show: bool = False) -> Path:
    """Save figure as PNG (and PDF alongside it) and optionally display it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved figure: {path.as_posix()} (and {pdf_path.name})")
    if show:
        plt.show()
    return path
