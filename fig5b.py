"""
Fluidic Half-Centre Oscillator (HCO) Simulation (Fig. 5b).

Simulates two identical fluidic EI neurons (A and B) coupled via mutually
inhibitory restrictor synapses gated by their slow recovery pressures (e_s,A
and e_s,B). A transient 50 ms flow pulse (20 mg/s) applied to neuron A at
t = 2.0 s breaks symmetry and initiates sustained anti-phase bursting.

Exports gauge-pressure time series to CSV and generates Figure 5b.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402
import numpy as np  # noqa: E402

from helpers import (  # noqa: E402
    COLORS,
    add_stimulus,
    apply_paper_style,
    build_neuron,
    build_synapse,
    make_base_system,
    run_simulation,
    save_csv,
    save_fig,
    to_gauge_bar,
    to_gauge_pa,
)

# =============================================================================
# User Configuration Flags
# =============================================================================

DURATION = 10.0           # Total simulation duration [s]
STEP = (1e-5, 1e-2)       # Adaptive solver step bounds (min_dt, max_dt) [s]

# Symmetry-breaking kick stimulus on neuron A
T_KICK = 2.0              # Kick onset time [s]
KICK_WIDTH = 0.05         # Kick pulse duration [s] (50 ms)
KICK_AMP = 20e-6          # Kick flow amplitude [kg/s] (20 mg/s)

# Optional slow-tube diameter overrides (tube_s, default 3e-4 m):
#   - Set D_S to override tube_s diameter for BOTH neurons A and B.
#   - Set D_S_A or D_S_B to override neuron A or B individually.
D_S: float | None = None
D_S_A: float | None = None
D_S_B: float | None = None

# Additional component parameter overrides
NEURON_OVERRIDES_A: dict = {}
NEURON_OVERRIDES_B: dict = {}
SYNAPSE_OVERRIDES: dict = {}

# Output controls
SAVE_FIGURES = True
SHOW_FIGURES = False
EXPORT_CSV = True
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"


# =============================================================================
# Simulation
# =============================================================================

def simulate_hco() -> dict[str, np.ndarray]:
    """Build and simulate the two-neuron mutually inhibitory HCO network."""
    ov_a = dict(NEURON_OVERRIDES_A)
    ov_b = dict(NEURON_OVERRIDES_B)
    if D_S is not None:
        ov_a["d_s"] = D_S
        ov_b["d_s"] = D_S
    if D_S_A is not None:
        ov_a["d_s"] = D_S_A
    if D_S_B is not None:
        ov_b["d_s"] = D_S_B

    system = make_base_system()
    neuron_a = build_neuron(system, suffix="A", **ov_a)
    neuron_b = build_neuron(system, suffix="B", **ov_b)

    # Reciprocal inhibitory synapses gated by presynaptic slow recovery pressure (e_s)
    syn_ab = build_synapse(system, pre=neuron_a, post=neuron_b, suffix="A_to_B", **SYNAPSE_OVERRIDES)
    syn_ba = build_synapse(system, pre=neuron_b, post=neuron_a, suffix="B_to_A", **SYNAPSE_OVERRIDES)

    # Transient flow pulse to neuron A at t = T_KICK
    steps = [
        (0.0, 0.0),
        (T_KICK, 1.0),
        (T_KICK + KICK_WIDTH, 0.0),
    ]
    stim_a = add_stimulus(system, target=neuron_a, steps=steps, amplitude=KICK_AMP, label="kick_A")

    sim, runtime_s = run_simulation(system, duration=DURATION, step=STEP)
    t = np.asarray(sim.times, dtype=float)
    print(
        f"[ECC_hco] Simulated {len(t)} steps in {runtime_s:.2f} s "
        f"(d_s_A={neuron_a.config['d_s']:.2e} m, d_s_B={neuron_b.config['d_s']:.2e} m)"
    )

    q_kick = -np.asarray(stim_a["source"].get_all("massflow")[0], dtype=float)
    em_a_abs = np.asarray(neuron_a.f.get_all("pressure")[0], dtype=float)
    es_a_abs = np.asarray(neuron_a.s.get_all("pressure")[0], dtype=float)
    em_b_abs = np.asarray(neuron_b.f.get_all("pressure")[0], dtype=float)
    es_b_abs = np.asarray(neuron_b.s.get_all("pressure")[0], dtype=float)

    g_syn_ab = syn_ab["I"].get_gating(es_a_abs)
    g_syn_ba = syn_ba["I"].get_gating(es_b_abs)

    return {
        "t": t,
        "q_kick_kg_s": q_kick,
        "em_A_gauge_pa": to_gauge_pa(em_a_abs),
        "es_A_gauge_pa": to_gauge_pa(es_a_abs),
        "em_B_gauge_pa": to_gauge_pa(em_b_abs),
        "es_B_gauge_pa": to_gauge_pa(es_b_abs),
        "em_A_gauge_bar": to_gauge_bar(em_a_abs),
        "es_A_gauge_bar": to_gauge_bar(es_a_abs),
        "em_B_gauge_bar": to_gauge_bar(em_b_abs),
        "es_B_gauge_bar": to_gauge_bar(es_b_abs),
        "g_syn_A_to_B": g_syn_ab,
        "g_syn_B_to_A": g_syn_ba,
    }


# =============================================================================
# Plotting (Fig. 5b)
# =============================================================================

def plot_hco_figure(data: dict[str, np.ndarray]) -> plt.Figure:
    """Create the HCO membrane-pressure traces figure (Fig. 5b) in gauge pressure."""
    apply_paper_style()
    t = data["t"]

    fig, ax = plt.subplots(figsize=(5.4, 2.35))
    ax.plot(t, data["em_A_gauge_bar"], color=COLORS["blue"], lw=1.4, label=r"$e_{m,A}$", zorder=3)
    ax.plot(t, data["em_B_gauge_bar"], color=COLORS["orange"], lw=1.4, label=r"$e_{m,B}$", zorder=3)

    ax.set_xlim(0.0, DURATION)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_ylim(-0.08, 1.08)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure [bar]")
    ax.legend(loc="upper right", ncol=1, handlelength=2.0)

    ax.text(
        0.5, -0.30,
        "(b) Membrane-pressure traces.",
        transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
    )

    return fig


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    data = simulate_hco()

    if EXPORT_CSV:
        headers = [
            "time_s",
            "Q_kick_kg_s",
            "em_A_gauge_Pa",
            "es_A_gauge_Pa",
            "em_B_gauge_Pa",
            "es_B_gauge_Pa",
            "em_A_gauge_bar",
            "es_A_gauge_bar",
            "em_B_gauge_bar",
            "es_B_gauge_bar",
            "g_syn_A_to_B",
            "g_syn_B_to_A",
        ]
        rows = zip(*(data[k] for k in (
            "t", "q_kick_kg_s",
            "em_A_gauge_pa", "es_A_gauge_pa", "em_B_gauge_pa", "es_B_gauge_pa",
            "em_A_gauge_bar", "es_A_gauge_bar", "em_B_gauge_bar", "es_B_gauge_bar",
            "g_syn_A_to_B", "g_syn_B_to_A",
        )))
        save_csv(DATA_DIR / "ECC_fig5b.csv", headers, rows)

    if SAVE_FIGURES or SHOW_FIGURES:
        fig = plot_hco_figure(data)
        save_fig(fig, FIG_DIR / "ECC_fig5b.png", show=SHOW_FIGURES)


if __name__ == "__main__":
    main()
