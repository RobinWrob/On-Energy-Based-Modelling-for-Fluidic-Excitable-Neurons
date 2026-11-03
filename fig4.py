"""
Spiking EI Fluidic Neuron Simulation (Fig. 4).

Simulates a single fluidic EI neuron driven by two 50 ms input flow pulses:
  - t = 2.0 s : supra-threshold pulse (12 mg/s) -> triggers all-or-nothing spike
  - t = 5.0 s : sub-threshold pulse   (10 mg/s) -> passive overdamped relaxation

Computes instantaneous power dissipation P_x = dP_x * Q_x >= 0 across all
resistive and memductive branches, exports gauge-pressure traces to CSV, and
generates Figure 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from helpers import (  # noqa: E402
    COLORS,
    MW,
    add_stimulus,
    apply_paper_style,
    branch_signals,
    build_neuron,
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

DURATION = 6.0            # Total simulation duration [s]
T_WINDOW = (1.0, 6.0)     # Plotted / exported time window [s] (after initial settling)
STEP = (1e-5, 1e-2)       # Adaptive solver step bounds (min_dt, max_dt) [s]

# Input flow pulse parameters
PULSE_WIDTH = 0.05        # Pulse duration [s] (50 ms)
T_SUPER = 2.0             # Supra-threshold pulse onset [s]
A_SUPER = 12e-6           # Supra-threshold amplitude [kg/s] (12 mg/s)
T_SUB = 5.0               # Sub-threshold pulse onset [s]
A_SUB = 10e-6             # Sub-threshold amplitude [kg/s] (10 mg/s)

# Optional neuron parameter overrides (e.g. D_S = 3e-4 for tube_s diameter [m])
D_S: float | None = None
NEURON_OVERRIDES: dict = {}

# Output controls
SAVE_FIGURES = True
SHOW_FIGURES = False
EXPORT_CSV = True
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"


# =============================================================================
# Simulation & Analysis
# =============================================================================

def simulate_spiking_neuron() -> dict[str, np.ndarray]:
    """Build and simulate the single EI neuron with supra- and sub-threshold pulses."""
    overrides = dict(NEURON_OVERRIDES)
    if D_S is not None:
        overrides["d_s"] = D_S

    system = make_base_system()
    neuron = build_neuron(system, suffix="0", **overrides)

    steps = [
        (0.0, 0.0),
        (T_SUPER, A_SUPER),
        (T_SUPER + PULSE_WIDTH, 0.0),
        (T_SUB, A_SUB),
        (T_SUB + PULSE_WIDTH, 0.0),
    ]
    stim = add_stimulus(system, target=neuron, steps=steps, amplitude=1.0, label="stim")

    sim, runtime_s = run_simulation(system, duration=DURATION, step=STEP)
    t_all = np.asarray(sim.times, dtype=float)
    mask = (t_all >= T_WINDOW[0]) & (t_all <= T_WINDOW[1])
    print(f"[ECC_neuron] Simulated {len(t_all)} steps in {runtime_s:.2f} s")

    def crop(arr) -> np.ndarray:
        return np.asarray(arr, dtype=float)[mask]

    t = crop(t_all)
    q_in = -crop(stim["source"].get_all("massflow")[0])  # [kg/s]
    e_m_abs = crop(neuron.f.get_all("pressure")[0])      # [Pa] (absolute)
    e_s_abs = crop(neuron.s.get_all("pressure")[0])      # [Pa] (absolute)

    # Branch power dissipations P_x = dP_x * Q_x [W]
    _, _, p_E = (crop(v) for v in branch_signals(neuron["E"]))
    _, _, p_I = (crop(v) for v in branch_signals(neuron["I"]))
    _, _, p_L = (crop(v) for v in branch_signals(neuron["tube_L"]))
    _, _, p_s = (crop(v) for v in branch_signals(neuron["tube_s"]))
    _, _, p_stim = (crop(v) for v in branch_signals(stim["tube"]))

    min_power_w = min(float(np.min(p)) for p in (p_E, p_I, p_L, p_s, p_stim))
    print(f"[ECC_neuron] Minimum branch dissipation across all branches: {min_power_w * MW + 0.0:.2e} mW (>= 0)")

    return {
        "t": t,
        "q_in_kg_s": q_in,
        "e_m_gauge_pa": to_gauge_pa(e_m_abs),
        "e_s_gauge_pa": to_gauge_pa(e_s_abs),
        "e_m_gauge_bar": to_gauge_bar(e_m_abs),
        "e_s_gauge_bar": to_gauge_bar(e_s_abs),
        "P_E_W": p_E,
        "P_s_W": p_s,
        "P_I_W": p_I,
        "P_stim_W": p_stim,
        "P_L_W": p_L,
    }


# =============================================================================
# Plotting (Fig. 4)
# =============================================================================

def plot_spiking_figure(data: dict[str, np.ndarray]) -> plt.Figure:
    """Create the 3-panel publication figure (Fig. 4) in gauge pressure."""
    from matplotlib.ticker import FormatStrFormatter

    apply_paper_style()
    t = data["t"]

    fig = plt.figure(figsize=(5.4, 7.1))
    # 5-row layout: [panel (a), gap_ab, panel (b), gap_bc (legend + caption), panel (c)]
    gs = fig.add_gridspec(5, 1, height_ratios=[1.15, 0.34, 1.35, 0.78, 0.55], hspace=0.0)
    ax_p = fig.add_subplot(gs[0])
    ax_w = fig.add_subplot(gs[2])
    ax_q = fig.add_subplot(gs[4])

    # Shade the 50 ms input pulse intervals across all panels
    for ax in (ax_p, ax_w, ax_q):
        for t_on in (T_SUPER, T_SUB):
            ax.axvspan(t_on, t_on + PULSE_WIDTH, color="#cccccc", alpha=0.65, lw=0, zorder=1)
        ax.set_xlim(T_WINDOW)
        ax.set_xticks([1, 2, 3, 4, 5, 6])

    # --- (a) Membrane (e_m) and recovery (e_s) gauge pressures ---
    ax_p.plot(t, data["e_m_gauge_bar"], color=COLORS["blue"], lw=1.4, label=r"$e_m$", zorder=3)
    ax_p.plot(t, data["e_s_gauge_bar"], color=COLORS["orange"], lw=1.4, label=r"$e_s$", zorder=3)
    ax_p.set_ylabel("Pressure [bar]")
    ax_p.set_ylim(-0.02, 1.05)
    ax_p.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_p.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax_p.set_xticklabels([])
    ax_p.legend(loc="upper right", ncol=2, columnspacing=1.2, handlelength=1.8)
    ax_p.text(
        0.5, -0.13,
        r"(a) Membrane ($e_m$) and recovery ($e_s$) pressures.",
        transform=ax_p.transAxes, ha="center", va="top", fontsize=10.5,
    )

    # --- (b) Branch power dissipation (P_x >= 0) ---
    h_E, = ax_w.plot(t, data["P_E_W"] * MW, color=COLORS["green"], lw=1.2, label=r"$\mathcal{P}_E$", zorder=3)
    h_s, = ax_w.plot(t, data["P_s_W"] * MW, color=COLORS["cyan"], lw=1.2, label=r"$\mathcal{P}_s$", zorder=3)
    h_I, = ax_w.plot(t, data["P_I_W"] * MW, color=COLORS["purple"], lw=1.2, label=r"$\mathcal{P}_I$", zorder=3)
    h_stim, = ax_w.plot(t, data["P_stim_W"] * MW, color=COLORS["black"], lw=1.2, label=r"$\mathcal{P}_{\mathrm{stim}}$", zorder=4)
    h_L, = ax_w.plot(t, data["P_L_W"] * MW, color=COLORS["gold"], lw=1.2, label=r"$\mathcal{P}_L$", zorder=3)

    ax_w.axhline(0.0, color="black", lw=0.8, zorder=2)
    ax_w.set_yscale("symlog", linthresh=1e-2)
    ax_w.set_ylim(-1e-2, 2e5)
    ax_w.set_yticks([-1e-2, 0.0, 1e-2, 1e0, 1e2, 1e4])
    ax_w.axhspan(-1e-2, 0.0, facecolor="#ffd6d6", edgecolor="#f08080", alpha=0.7, zorder=1)
    h_neg = Patch(facecolor="#ffd6d6", edgecolor="#f08080", label=r"$\mathcal{P} < 0$")

    ax_w.set_ylabel(r"$\mathcal{P}_x$ [mW]")
    ax_w.set_xticklabels([])
    ax_w.legend(
        handles=[h_E, h_stim, h_s, h_L, h_I, h_neg],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        columnspacing=1.8,
        handlelength=1.8,
    )
    ax_w.text(
        0.5, -0.44,
        r"(b) Branch power dissipation ($\mathcal{P}_x \geq 0$).",
        transform=ax_w.transAxes, ha="center", va="top", fontsize=10.5,
    )

    # --- (c) Input stimulus flow pulses ---
    ax_q.plot(t, data["q_in_kg_s"] * 1e6, color=COLORS["black"], lw=1.3, zorder=3)
    ax_q.set_ylabel(r"$Q_{\mathrm{in}}$ [mg/s]")
    ax_q.set_xlabel("Time [s]")
    ax_q.set_ylim(-1.0, 14.0)
    ax_q.set_yticks([0, 5, 10])
    ax_q.text(
        0.5, -0.58,
        "(c) Input stimulus flow pulses.",
        transform=ax_q.transAxes, ha="center", va="top", fontsize=10.5,
    )

    return fig


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    data = simulate_spiking_neuron()

    if EXPORT_CSV:
        headers = [
            "time_s",
            "Q_in_kg_s",
            "e_m_gauge_Pa",
            "e_s_gauge_Pa",
            "e_m_gauge_bar",
            "e_s_gauge_bar",
            "P_E_W",
            "P_s_W",
            "P_I_W",
            "P_stim_W",
            "P_L_W",
        ]
        rows = zip(*(data[k] for k in (
            "t", "q_in_kg_s", "e_m_gauge_pa", "e_s_gauge_pa",
            "e_m_gauge_bar", "e_s_gauge_bar",
            "P_E_W", "P_s_W", "P_I_W", "P_stim_W", "P_L_W",
        )))
        save_csv(DATA_DIR / "ECC_fig4.csv", headers, rows)

    if SAVE_FIGURES or SHOW_FIGURES:
        fig = plot_spiking_figure(data)
        save_fig(fig, FIG_DIR / "ECC_fig4.png", show=SHOW_FIGURES)


if __name__ == "__main__":
    main()
