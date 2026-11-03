# Fluidic Neuromorphic Circuits — Simulation Code

Self-contained simulation scripts and helper utilities accompanying the paper on port-Hamiltonian / bond-graph fluidic neuromorphic circuits (spiking EI neuron and half-centre oscillator).

All internal thermodynamic states are solved in absolute pressure using [FONSim](https://fonsim.readthedocs.io/), while all figures and exported CSV datasets report pressures in **gauge pressure** ($P_{\text{gauge}} = P - P_{\text{atm}}$, with $P_{\text{atm}} = 101\,300\text{ Pa}$).

---

## Repository Structure

```text
REWORKED/
├── helpers.py        # Shared physical parameters, fast vectorised solver, network builders, and plotting helpers
├── ECC_neuron.py     # Single spiking EI fluidic neuron simulation & power dissipation analysis (Fig. 4)
├── ECC_hco.py        # Mutually inhibitory two-neuron half-centre oscillator (HCO) simulation (Fig. 5b)
├── data/             # Exported CSV time-series data (gauge pressure)
└── figures/          # Generated publication figures (.png and .pdf)
```

---

## Requirements

- Python 3.10+
- `fonsim` (`0.6.0`)
- `numpy`
- `matplotlib`

Install dependencies via:

```bash
pip install fonsim numpy matplotlib
```

---

## Usage

Run either standalone script directly from the terminal or an IDE:

```bash
# 1. Single spiking EI neuron (generates figures/ECC_neuron_fig3.{png,pdf} and data/ECC_neuron.csv)
python ECC_neuron.py

# 2. Half-centre oscillator (generates figures/ECC_hco_fig4b.{png,pdf} and data/ECC_hco.csv)
python ECC_hco.py
```

### Configuration Flags

All simulation settings and parameter overrides are exposed as constants at the top of [`ECC_neuron.py`](ECC_neuron.py) and [`ECC_hco.py`](ECC_hco.py) (no command-line arguments required):

- `DURATION`, `T_WINDOW`, `STEP`: Simulation time horizon and adaptive time-step bounds.
- `T_SUPER`, `A_SUPER`, `T_SUB`, `A_SUB` (in `ECC_neuron.py`): Onset times and mass-flow amplitudes of the supra- and sub-threshold input pulses.
- `T_KICK`, `KICK_WIDTH`, `KICK_AMP` (in `ECC_hco.py`): Symmetry-breaking input pulse applied to neuron A.
- `D_S`, `D_S_A`, `D_S_B`: Direct overrides for the slow recovery restrictor (`tube_s`) diameter $d_s$ [m].
- `SAVE_FIGURES`, `SHOW_FIGURES`, `EXPORT_CSV`: Toggle figure saving, interactive plot display, and CSV data export.
