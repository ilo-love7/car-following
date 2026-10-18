# Car-Following Traffic Modelling Toolkit

A self-contained, reproducible microscopic traffic modelling library with 7 classic car-following models, stability analysis, calibration, trajectory prediction, lane changing, and EV energy consumption.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 46 passing](https://img.shields.io/badge/tests-46%20passing-brightgreen.svg)](./tests)

## Features

- **7 car-following models** with a unified interface: IDM, OVM, FVD, Helly/ACC, GHR, Gipps, Newell
- **Fundamental diagram & capacity** derived analytically from equilibrium gaps
- **Linear stability** with closed-form criterion, cross-validated against ring-road eigenvalues
- **String stability** for Helly/ACC with analytic transfer function
- **Bounded least-squares calibration** with noise-robustness analysis
- **Trajectory prediction** with CV / CA / IDM / LSTM baselines
- **MOBIL lane-changing** two-lane microsimulation
- **EV energy consumption** model with regeneration
- **NGSIM data adapter** with synthetic fixture for end-to-end validation

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR-USERNAME/car-following.git
cd car-following

# Install
pip install -r requirements.txt

# Run all experiments (CPU only)
python scripts/run_all.py

# Run tests
python -m pytest tests -q
```

## Usage

```python
from carfollowing import IDM, make_model
from carfollowing.simulate import simulate_following
from carfollowing.stability import linear_stability

m = IDM(v0=30, T=1.2, s0=2.0, a=1.5, b=2.0)
print(m.equilibrium_gap(20.0))          # equilibrium gap
print(linear_stability(m, 20.0))        # linear stability margin
```

## Repository Structure

```
car-following/
├── src/carfollowing/          # Core library
│   ├── models.py              # 7 car-following models
│   ├── simulate.py            # Single-follower & ring-road simulation
│   ├── equilibrium.py         # Microscopic fundamental diagram, capacity
│   ├── stability.py           # Linear, ring & string stability
│   ├── calibration.py         # Bounded non-linear least-squares calibration
│   ├── metrics.py             # Accuracy & safety indicators
│   ├── energy.py              # Longitudinal EV energy model
│   ├── lanechange.py          # MOBIL lane changing & two-lane microsim
│   └── datasets/
│       ├── synthetic.py       # Reproducible synthetic benchmark
│       └── ngsim.py           # NGSIM loader & event extraction
├── scripts/                   # Experiment scripts + run_all.py
├── tests/                     # 46 pytest cases
├── data/                      # Generated CSV / JSON / fixtures
├── figures/                   # Result figures
├── reports/                   # Technical report, acceptance checklist, data dictionary
├── README.md
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Reports

- `reports/技术报告.md` — Full technical report
- `reports/验收核查表.md` — Acceptance checklist
- `reports/数据来源与数据字典.md` — Data sources & dictionary

## License

MIT License. See `LICENSE` for details.
