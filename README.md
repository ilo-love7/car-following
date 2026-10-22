<div align="center">

# Car-Following Traffic Modelling Toolkit

**Seven classic car-following models with stability analysis, calibration and trajectory prediction — one reproducible Python package.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 46 passing](https://img.shields.io/badge/tests-46%20passing-brightgreen.svg)](./tests)

<p align="center">
  <b>English</b> · <a href="./README.zh-CN.md">简体中文</a>
</p>

</div>

---

A self-contained, reproducible microscopic traffic modelling library: 7 classic car-following models behind one interface, with fundamental-diagram derivation, linear/ring/string stability analysis, bounded calibration, trajectory prediction, MOBIL lane changing and EV energy consumption.

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

## Quick start

```bash
# Clone
git clone https://github.com/ilo-love7/car-following.git
cd car-following

# Install
pip install -r requirements.txt

# Run all experiments (CPU only)
python scripts/run_all.py

# Run tests
python -m pytest tests -q
```

Expected output of the test run:

```
46 passed
```

## What you can do with this toolkit

Six concrete jobs, each with a runnable entry point (CPU only; no GPU and no private data):

| Job | Entry point |
|---|---|
| **Compare seven car-following models under identical conditions** (IDM, OVM, FVD, Helly, Newell, …) | `python scripts/run_all.py case1` · `carfollowing.models.make_model` |
| **Check linear / string stability and locate capacity** | `carfollowing.stability.linear_stability`, `carfollowing.equilibrium.capacity_point` · `scripts/run_all.py case2 case3` |
| **Calibrate a model against real trajectories (NGSIM)** | `carfollowing.calibration` · `scripts/run_all.py case4` |
| **Predict short-horizon trajectories and score them** | `carfollowing.metrics` · `scripts/run_all.py case5` |
| **Lane-changing with MOBIL** | `carfollowing.lanechange` · `scripts/run_all.py case6` |
| **EV energy consumption from a following trace** | `carfollowing.energy` · `scripts/run_all.py case7` |

Typical uses: (1) a **baseline** to compare a new car-following model against;
(2) a **reproducibility harness** for calibration papers; (3) **teaching material** for traffic-flow
courses; (4) a **sanity check** before putting ACC/ADAS logic into a simulator.

Everything runs offline: `python -m pytest tests -q` (46 tests) needs no datasets, and CI covers
both Linux and Windows.

## Usage

```python
from carfollowing import IDM, make_model
from carfollowing.simulate import simulate_following
from carfollowing.stability import linear_stability

m = IDM(v0=30, T=1.2, s0=2.0, a=1.5, b=2.0)
print(m.equilibrium_gap(20.0))          # equilibrium gap
print(linear_stability(m, 20.0))        # linear stability margin
```

## Repository structure

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

- [`reports/技术报告.md`](reports/技术报告.md) — full technical report
- [`reports/验收核查表.md`](reports/验收核查表.md) — acceptance checklist
- [`reports/数据来源与数据字典.md`](reports/数据来源与数据字典.md) — data sources & dictionary

*Report filenames are in Chinese.*

## License

MIT License. See [`LICENSE`](LICENSE) for details.
