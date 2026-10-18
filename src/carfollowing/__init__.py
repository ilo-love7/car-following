"""carfollowing — car-following modelling toolkit.

A small, self-contained library for longitudinal traffic modelling:

* :mod:`carfollowing.models`      — IDM, OVM, FVD, Helly/ACC, GHR, Gipps, Newell
* :mod:`carfollowing.simulate`    — single-follower and ring-road simulation
* :mod:`carfollowing.equilibrium` — microscopic fundamental diagram, capacity
* :mod:`carfollowing.stability`   — linear, ring-eigenvalue and string stability
* :mod:`carfollowing.calibration` — bounded non-linear least-squares calibration
* :mod:`carfollowing.metrics`     — accuracy and safety indicators
* :mod:`carfollowing.energy`      — longitudinal EV energy model
* :mod:`carfollowing.lanechange`  — MOBIL lane changing and a two-lane microsim
* :mod:`carfollowing.datasets`    — synthetic benchmark + NGSIM adapter
"""

from __future__ import annotations

from . import (models, simulate, equilibrium, stability, calibration,
               metrics, energy, lanechange, viz, datasets)
from .models import (CarFollowingModel, IDM, OVM, FVD, Helly, GHR, Gipps,
                     Newell, MODEL_REGISTRY, make_model, DEFAULT_PARAMS)
from .simulate import LeaderTrace, simulate_following, simulate_ring, FollowResult
from .equilibrium import fundamental_diagram, capacity_point, ring_length_for
from .stability import (linear_stability, ring_eigenvalues, max_ring_growth,
                        helly_string_stability, helly_string_stability_condition,
                        numeric_string_stability)
from .calibration import calibrate_following, CalibrationResult
from .metrics import safety_report, time_to_collision, time_headway, drac
from .energy import EVParams, energy_from_speed, compare_styles

__version__ = "1.0.0"

__all__ = [
    "models", "simulate", "equilibrium", "stability", "calibration",
    "metrics", "energy", "lanechange", "viz", "datasets",
    "CarFollowingModel", "IDM", "OVM", "FVD", "Helly", "GHR", "Gipps", "Newell",
    "MODEL_REGISTRY", "make_model", "DEFAULT_PARAMS",
    "LeaderTrace", "simulate_following", "simulate_ring", "FollowResult",
    "fundamental_diagram", "capacity_point", "ring_length_for",
    "linear_stability", "ring_eigenvalues", "max_ring_growth",
    "helly_string_stability", "helly_string_stability_condition",
    "numeric_string_stability",
    "calibrate_following", "CalibrationResult",
    "safety_report", "time_to_collision", "time_headway", "drac",
    "EVParams", "energy_from_speed", "compare_styles",
    "__version__",
]
