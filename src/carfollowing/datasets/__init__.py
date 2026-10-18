"""Dataset adapters for carfollowing.

* :mod:`carfollowing.datasets.synthetic` — reproducible heterogeneous-platoon benchmark
* :mod:`carfollowing.datasets.ngsim`     — NGSIM loader and car-following event extractor
"""

from __future__ import annotations

from .synthetic import (PlatoonData, realistic_speed_profile, generate_platoon,
                        IDM_PRIORS)
from .ngsim import (FollowingEvent, load_ngsim, extract_following_events,
                    ngsim_available, NGSIM_FPS)

__all__ = [
    "PlatoonData", "realistic_speed_profile", "generate_platoon", "IDM_PRIORS",
    "FollowingEvent", "load_ngsim", "extract_following_events",
    "ngsim_available", "NGSIM_FPS",
]
