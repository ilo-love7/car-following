# Real trajectory data goes here (optional)

Drop real NGSIM trajectory files (`.txt` or `.csv`) into this folder and
`scripts/case8_ngsim.py` will use them automatically instead of the built-in
synthetic fixture.

## NGSIM

* Source: FHWA "Next Generation Simulation (NGSIM) Vehicle Trajectories"
  (data.gov), e.g. US-101, I-80, Lankershim Boulevard.
* Format: text, 10 Hz, US customary units (positions in ft, speed in ft/s).
* The loader expects the standard columns:

  `Vehicle_ID  Frame_ID  Global_Time  Local_X  Local_Y  v_Vel  v_Acc  Lane_ID  Preceding  Following`

* Units are converted to SI automatically by `carfollowing.datasets.ngsim.load_ngsim`.

## highD (alternative)

highD files are metric (m, m/s) and use different column names; a small column
mapping is all that is required. See `reports/数据来源与数据字典.md`.

> Real datasets are **not** redistributed with this repository (size / licence).
