from pathlib import Path

import numpy as np

from carfollowing.datasets.ngsim import (load_ngsim, extract_following_events, NGSIM_FPS,
                                 FT_TO_M)
from carfollowing.models import IDM
from carfollowing.simulate import LeaderTrace, simulate_following


def _write_fixture(path: Path, t_end=30.0):
    dt = 1.0 / NGSIM_FPS
    t = np.arange(0.0, t_end + 1e-9, dt)
    v_lead = 20.0 + 2.0 * np.sin(0.3 * t)
    lead = LeaderTrace.from_speed_profile(t, v_lead)
    m = IDM(v0=28.0, T=1.2)
    res = simulate_following(m, lead, v0=float(lead.v[0]), dt=dt, t_end=t_end)
    rows = ["Vehicle_ID\tFrame_ID\tGlobal_Time\tLocal_X\tLocal_Y\tv_Vel\tv_Acc\t"
            "Lane_ID\tPreceding\tFollowing"]
    FT = 1.0 / 0.3048
    for k in range(len(t)):
        rows.append(f"1\t{k}\t{int(t[k]*1000)}\t{lead.x[k]*FT:.4f}\t0\t"
                    f"{lead.v[k]*FT:.4f}\t0\t1\t0\t2")
        rows.append(f"2\t{k}\t{int(t[k]*1000)}\t{res.x[k]*FT:.4f}\t0\t"
                    f"{res.v[k]*FT:.4f}\t0\t1\t1\t0")
    path.write_text("\n".join(rows), encoding="utf-8")
    return lead, res


def test_load_converts_units(tmp_path):
    path = tmp_path / "ngsim_fixture.txt"
    _write_fixture(path)
    df = load_ngsim(path)
    assert df["vid"].nunique() == 2
    # time in seconds, realistic Highway speeds and positions
    assert df["t"].max() == 30.0
    assert 15.0 < df["v"].max() < 30.0
    assert df["x"].max() > 100.0


def test_extract_following_event(tmp_path):
    path = tmp_path / "ngsim_fixture.txt"
    _write_fixture(path, t_end=40.0)
    df = load_ngsim(path)
    events = extract_following_events(df, min_duration=20.0, gap_range=(2.0, 80.0))
    assert len(events) >= 1
    ev = events[0]
    assert ev.follower_id == 2 and ev.leader_id == 1
    assert len(ev.t) > 100
    assert np.all(ev.gap > 0)
    lt = ev.leader_trace()
    assert lt.t.shape == ev.t.shape


def test_missing_file_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_ngsim(tmp_path / "does_not_exist.txt")
