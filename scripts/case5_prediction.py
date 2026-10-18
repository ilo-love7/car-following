"""Case 5 - short-horizon trajectory prediction.

Four predictors roll out the follower over a 10 s horizon while the leader's
future is assumed known (the standard car-following prediction setting):

* **CV**   constant velocity
* **CA**   constant acceleration
* **IDM**  physics rollout of the calibrated/mean IDM parameters
* **LSTM** a small recurrent net trained one-step-ahead on the synthetic data

Evaluation is leave-out: the net is trained on some platoon seeds and tested on
unseen seeds, and all models see the same test windows.
"""

from __future__ import annotations

import numpy as np

from _common import FIGS, banner, update_summary, write_csv, write_metrics
from carfollowing.datasets import generate_platoon
from carfollowing.models import IDM
from carfollowing.viz import new_fig, savefig, PALETTE

DT = 0.2
HIST = 20          # 4 s of history
HORIZON = 50       # 10 s prediction horizon


def _windows(pd_data, n_windows: int, rng) -> list[dict]:
    t = pd_data.t
    out = []
    for i in range(1, pd_data.n_vehicles):
        x = pd_data.trajectories[i]
        lead = pd_data.trajectories[i - 1]
        n = len(t)
        if n < HIST + HORIZON + 1:
            continue
        for _ in range(n_windows):
            k0 = rng.integers(0, n - HIST - HORIZON - 1)
            out.append({
                "v": x["v"][k0:k0 + HIST + HORIZON],
                "s": x["s"][k0:k0 + HIST + HORIZON],
                "x": x["x"][k0:k0 + HIST + HORIZON],
                "xl": lead["x"][k0:k0 + HIST + HORIZON],
                "vl": lead["v"][k0:k0 + HIST + HORIZON],
            })
    return out


# ---------------------------------------------------------------------------
def predict_cv(w, h=HORIZON):
    v = w["v"][HIST - 1]
    return np.full(h, v), _integrate_gap(w, np.full(h, v))


def predict_ca(w, h=HORIZON):
    a = (w["v"][HIST - 1] - w["v"][HIST - 2]) / DT
    steps = np.arange(1, h + 1)
    v = np.clip(w["v"][HIST - 1] + a * steps * DT, 0, 60)
    return v, _integrate_gap(w, v)


def _integrate_gap(w, v_pred, h=HORIZON):
    """Gap from d s/dt = v_lead - v_follower."""
    s = w["s"][HIST - 1]
    vl = w["vl"][HIST:HIST + h]
    vf = np.concatenate([[w["v"][HIST - 1]], v_pred[:-1]])
    gaps = [s]
    for k in range(h):
        s = s + (vl[k] - vf[k]) * DT
        gaps.append(s)
    return np.array(gaps[1:])


def predict_idm(w, h=HORIZON, model=None):
    if model is None:
        model = IDM(v0=28.0, T=1.2, s0=2.0, a=1.5, b=2.0)
    s = w["s"][HIST - 1]; v = w["v"][HIST - 1]
    vpred = []; gpred = []
    for k in range(h):
        vl = w["vl"][HIST + k]
        a = model.accel(max(s, 1e-3), v, v - vl)
        v = max(0.0, v + a * DT)
        s = s + (vl - v) * DT
        vpred.append(v); gpred.append(s)
    return np.array(vpred), np.array(gpred)


# ---------------------------------------------------------------------------
def _feat(v, s, vl):
    return [v, s, vl, vl - v]


def _train_lstm(train_windows, epochs: int = 200, hidden: int = 48, seed: int = 0):
    """Train a small LSTM to predict the one-step speed *increment* ``dv``."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    Tn = HIST + HORIZON - 1
    X = np.stack([np.array([_feat(w["v"][k], w["s"][k], w["vl"][k]) for k in range(Tn)],
                           dtype=np.float32) for w in train_windows])
    Y = np.stack([np.array([w["v"][k + 1] - w["v"][k] for k in range(Tn)],
                           dtype=np.float32) for w in train_windows])
    mu = X.reshape(-1, X.shape[-1]).mean(0)
    sd = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    Xn = (X - mu) / sd
    ysd = Y.std() + 1e-6
    Yn = Y / ysd

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(X.shape[-1], hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                      nn.Linear(hidden, 1))

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    Xt = torch.tensor(Xn); Yt = torch.tensor(Yn)
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xt), Yt)
        loss.backward(); opt.step()
    net.eval()
    return net, mu, sd, ysd


def predict_lstm(w, net, mu, sd, ysd, h=HORIZON):
    import torch
    vpred = []
    v = w["v"][HIST - 1]; s = w["s"][HIST - 1]
    hist = np.array([_feat(w["v"][k], w["s"][k], w["vl"][k]) for k in range(HIST)],
                    dtype=np.float32)
    for k in range(h):
        vl = w["vl"][HIST + k]
        inp = (hist - mu) / sd
        with torch.no_grad():
            o, _ = net.lstm(torch.tensor(inp[None]))
            dv = net.head(o[:, -1]).squeeze(-1).item() * ysd
        v = max(0.0, v + dv)
        s = s + (vl - v) * DT
        vpred.append(v)
        hist = np.vstack([hist[1:], np.array(_feat(v, s, vl), dtype=np.float32)])
    return np.array(vpred), _integrate_gap(w, np.array(vpred))


# ---------------------------------------------------------------------------
def _rmse(models_pred: dict, test: list[dict]) -> dict:
    acc = {k: {"v": [], "s": []} for k in models_pred}
    for w in test:
        for name, fn in models_pred.items():
            vp, sp = fn(w)
            acc[name]["v"].append(vp - w["v"][HIST:HIST + HORIZON])
            acc[name]["s"].append(sp - w["s"][HIST:HIST + HORIZON])
    out = {}
    for name in models_pred:
        ev = np.concatenate(acc[name]["v"]); es = np.concatenate(acc[name]["s"])
        out[name] = {"rmse_v": float(np.sqrt(np.mean(ev ** 2))),
                     "rmse_s": float(np.sqrt(np.mean(es ** 2)))}
    return out


def main() -> dict:
    banner("Case 5 - trajectory prediction (CV / CA / IDM / LSTM)")
    rng = np.random.default_rng(0)
    train = []
    for seed in range(12):
        train += _windows(generate_platoon(n_followers=4, t_end=200, dt=DT, seed=seed), 6, rng)
    test = []
    for seed in range(100, 106):
        test += _windows(generate_platoon(n_followers=4, t_end=200, dt=DT, seed=seed), 3, rng)
    print(f"  train windows: {len(train)}, test windows: {len(test)}")

    net, mu, sd, ysd = _train_lstm(train, epochs=250, seed=0)
    models = {
        "CV": predict_cv,
        "CA": predict_ca,
        "IDM": predict_idm,
        "LSTM": lambda w: predict_lstm(w, net, mu, sd, ysd),
    }
    scores = _rmse(models, test)
    print(f"\n  {'model':6s} {'RMSE v [m/s]':>14s} {'RMSE gap [m]':>14s}")
    for name, s in scores.items():
        print(f"  {name:6s} {s['rmse_v']:14.3f} {s['rmse_s']:14.3f}")

    # error vs horizon
    horizons = [5, 10, 20, 50]
    fig, axes = new_fig(1, 2, figsize=(12, 4.2))
    for i, name in enumerate(models):
        errs_v, errs_s = [], []
        for H in horizons:
            ev, es = [], []
            for w in test:
                vp, sp = models[name](w)
                ev.append((vp - w["v"][HIST:HIST + HORIZON])[:H])
                es.append((sp - w["s"][HIST:HIST + HORIZON])[:H])
            ev = np.concatenate(ev); es = np.concatenate(es)
            errs_v.append(np.sqrt(np.mean(ev ** 2)))
            errs_s.append(np.sqrt(np.mean(es ** 2)))
        axes[0].plot(horizons, errs_v, "o-", color=PALETTE[i], label=name)
        axes[1].plot(horizons, errs_s, "o-", color=PALETTE[i], label=name)
    axes[0].set_xlabel("horizon [steps]"); axes[0].set_ylabel("speed RMSE [m/s]")
    axes[0].set_title("Speed error vs horizon"); axes[0].legend()
    axes[1].set_xlabel("horizon [steps]"); axes[1].set_ylabel("gap RMSE [m]")
    axes[1].set_title("Gap error vs horizon"); axes[1].legend()
    savefig(fig, FIGS / "case5_prediction.png")

    # example rollout
    fig2, ax = new_fig(1, 2, figsize=(12, 4.2))
    w = test[0]
    tt = np.arange(HORIZON) * DT
    ax[0].plot(np.arange(HIST + HORIZON) * DT, w["v"], "k", label="true follower")
    ax[0].plot(np.arange(HIST + HORIZON) * DT, w["vl"], color="gray", alpha=0.6, label="leader")
    for i, name in enumerate(models):
        vp, _ = models[name](w)
        ax[0].plot((HIST + np.arange(HORIZON)) * DT, vp, color=PALETTE[i], label=name)
    ax[0].axvline(HIST * DT, color="k", ls=":", lw=1)
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("speed [m/s]")
    ax[0].set_title("Example rollout"); ax[0].legend(fontsize=8)
    for i, name in enumerate(models):
        _, sp = models[name](w)
        ax[1].plot(tt, w["s"][HIST:HIST + HORIZON], "k", label="true", lw=1)
        ax[1].plot(tt, sp, color=PALETTE[i], label=name)
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("gap [m]")
    ax[1].set_title("Gap rollout"); ax[1].legend(fontsize=8)
    savefig(fig2, FIGS / "case5_prediction_example.png")

    metrics = {"config": {"hist_steps": HIST, "horizon_steps": HORIZON, "dt": DT,
                          "n_train": len(train), "n_test": len(test)},
               "scores": scores,
               "horizon_steps": horizons}
    write_metrics("case5", metrics)
    update_summary("case5", {"rmse_v": {k: round(v["rmse_v"], 3) for k, v in scores.items()},
                             "rmse_gap": {k: round(v["rmse_s"], 3) for k, v in scores.items()}})
    print("\nCase 5 done.")
    return metrics


if __name__ == "__main__":
    main()
