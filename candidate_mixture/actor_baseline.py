"""Train-only masked conditional-logit controls: role, and role + visible-owner recency."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from evaluate import compare


def features(player, role, owner, token):
    k = token.shape[1]
    visible = (owner[:, None, :] == player[:, :, None]) & (token[:, None, :] > 0) & (player[:, :, None] > 0)
    latest = np.where(visible, np.arange(k), -1).max(-1)
    distance = np.where(latest < 0, k, k - 1 - latest)
    # Role 7 is the reference (including unknown); recency reference is unseen in the window.
    role_x = np.eye(8, dtype=np.float32)[role][:, :, :7]
    edges = np.array([0, 1, 3, 7, 15, 31])
    bucket = np.searchsorted(edges, distance)
    recent_x = np.eye(len(edges) + 1, dtype=np.float32)[bucket][:, :, :len(edges)]
    freq = (np.log1p(visible.sum(-1)) / np.log1p(k)).astype(np.float32)[..., None]
    return role_x, np.concatenate((role_x, recent_x, freq), -1)


def log_probs(x, valid, beta):
    logit = x @ beta.astype(np.float32)
    logit[~valid] = -np.inf
    return logit - logsumexp(logit, axis=1, keepdims=True)


def fit(x, valid, target, reg):
    n, k, width = x.shape
    batch = np.arange(n)
    observed = x[batch, target].mean(0, dtype=np.float64)
    flat = np.ascontiguousarray(x.reshape(n*k, width).T)
    def objective(beta):
        lp = log_probs(x, valid, beta)
        loss = -lp[batch, target].mean(dtype=np.float64) + .5 * reg * (beta @ beta)
        modeled = (flat @ np.exp(lp).reshape(-1)).astype(np.float64) / n
        return loss, modeled - observed + reg * beta
    res = minimize(objective, np.zeros(width), method="L-BFGS-B", jac=True,
                   options={"gtol": 5e-6, "maxiter": 200})
    if not res.success:
        raise RuntimeError(str(res.message))
    return res.x


def score(lp, target):
    a = lp[np.arange(len(target)), target]
    above = (lp > a[:, None]).sum(1)
    same = (lp == a[:, None]).sum(1)
    return np.stack((-a, np.clip((1 - above) / same, 0, 1),
                     np.clip((3 - above) / same, 0, 1)), -1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--reg", type=float, default=1e-4)
    a = p.parse_args()
    if a.reg <= 0:
        p.error("--reg must be positive")
    with np.load(a.data, allow_pickle=False) as z:
        slot, val, match = (z[k] for k in ("slot", "val", "match"))
        mask = slot >= 0
        dev = val[mask]
        target = slot[mask]
        player, role, owner, token = (z[k][mask] for k in ("player", "group", "window_actor", "token"))
        event = z["event"][mask][dev]
    if set(match[val]) & set(match[~val]):
        raise ValueError("Train/dev match contamination.")
    valid = player > 0
    if not valid[np.arange(len(player)), target].all():
        raise ValueError("A target is outside the valid candidate slots.")
    role_x, recent_x = features(player, role, owner, token)
    values = {"uniform": score(log_probs(role_x[dev], valid[dev], np.zeros(role_x.shape[-1])), target[dev])}
    coefficients = {}
    for name, x in (("role", role_x), ("role_recency", recent_x)):
        beta = fit(x[~dev], valid[~dev], target[~dev], a.reg)
        coefficients[name] = beta.tolist()
        values[name] = score(log_probs(x[dev], valid[dev], beta), target[dev])
    keys = ("actor_nll", "actor_top1", "actor_top3")
    report = {"dev_matches": int(len(np.unique(match[val]))), "dev_total": int(val.sum()),
              "dev_reached": int(dev.sum()), "train_reached": int((~dev).sum()), "reg": a.reg,
              "role_logits_reference": "role_7",
              "recency_slot_cutoffs_inclusive": [0, 1, 3, 7, 15, 31],
              "averages": {k: dict(zip(keys, v.mean(0, dtype=np.float64).tolist())) for k,v in values.items()},
              "coefficients": coefficients, "gain_positive_is_better": {}}
    for base, test in (("uniform", "role"), ("role", "role_recency")):
        d = (values[base] - values[test]) * np.array([1, -1, -1])
        report["gain_positive_is_better"][f"{test}_vs_{base}"] = {
            key: compare(d[None, :, j], match[mask][dev], 10_000) for j,key in enumerate(keys)
        }
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "summary.json").open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    np.savez(a.out / "development.npz", event=event, match=match[mask][dev], **values)
    print(json.dumps(report["averages"], indent=2))


if __name__ == "__main__":
    main()
