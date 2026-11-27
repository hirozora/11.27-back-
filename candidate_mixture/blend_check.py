"""Compare a fixed half-and-half action-probability blend using paired pilot archives."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    meta = json.loads((path / "run.json").read_text())
    with np.load(path / "validation.npz", allow_pickle=False) as z:
        # Column 4 is event_nll in the candidate_mixture train archive contract.
        return meta, z["event"], z["match"], z["label"], z["scores"][:, 4].astype(np.float64)


def match_gain(diff, matches):
    _, rows = np.unique(matches, return_inverse=True)
    n = np.bincount(rows)
    if len(n) < 2:
        raise ValueError("At least two development matches are needed.")
    t = np.bincount(rows, weights=diff)
    draw = np.random.default_rng(20260715).integers(len(n), size=(10_000, len(n)))
    sampled = t[draw].sum(1) / n[draw].sum(1)
    return {"gain": float(diff.mean()), "ci95": np.quantile(sampled, [.025, .975]).tolist()}


def summarize(direct, candidate):
    config, event, match, label, x = read(direct)
    mix, e2, m2, y2, y = read(candidate)
    if not (config["arm"].endswith("-context") and mix["arm"].endswith("-mixture")):
        raise ValueError("Pass the direct context arm first and mixture arm second.")
    skip = {"arm", "best_epoch"}
    if {k:v for k,v in config.items() if k not in skip} != {k:v for k,v in mix.items() if k not in skip}:
        raise ValueError("Use matched arms from the same data, seed, and training protocol.")
    for left, right in ((event,e2),(match,m2),(label,y2)):
        np.testing.assert_array_equal(left, right)
    if len(set(event)) != len(event) or (x < -1e-5).any() or (y < -1e-5).any() or not np.isfinite([x,y]).all():
        raise ValueError("Invalid or nonunique validation scores.")
    blend = np.log(2) - np.logaddexp(-x, -y)
    if ((blend - (x+y)*.5) > 1e-12).any():
        raise ValueError("The linear probability pool violates convexity of log loss.")
    return dict(split="development", weight_mixture=.5, seed=config["seed"],
                arms=[config["arm"],mix["arm"]], matches=len(np.unique(match)), events=len(event),
                nll=dict(direct=float(x.mean()), candidate=float(y.mean()), blend=float(blend.mean())),
                gain_positive_is_better=dict(against_direct=match_gain(x-blend,match),
                                             against_candidate=match_gain(y-blend,match)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("direct","candidate","out"):
        parser.add_argument("--" + option, type=Path, required=True)
    a = parser.parse_args()
    report = summarize(a.direct,a.candidate)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as file:
        file.write(payload)
    print(payload, end="")
