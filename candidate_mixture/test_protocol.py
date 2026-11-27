"""Run from the backup repository: python candidate_mixture/test_protocol.py --upstream ../11.27."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_views import extend
from model import CandidateMLP, marginal, ownership_mask, role_roll


def check_visibility(base):
    events = pd.DataFrame({
        "period": [0]*7, "sec": [6., 7., 8., 8., 9., 10., 11.],
        "team_id": [2, 1, 2, 1, 1, 1, 1], "event_name": ["Pass"]*7,
        "sub_event_name": ["Simple pass"]*7, "player_id": [1, 2, 3, 1, 0, 2, 1],
        "x": [40.]*7, "y": [51.]*7, "xy_valid": [True]*7,
    })
    query = pd.DataFrame({"period":[0], "sec":[10.], "clock":[10.], "team_id":[1]})
    roster = pd.DataFrame({"player_id":[1, 2], "card_period":[-1, -1], "card_sec":[np.nan, np.nan]})
    substitutions = pd.DataFrame(columns=["player_in", "player_out", "minute"])
    future = events.copy()
    future.loc[future.sec >= 10, ["player_id", "x", "y"]] = [1, 1., 99.]
    for lag, expected in ((False, [0, 2, 1, 0]), (True, [0, 2])):
        result = extend(base, base.inputs, events, query, roster, substitutions, lag)
        np.testing.assert_array_equal(result["window_actor"][0, -len(expected):], expected)
        assert np.count_nonzero(result["token"]) == len(expected)
        altered = extend(base, base.inputs, future, query, roster, substitutions, lag)
        for field in result:
            np.testing.assert_array_equal(result[field], altered[field])
        owners = ownership_mask(*(torch.as_tensor(result[k]) for k in ("token", "player", "window_actor")))
        assert not owners[:, 2:].any()
        assert owners[0, 0].sum().item() == (0 if lag else 1)
        assert owners[0, 1].sum().item() == 1


def check_model():
    model = CandidateMLP(20).eval()
    x = dict(token=torch.ones(2, 32, dtype=torch.long), wzone=torch.zeros(2, 32, dtype=torch.long),
             context=torch.rand(2,3), age=torch.rand(2,32), hist=torch.rand(2,11,22),
             player=torch.tensor([[11,22,33]+[0]*8,[11,22,33]+[0]*8]),
             group=torch.tensor([[0,0,1]+[0]*8,[0,1,2]+[0]*8]),
             window_actor=torch.tensor([[11,22,0,33]*8,[11,22,0,33]*8]))
    x["token"][:,2] = 0
    for head in ("context", "mixture"):
        q, z, r = model(x, True, head)
        probability = marginal(q, r).exp()
        torch.testing.assert_close(probability.sum(1), torch.ones(2))
        torch.testing.assert_close(probability, (q.exp()[:, :, None]*r.exp()).sum(1))
        assert not q.exp()[:,3:].any()
        perm = torch.tensor([2, 0, 1, *range(3, 11)])
        px = dict(x)
        for k in ("player", "group", "hist"):
            px[k] = x[k][:, perm]
        pq, pz, pr = model(px, True, head)
        for a,b in ((pq,q),(pz,z),(pr,r)):
            torch.testing.assert_close(a, b[:,perm])
        torch.testing.assert_close(marginal(pq, pr).exp(), probability)
        swapped, exch = role_roll(x["player"], x["group"], r)
        assert exch.tolist() == [True,False]
        torch.testing.assert_close(swapped[0, 0], r[0, 1])
        torch.testing.assert_close(swapped[0, 2:], r[0, 2:])
        if head == "context":
            torch.testing.assert_close(probability, r[:,0].exp())
        q0, _, r0 = model(x, False, head)
        x0 = dict(x, window_actor=torch.full_like(x["window_actor"], 33))
        q1, _, r1 = model(x0, False, head)
        torch.testing.assert_close(q0, q1)
        torch.testing.assert_close(r0, r1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.upstream.resolve() / "onball_participation"))
    import build
    check_visibility(build)
    check_model()
    print("PASS: timestamp cutoff, target/future perturbation, unknown/padded owners, candidate equivariance, probability normalization, role permutation")
