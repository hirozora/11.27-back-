"""Replay saved mixture checkpoints; remove within-role alignment without changing role mass."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import compare
from model import CandidateMLP, marginal
from train import FEATURES, METRICS, fingerprint


def exchangeable(player, group):
    valid = player > 0
    return (valid[:, :, None] & valid[:, None, :]
            & (group[:, :, None] == group[:, None, :]) & (group[:, :, None] < 7))


def role_average(log_q, log_r, player, group):
    """Expected event probability under uniform same-role reassignments of r, with q fixed."""
    pairs = exchangeable(player, group)
    pairs = pairs | torch.eye(player.shape[1], dtype=torch.bool, device=player.device)
    # Redistribution form: same sum of q weights within each role, uniform inside it.
    # Therefore sum(mean_role(q) * r) = sum(q * mean_role(r)).
    mass = torch.logsumexp(log_q[:, None, :].masked_fill(~pairs, -torch.inf), dim=-1)
    log_role_uniform_q = mass - pairs.sum(-1).log().to(log_q.dtype)
    result = marginal(log_role_uniform_q, log_r)
    return result, (pairs.sum(-1) > 1).any(-1)


def self_test():
    q = torch.tensor([[.35, .25, .05, .15, .20, 0.]], dtype=torch.float64)
    p = torch.tensor([[.05, .8, .2, .4, .25, .3]], dtype=torch.float64)
    r = torch.stack((p, 1-p), -1)
    player = torch.tensor([[101, 102, 103, 104, 105, 0]])
    role = torch.tensor([[2, 2, 3, 7, 7, 2]])
    base, take = role_average(q.log(), r.log(), player, role)
    perm = [1, 0, 2, 3, 4, 5]
    formal = ((q[..., None] * r).sum(1) + (q[..., None] * r[:, perm]).sum(1)) * .5
    torch.testing.assert_close(base.exp(), formal)
    assert take.item()
    changed = q.clone()
    changed[:, :2] = torch.tensor([.55, .05], dtype=q.dtype)
    changed_base, _ = role_average(changed.log(), r.log(), player, role)
    torch.testing.assert_close(changed_base, base)
    assert marginal(changed.log(), r.log())[0, 0] < marginal(q.log(), r.log())[0, 0]
    isolated_role = torch.tensor([[2, 4, 3, 7, 7, 2]])
    same, take = role_average(q.log(), r.log(), player, isolated_role)
    assert not take.item()
    torch.testing.assert_close(same, marginal(q.log(), r.log()))
    print("PASS: exact permutation expectation, role-mass invariant, singleton/unknown/padding.")


def replay(data, runs, output, device, batch_size):
    signature = fingerprint(data)
    with np.load(data, allow_pickle=False) as src:
        rows = np.flatnonzero(src["val"] & (src["slot"] >= 0))
        event, match, label = (src[k][rows] for k in ("event", "match", "action"))
        table = {k: torch.as_tensor(src[k][rows], device=device) for k in FEATURES}
    deltas = []
    same = None
    seeds, arm, protocol = [], None, None
    for folder in runs:
        cfg = json.loads((folder / "run.json").read_text())
        if cfg["data_sha256"] != signature:
            raise ValueError(f"Input checksum mismatch: {folder}")
        if not cfg["arm"].endswith("-mixture") or cfg["event_weight"] <= 0:
            raise ValueError("Replay requires a supervised mixture checkpoint.")
        if arm is not None and arm != cfg["arm"]:
            raise ValueError("Specify one arm with unique seeds.")
        if cfg["seed"] in seeds:
            raise ValueError("Duplicate seed.")
        sig = {k: v for k, v in cfg.items() if k not in ("seed", "best_epoch")}
        if protocol is not None and sig != protocol:
            raise ValueError("Checkpoints must use the same fitting protocol.")
        protocol = sig
        seeds.append(cfg["seed"])
        arm = cfg["arm"]
        weights = torch.load(folder / "best.pt", map_location=device, weights_only=True)
        model = CandidateMLP(weights["event.weight"].shape[0]).to(device).eval()
        model.load_state_dict(weights)
        plus, select, original = [], [], []
        with torch.inference_mode():
            for i in range(0, len(rows), batch_size):
                j = min(i + batch_size, len(rows))
                x = {k: v[i:j] for k, v in table.items()}
                y = torch.as_tensor(label[i:j], device=device).long()[:, None]
                q, _, r = model(x, arm.startswith("on-"), "mixture")
                fixed, take = role_average(q, r, x["player"], x["group"])
                learned_logp = marginal(q, r).gather(1, y).squeeze(1)
                fixed_logp = fixed.gather(1, y).squeeze(1)
                plus.append((learned_logp - fixed_logp).cpu().numpy())
                select.append(take.cpu().numpy())
                original.append(-learned_logp.cpu().numpy())
        pairs = np.concatenate(select)
        with np.load(folder / "validation.npz", allow_pickle=False) as prev:
            for key, tensor in dict(event=event, match=match, label=label, exchangeable=pairs).items():
                np.testing.assert_array_equal(prev[key], tensor)
            np.testing.assert_allclose(prev["scores"][:, METRICS.index("event_nll")],
                                       np.concatenate(original), atol=3e-5, rtol=1e-5)
        if same is not None:
            np.testing.assert_array_equal(same, pairs)
        same = pairs
        deltas.append(np.concatenate(plus).astype(float)[same])
    report = {"arm": arm, "seeds": seeds, "split": "development", "reachable_actions": len(rows),
              "known_role_exchangeable_actions": int(same.sum()),
              "gain_positive_is_better": compare(np.stack(deltas), match[same], 10_000)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--run", nargs="+", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=256)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not (args.data and args.run and args.out and args.batch > 0):
            parser.error("Provide --data --run --out and a positive --batch.")
        torch.set_num_threads(1)
        replay(args.data, args.run, args.out, torch.device(args.device), args.batch)
