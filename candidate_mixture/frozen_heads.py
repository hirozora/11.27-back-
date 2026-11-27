"""Isolate the marginal training objective by updating only a linear action head."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

from evaluate import compare
from model import CandidateMLP, marginal
from train import FEATURES, METRICS, fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for opt in ("upstream", "data", "run", "out"):
        p.add_argument("--" + opt, type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--per-match", type=int, default=32)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--weight", type=float, default=.5)
    a = p.parse_args()
    if min(a.epochs, a.per_match, a.batch, a.weight) <= 0:
        p.error("Use positive budgets and marginal weight.")
    cfg = json.loads((a.run / "run.json").read_text())
    if not cfg["arm"].endswith("-mixture") or fingerprint(a.data) != cfg["data_sha256"]:
        raise ValueError("Use the original identical NPZ and a mixture checkpoint.")
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise ValueError("Output must be empty.")
    sys.path.insert(0, str(a.upstream.resolve() / "scene_regularity_release/src"))
    import train_seq
    train_seq.deterministic(cfg["seed"])
    dev = torch.device(a.device)
    weights = torch.load(a.run / "best.pt", map_location=dev, weights_only=True)
    base = CandidateMLP(weights["event.weight"].shape[0]).to(dev)
    base.load_state_dict(weights)
    base.eval().requires_grad_(False)

    layer = {}
    def candidate_state(_module, args):
        layer["state"] = args[0]
    def context_state(_module, _args, output):
        layer["context"] = output
    candidate_hook = base.action.register_forward_pre_hook(candidate_state)
    context_hook = base.sequence.register_forward_hook(context_state)

    with np.load(a.data, allow_pickle=False) as z:
        slot, valid, match, event = (z[k] for k in ("slot", "val", "match", "event"))
        features = {k: torch.as_tensor(z[k], device=dev) for k in (*FEATURES, "slot", "action")}
    if set(match[valid]) & set(match[~valid]):
        raise ValueError("Train and development matches overlap.")
    tr, va = np.flatnonzero(~valid & (slot >= 0)), np.flatnonzero(valid & (slot >= 0))
    schedule = tr[train_seq.rotate_orders(match[tr], cfg["seed"], a.epochs, a.per_match)]
    ridx = torch.as_tensor(va, device=dev)
    relation = cfg["arm"].startswith("on-")
    labels = features["action"][ridx].cpu().numpy()

    @torch.no_grad()
    def encode(i):
        q, _, _ = base({k: features[k][i] for k in FEATURES}, relation, "mixture")
        return q, layer.pop("state"), layer.pop("context")

    def losses(head, i, arm, q, h, c):
        y, actor = (features[k][i].long() for k in ("action", "slot"))
        ix = torch.arange(len(i), device=dev)
        if arm == "context":
            action_nll = -head(c).log_softmax(-1)[ix, y]
            conditional_nll = action_nll
        else:
            r = head(h).log_softmax(-1)
            action_nll = -marginal(q, r)[ix, y]
            conditional_nll = -r[ix, actor, y]
        actor_nll = -q[ix, actor]
        return action_nll, conditional_nll, actor_nll

    @torch.no_grad()
    def evaluate(head, arm):
        chunks = []
        for i in ridx.split(a.batch):
            chunks.append(torch.stack(losses(head, i, arm, *encode(i)), -1).cpu().numpy())
        return np.concatenate(chunks)

    original = evaluate(base.action, "conditional")
    with np.load(a.run / "validation.npz", allow_pickle=False) as prior:
        for k, v in dict(event=event[va], match=match[va], label=labels).items():
            np.testing.assert_array_equal(prior[k], v)
        for i, metric in enumerate(("event_nll", "conditional_event_nll_oracle", "actor_nll")):
            np.testing.assert_allclose(prior["scores"][:, METRICS.index(metric)],
                                       original[:, i], rtol=1e-5, atol=3e-5)

    result = {"arm": cfg["arm"], "seed": cfg["seed"], "train_matches": int(len(np.unique(match[tr]))),
              "dev_matches": int(len(np.unique(match[va]))), "dev_reachable": len(va),
              "epochs": a.epochs, "sampled_per_match_per_epoch": a.per_match,
              "marginal_loss_weight": a.weight, "runs": {}}
    score = {"original_mixture": original}
    metrics = ("event_nll", "conditional_event_nll", "actor_nll")
    for arm in ("context", "conditional", "conditional_plus_marginal"):
        # Warm-start both mixture heads identically, with exactly the same sampled mini-batches.
        head = copy.deepcopy(base.action).requires_grad_(True)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.0027, weight_decay=.0001)
        for order in schedule:
            for i in torch.as_tensor(order, device=dev).split(a.batch):
                event_nll, conditional, _ = losses(head, i, arm, *encode(i))
                loss = conditional.mean()
                if arm == "conditional_plus_marginal":
                    loss = loss + a.weight * event_nll.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        score[arm] = evaluate(head, arm)
        np.testing.assert_array_equal(score[arm][:, 2], original[:, 2])
        torch.save(head.state_dict(), a.out / f"{arm}.pt")
    result["runs"] = {k: dict(zip(metrics, v.mean(0, dtype=np.float64).tolist())) for k,v in score.items()}
    result["marginal_over_conditional"] = compare(
        (score["conditional"][:, 0] - score["conditional_plus_marginal"][:, 0])[None, :],
        match[va], 10_000)
    (a.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
    candidate_hook.remove()
    context_hook.remove()


if __name__ == "__main__":
    main()
