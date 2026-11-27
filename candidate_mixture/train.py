"""Train one predeclared development arm, or run an end-to-end CPU smoke test."""
import argparse
import json
import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from model import CandidateMLP, marginal, role_roll

FEATURES = ("token", "wzone", "age", "context", "hist", "player", "group", "window_actor")
METRICS = ("actor_nll", "actor_top1", "actor_top3", "zone_nll_given_actor",
           "event_nll", "event_acc", "conditional_event_nll_oracle",
           "event_nll_hard_top1_mixture_ties", "event_nll_uniform", "event_nll_role_permuted")


def per_query(q, z, r, x):
    a, e, zone = (x[k].long() for k in ("slot", "action", "zone"))
    b = torch.arange(len(a), device=a.device)
    truth = q[b, a]
    p = marginal(q, r)
    tied_best = q == q.amax(1, keepdim=True)
    hard = marginal(-tied_best.sum(1, keepdim=True).log().expand_as(q).masked_fill(~tied_best, torch.inf), r)
    uniform = marginal(torch.zeros_like(q).masked_fill(~torch.isfinite(q), -torch.inf).log_softmax(-1), r)
    swapped, exchangeable = role_roll(x["player"], x["group"], r)
    above = (q > truth[:, None]).sum(1)
    ties = (q == truth[:, None]).sum(1)
    return torch.stack((
        -truth, *[((k - above) / ties).clamp(0, 1) for k in (1, 3)],
        -z[b, a, zone], -p[b, e], (p.argmax(-1) == e).float(), -r[b, a, e],
        -hard[b, e], -uniform[b, e], -marginal(q, swapped)[b, e],
    ), -1), exchangeable, p


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args, train_util, vocab_size):
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise ValueError("Run output directory must be empty.")
    device = torch.device(args.device)
    train_util.deterministic(args.seed)
    with np.load(args.data, allow_pickle=False) as src:
        match, event, val, slot = (src[k] for k in ("match", "event", "val", "slot"))
        names = src["action_names"].tolist()
        data = {k: torch.as_tensor(src[k], device=device) for k in (*FEATURES, "slot", "zone", "action")}
    use = slot >= 0
    if set(match[~val]) & set(match[val]) or len(np.unique(event)) != len(event):
        raise ValueError("Development and training matches must be disjoint, event IDs unique.")
    select = torch.as_tensor(np.flatnonzero(use), device=device)
    if not len(select) or len(names) != 10 or not ((data["action"] >= 0) & (data["action"] < 10)).all():
        raise ValueError("Expected reachable rows and ten action classes.")
    if not ((data["zone"][select] >= 0) & (data["zone"][select] < 9)).all():
        raise ValueError("Invalid auxiliary location label.")
    if not (data["player"][select, data["slot"][select].long()] > 0).all():
        raise ValueError("Reachable target refers to a padded candidate.")
    tr, va = np.flatnonzero(~val & use), np.flatnonzero(val & use)
    if not len(tr) or not len(va):
        raise ValueError("No reachable training or validation samples.")
    orders = tr[train_util.rotate_orders(match[tr], args.seed, args.epochs, args.per_match)]
    model = CandidateMLP(vocab_size).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=0.0027, weight_decay=0.0001)
    relation, head = args.arm.split("-")
    relation = relation == "on"
    hp = {"lr": 0.0027, "milestones": [96, 108], "factor": 0.3}
    config = {"seed": args.seed, "arm": args.arm, "epochs": args.epochs,
              "per_match": args.per_match, "batch": args.batch, "select": args.select,
              "event_weight": args.event_weight, "marginal_weight": args.marginal_weight,
              "match_weighting": "action", "labels": names, "data_sha256": fingerprint(args.data)}
    best, curve = float("inf"), []
    val_idx = torch.as_tensor(va, device=device)

    def batch(i):
        return {k: v[i] for k, v in data.items()}

    @torch.no_grad()
    def validation():
        model.eval()
        parts = [[], [], []]
        for i in val_idx.split(args.batch):
            x = batch(i)
            q, z, r = model(x, relation, head)
            score, swap, p = per_query(q, z, r, x)
            torch.testing.assert_close(torch.logsumexp(p, -1), torch.zeros(len(i), device=device),
                                       atol=2e-5, rtol=0)
            parts[0].append(score.cpu().numpy())
            parts[1].append(swap.cpu().numpy())
            parts[2].append(p.argmax(-1).cpu().numpy())
        scores, swaps, predicted = (np.concatenate(x) for x in parts)
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite validation scores.")
        return scores, swaps, predicted

    for epoch, order in enumerate(orders, 1):
        model.train()
        for pg in optim.param_groups:
            pg["lr"] = train_util.lr_at(hp, epoch)
        for i in torch.as_tensor(order, device=device).split(args.batch):
            x = batch(i)
            q, z, r = model(x, relation, head)
            b = torch.arange(len(i), device=device)
            a, y, zone = (x[k].long() for k in ("slot", "action", "zone"))
            loss = (-q[b, a] - z[b, a, zone] - args.event_weight * r[b, a, y]).mean()
            if args.marginal_weight:
                loss = loss - args.marginal_weight * marginal(q.detach(), r)[b, y].mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optim.step()
        if epoch % 4 == 0 or epoch == args.epochs:
            scores, exchangeable, predicted = validation()
            mean = dict(zip(METRICS, scores.mean(0, dtype=np.float64).tolist()))
            curve.append({"epoch": epoch, **mean})
            value = mean[args.select]
            if value < best:
                best = value
                config["best_epoch"] = epoch
                torch.save(model.state_dict(), args.out / "best.pt")
                np.savez(args.out / "validation.npz", event=event[va], match=match[va],
                         label=data["action"][val_idx].cpu().numpy(), scores=scores, exchangeable=exchangeable, predicted=predicted)
            print(epoch, mean, flush=True)
    config.update(validation_total=int(val.sum()), validation_reached=len(va),
                  training_total=int((~val).sum()), training_reached=len(tr))
    (args.out / "run.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "curve.json").write_text(json.dumps(curve, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--per-match", type=int, default=128)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--event-weight", type=float, default=1)
    p.add_argument("--marginal-weight", type=float, default=0)
    p.add_argument("--select", choices=("event_nll", "actor_nll"), default="event_nll")
    p.add_argument("--arm", choices=("off-context", "on-context", "off-mixture", "on-mixture"), required=True)
    args = p.parse_args()
    if min(args.epochs, args.per_match, args.batch) < 1 or min(args.event_weight, args.marginal_weight) < 0:
        p.error("Training sizes must be positive and loss weights nonnegative.")
    sys.path.insert(0, str(args.upstream.resolve() / "scene_regularity_release/src"))
    import train_seq
    from model_inputs import VOCAB
    run(args, train_seq, 1 + 2 * len(VOCAB["sub_event_name"]))


if __name__ == "__main__":
    main()
