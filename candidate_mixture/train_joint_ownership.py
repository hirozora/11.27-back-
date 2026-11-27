"""Matched canonical joint actor-zone experiment with optional ownership bits.

Training loop adapted from hirozora/11.27 at cb165beb, onball_participation/train_joint.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Canonical view NPZ")
    parser.add_argument("--ownership", choices=["off", "on"], required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--per-match", type=int, default=128)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.per_match <= 0:
        parser.error("Training sizes must be positive")
    sys.path.insert(0, str(args.canonical.resolve()))
    from train import Actor, deterministic, rotate_orders, lr_at

    class OwnershipActor(Actor):
        def __init__(self):
            super().__init__()
            self.owner = nn.Linear(32, 64, bias=False)
            nn.init.zeros_(self.owner.weight)

        def candidate_hidden(self, d, i, role):
            context = self.encode_context(d, i)
            x = torch.cat([context[:, None].expand(-1, 11, -1),
                           d["hist"][i].float(), role], -1)
            hidden = self.scorer[0](x)
            player = d["player"][i]
            owner = d["window_actor"][i]
            bits = (player[..., None] == owner[:, None]) & (player[..., None] != 0)
            end = self.scorer[2](self.scorer[1](hidden + self.owner(bits.float())))
            return end, context

    args.out.mkdir(parents=True, exist_ok=True)
    deterministic(args.seed)
    device = torch.device(args.device)
    with np.load(args.data, allow_pickle=False) as src, np.load(args.reference, allow_pickle=False) as ref:
        for k in ("token", "wzone", "age", "context", "hist", "player", "group", "slot", "zone", "match", "event", "val"):
            np.testing.assert_array_equal(src[k], ref[k], err_msg=k)
        owners = src["window_actor"]
        if not ((owners >= 0).all() and not owners[src["token"] == 0].any()):
            raise ValueError("Invalid ownership values in the visible window")
        d_extra = {"player": torch.as_tensor(src["player"], device=device),
                   "window_actor": torch.as_tensor(owners if args.ownership == "on"
                                                   else np.zeros_like(owners), device=device)}
        players = src["player"]
        slot, match, valflag = src["slot"], src["match"], src["val"]
        training = np.flatnonzero((slot >= 0) & ~valflag)
        validation = np.flatnonzero((slot >= 0) & valflag)
        events = src["event"][validation]
        d = {
            k: torch.as_tensor(src[k], device=device)
            for k in ("token", "wzone", "age", "context", "hist", "slot", "zone")
        }
        d["mask"] = torch.as_tensor(src["player"] != 0, device=device)
        d.update(d_extra)
        role = F.one_hot(torch.as_tensor(src["group"], device=device).long(), 8).float()
    schedule = training[rotate_orders(match[training], args.seed, args.epochs, args.per_match)]
    orders = torch.as_tensor(schedule, device=device)
    val = torch.as_tensor(validation, device=device)
    # Preserve canonical location initialization despite the extra projection's constructor.
    with torch.random.fork_rng(devices=[]):
        actor = OwnershipActor().to(device)
    Actor()  # Advance CPU initialization RNG by exactly the canonical actor draw.
    location = nn.Linear(64, 9).to(device)
    deterministic(args.seed)
    optimizer = torch.optim.AdamW(
        [*actor.parameters(), *location.parameters()], lr=0.0027, weight_decay=0.0001
    )
    config = {"lr": 0.0027, "milestones": [96, 108], "factor": 0.3}

    def forward(i):
        state, _ = actor.candidate_hidden(d, i, role[i])
        return actor.actor_logits(state, d["mask"][i]), state

    best = float("inf")
    curve = []
    for epoch, order in enumerate(orders, 1):
        actor.train()
        for g in optimizer.param_groups:
            g["lr"] = lr_at(config, epoch)
        for i in order.split(256):
            scores, state = forward(i)
            truth = d["slot"][i].long()
            chosen = state[torch.arange(len(i), device=device), truth]
            loss = F.cross_entropy(scores, truth) + F.cross_entropy(
                location(chosen), d["zone"][i].long()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for model in (actor, location):
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch % 4 == 0 or epoch == args.epochs:
            actor.eval()
            with torch.no_grad():
                nll = sum(
                    F.cross_entropy(
                        forward(i)[0], d["slot"][i].long(), reduction="sum"
                    ).item()
                    for i in val.split(2048)
                ) / len(val)
            if not np.isfinite(nll):
                raise ValueError("nonfinite validation NLL")
            curve.append([epoch, nll])
            print(args.ownership, epoch, nll, flush=True)
            if nll < best:
                best = nll
                torch.save(
                    {"actor": actor.state_dict(), "location": location.state_dict()},
                    args.out / "best.pt",
                )
    checkpoint = torch.load(
        args.out / "best.pt", map_location=device, weights_only=True
    )
    actor.load_state_dict(checkpoint["actor"])
    location.load_state_dict(checkpoint["location"])
    results, probabilities = [], []
    actor.eval()
    with torch.no_grad():
        for i in val.split(1024):
            logits, state = forward(i)
            q = logits.log_softmax(-1)
            probabilities.append(q.exp().cpu().numpy())
            c = location(state).log_softmax(-1)
            joint = q[:, :, None] + c
            marginal = joint.logsumexp(1)
            assert marginal.logsumexp(-1).abs().max() < 1e-5
            truth = d["slot"][i].long()
            region = d["zone"][i].long()
            b = torch.arange(len(i), device=device)
            target = q[b, truth]
            higher = (q > target[:, None]).sum(1)
            tied = (q == target[:, None]).sum(1)
            metrics = [
                -target,
                *[((k - higher) / tied).clamp(0, 1) for k in (1, 3)],
                -c[b, truth, region],
                -marginal[b, region],
                -joint[b, truth, region],
            ]
            results.append(torch.stack(metrics, 1).cpu().numpy())
    values = np.concatenate(results)
    assert np.isfinite(values).all()
    np.testing.assert_allclose(values[:, 0] + values[:, 3], values[:, 5], atol=2e-6)
    np.savez(
        args.out / "validation.npz",
        event=events,
        match=match[validation],
        scores=values,
    )
    (args.out / "curve.json").write_text(json.dumps(curve) + "\n")
    np.savez(args.out / "actor_weights.npz", event=events, match=match[validation],
             player=players[validation], q=np.concatenate(probabilities))
    metadata = dict(ownership=args.ownership, seed=args.seed, epochs=args.epochs,
                    per_match=args.per_match, best_epoch=min(curve, key=lambda row: row[1])[0])
    (args.out / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(args.ownership, args.seed, best, flush=True)


if __name__ == "__main__":
    main()
