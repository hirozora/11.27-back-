"""Check prior open-play annotation ownership and family, not inferred ball possession."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from actor_baseline import features, fit, log_probs, score


def summarize(dev, family, known, eligible, correct, age):
    out = {}
    for code, name in enumerate(("pass", "carry_touch", "other_open_play")):
        rows = dev & (family == code)
        n = int(rows.sum())
        ok = int((rows & eligible).sum())
        hit = int((rows & correct).sum())
        out[name] = dict(rows=n, owner_known=int((rows & known).sum()), eligible=ok,
                         same_actor=hit, same_actor_rate=hit / ok if ok else None,
                         encoded_age_median=float(np.median(age[rows])) if n else None)
    out["no_open_play"] = int((dev & (family < 0)).sum())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("canonical", "data", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.canonical.resolve() / "scene_regularity_release/src"))
    from event_schema import ACTION_OF
    from model_inputs import VOCAB
    from canonical_bridge import paired

    vocab = VOCAB["sub_event_name"]
    code = {sub: 0 if event == "Pass" else 1 if action in ("carry", "touch") else 2
            for (event, sub), action in ACTION_OF.items()}
    lookup = np.full(1 + 2 * len(vocab), -1)
    for name, family in code.items():
        lookup[1 + vocab.index(name)] = family
    with np.load(a.data, allow_pickle=False) as src:
        r = src["slot"] >= 0
        player, role, owner, token, age, target, match, dev = (
            src[k][r] for k in ("player", "group", "window_actor", "token", "age", "slot", "match", "val"))
    if np.intersect1d(match[dev], match[~dev]).size:
        raise ValueError("Training and development matches overlap")
    n = np.arange(len(target))
    family = lookup[token]
    latest = np.where(family >= 0, np.arange(token.shape[1]), -1).max(1)
    family = np.where(latest >= 0, family[n, latest], -1)
    last_owner = np.where(latest >= 0, owner[n, latest], 0)
    known = last_owner > 0
    own = known[:, None] & (last_owner[:, None] == player)
    eligible = own.any(1)
    results = {"development_table": summarize(dev, family, known, eligible, own[n, target], age[n, latest]),
               "training_table": summarize(~dev, family, known, eligible, own[n, target], age[n, latest]),
               "actor_nll": {}, "gain": {}}
    _, recency = features(player, role, owner, token)
    one = own[..., None].astype(np.float32)
    interaction = one * np.stack((family == 0, family == 1), -1)[:, None, :]
    mask = player > 0
    losses = {}
    for name, extra in (("baseline", ()), ("last_owner", (one,)),
                        ("family_interaction", (one, interaction))):
        x = np.concatenate((recency, *extra), axis=-1) if extra else recency
        beta = fit(x[~dev], mask[~dev], target[~dev], 1e-4)
        lp = log_probs(x[dev], mask[dev], beta)
        loss = score(lp, target[dev])[:, 0].astype(float)
        losses[name] = loss
        results["actor_nll"][name] = float(loss.mean())
    for base, test in (("baseline", "last_owner"), ("last_owner", "family_interaction")):
        results["gain"][test] = paired(losses[base] - losses[test], match[dev])
    with a.out.open("x") as f:
        json.dump(results, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
