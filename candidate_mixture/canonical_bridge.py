"""Replay frozen canonical role/personal action heads with alternate actor marginals."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import CandidateMLP


def reorder_q(old_player, new_player, q):
    np.testing.assert_array_equal(np.sort(old_player, 1), np.sort(new_player, 1))
    eq = old_player[:, :, None] == new_player[:, None, :]
    live = old_player != 0
    if not np.all(eq.sum(2)[live] == 1):
        raise ValueError("Duplicate active candidate ID")
    aligned = np.where(live, np.take_along_axis(q, eq.argmax(2), axis=1), 0)
    np.testing.assert_allclose(aligned.sum(1), 1, atol=1e-6)
    return aligned


def replace_actor(joint, q, mask):
    mass = joint.sum(-1)
    if not (np.all(mass[mask] > 0) and np.all(joint >= 0)
            and np.all(q[mask] > 0) and not joint[~mask].any() and not q[~mask].any()):
        raise ValueError("Candidate supports or probabilities are invalid")
    conditional = joint / np.where(mask, mass, 1)[..., None]
    changed = conditional * q[..., None]
    np.testing.assert_allclose(changed.sum(-1), q, atol=1e-7, rtol=1e-6)
    return changed


def role_counterfactuals(q0, q1, group, mask):
    """Cross group mass and within-group conditional weights; unknowns are singleton groups."""
    if not np.isin(group[mask], np.arange(8)).all():
        raise ValueError("Invalid canonical role labels")
    known = (group < 7) & mask
    same = ((group[:, :, None] == group[:, None, :]) &
            known[:, :, None] & known[:, None, :])
    same |= np.eye(group.shape[1], dtype=bool)[None] & mask[:, :, None]
    mass = [np.einsum("nij,nj->ni", same, q) for q in (q0, q1)]
    cond = [q / np.where(mask, m, 1) for q, m in zip((q0, q1), mass, strict=True)]
    off_role = mass[0] * cond[1]
    on_role = mass[1] * cond[0]
    for q, m in zip((off_role, on_role), mass, strict=True):
        np.testing.assert_allclose(q.sum(1), 1, atol=1e-6)
        np.testing.assert_allclose(np.einsum("nij,nj->ni", same, q), m, atol=1e-7, rtol=1e-6)
    return {"q01": off_role, "q10": on_role}, (same.sum(2) > 1).any(1)


def role_decompose(l00, l01, l10, l11, match):
    between = .5 * ((l00 - l10) + (l01 - l11))
    within = .5 * ((l00 - l01) + (l10 - l11))
    np.testing.assert_allclose(between + within, l00 - l11, atol=1e-12)
    return {"between_role_mass": paired(between, match),
            "within_role_weighting": paired(within, match),
            "cells_nll": {k: float(v.mean()) for k, v in zip(
                ("q00", "q01", "q10", "q11"), (l00, l01, l10, l11), strict=True)}}


def ids(match, event):
    key = pd.MultiIndex.from_arrays((match, event))
    if not key.is_unique:
        raise ValueError("Duplicate match/event key")
    return key


def exact(old, new):
    take = new.get_indexer(old)
    if len(old) != len(new) or (take < 0).any():
        raise ValueError("Development reachable cohorts differ")
    return take


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def alternate(backup_data, folders, keys, players, true_players, labels):
    features = ("token", "wzone", "age", "context", "hist", "player", "group", "window_actor")
    with np.load(backup_data, allow_pickle=False) as b:
        rows = np.flatnonzero(b["val"] & (b["slot"] >= 0))
        mk, ek = b["match"][rows], b["event"][rows]
        take = exact(keys, ids(mk, ek))
        base_player, base_y, slot = b["player"][rows], b["action"][rows], b["slot"][rows]
        np.testing.assert_array_equal(base_player[np.arange(len(rows)), slot][take], true_players)
        np.testing.assert_array_equal(base_y[take], labels)
        table = {k: torch.from_numpy(b[k][rows]) for k in features}
    signature = sha256(backup_data)
    out, control = {}, None
    for relation, folder in zip(("off", "on"), folders, strict=True):
        cfg = json.loads((folder / "run.json").read_text())
        if cfg["arm"] != f"{relation}-mixture" or cfg["data_sha256"] != signature:
            raise ValueError(f"Wrong arm/input view: {folder}")
        comparable = {k: v for k, v in cfg.items() if k not in ("arm", "best_epoch")}
        if control is not None and control != comparable:
            raise ValueError("Alternate actor runs have different fitting protocols")
        control = comparable
        states = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        model = CandidateMLP(states["event.weight"].shape[0]).eval()
        model.load_state_dict(states)
        probs = [
            model({k: v[a:a + 512] for k, v in table.items()}, relation == "on", "mixture")[0].exp().numpy()
            for a in range(0, len(rows), 512)
        ]
        q = np.concatenate(probs)
        with np.load(folder / "validation.npz", allow_pickle=False) as fitted:
            for k, array in dict(event=ek, match=mk, label=base_y).items():
                np.testing.assert_array_equal(fitted[k], array)
            np.testing.assert_allclose(-np.log(q[np.arange(len(q)), slot]), fitted["scores"][:, 0],
                                       atol=3e-5, rtol=1e-5)
        out[relation] = reorder_q(players, base_player[take], q[take])
    return out, control


def score(p, y):
    return -np.log(p[np.arange(len(y)), y].astype(np.float64))


def folded(p):
    out = p[:, :10].copy()
    out[:, 0] += p[:, 10:].sum(1)
    return out


def paired(delta, match):
    _, group, n = np.unique(match, return_inverse=True, return_counts=True)
    sums = np.bincount(group, weights=delta)
    batch = np.random.default_rng(81345).integers(0, len(n), (10000, len(n)))
    return {"mean": float(delta.mean()), "ci95": np.quantile(sums[batch].sum(1) /
                                                              n[batch].sum(1), [.025, .975]).tolist()}


def run(a):
    sys.path.insert(0, str(a.canonical.resolve()))
    from build import ACTION13_NAMES
    from train_events import EventModel, action_profiles

    with np.load(a.data, allow_pickle=False) as d:
        np.testing.assert_array_equal(d["action13_names"], ACTION13_NAMES)
        match, event, val = d["match"], d["event"], d["val"]
        slot, zone, player, group, action = [d[k] for k in ("slot", "zone", "player", "group", "action13")]
        valid = np.flatnonzero(val & (slot >= 0))
        dev = {k: torch.from_numpy(d[k][valid]) for k in ("token", "wzone", "context", "age")}
    keys = ids(match[valid], event[valid])
    labels = action[valid]
    label10 = np.where(labels >= 10, 0, labels)
    players = player[valid]
    act_slot = slot[valid]
    true_players = players[np.arange(len(valid)), act_slot]
    y = (labels, label10)
    roster = players != 0
    with np.load(a.archive / "participants.npz", allow_pickle=False) as p:
        archived = exact(keys, ids(p["match"], p["event"]))
        np.testing.assert_array_equal(players, p["player"][archived])
        joint = p["joint"][archived]
    np.testing.assert_allclose(joint.sum((1, 2)), 1, atol=1e-6)
    q_old = joint.sum(2)
    alternatives, cfg = alternate(a.backup_data, (a.off, a.on), keys, players, true_players, label10)
    qs = dict(old=q_old, **alternatives)
    crossed, exchangeable = role_counterfactuals(qs["off"], qs["on"], group[valid], roster)
    all_qs = dict(qs, **crossed)
    weights = {name: torch.from_numpy(replace_actor(joint, q, roster)) for name, q in all_qs.items()}
    _, _, rp, pp, person_key = action_profiles(pd.read_parquet(a.matches),
                                                match, val, slot, zone, action, player, group)
    losses, replay = {}, {}
    for arm, lookup, index in (("role", rp, group[valid]), ("personal", pp, person_key[valid])):
        model = EventModel(13).eval()
        folder = a.archive / arm
        model.load_state_dict(torch.load(folder / "best.pt", weights_only=True, map_location="cpu"))
        accum = {k: [] for k in weights}
        with torch.inference_mode():
            offset = model.area.weight + model.profile(torch.from_numpy(lookup))
            for ix in torch.arange(len(valid)).split(512):
                branch = (model.context_logits(dev, ix)[:, None, None] +
                          offset[torch.from_numpy(index[ix.numpy()]).long()]).softmax(-1)
                for key, w in weights.items():
                    accum[key].append((w[ix, :, :, None] * branch).sum((1, 2)).numpy())
        probs = {k: np.concatenate(v) for k, v in accum.items()}
        with np.load(folder / "validation.npz", allow_pickle=False) as p:
            target = p["predicted"][exact(keys, ids(p["match"], p["event"]))]
        np.testing.assert_allclose(probs["old"], target, atol=2e-6, rtol=2e-5)
        replay[arm] = float(np.max(np.abs(probs["old"] - target)))
        for k, p in probs.items():
            np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
            losses[arm, k] = (score(p, y[0]), score(folded(p), y[1]))
    m = match[valid]
    report = {"reachable_development": len(valid), "matches": len(np.unique(m)), "backup_fit": cfg,
              "replay_max_abs": replay, "actor_nll": {}, "action": {}}
    for k, q in qs.items():
        report["actor_nll"][k] = float(score(q, act_slot).mean())
    order = ("off", "q01", "q10", "on")
    report["same_known_role_rows"] = int(exchangeable.sum())
    report["actor_role_decomposition"] = role_decompose(*(score(all_qs[k], act_slot) for k in order), m)
    report["actor_ownership_gain"] = paired(score(qs["off"], act_slot) - score(qs["on"], act_slot), m)
    for j, granularity in enumerate(("13", "10")):
        l = {k: v[j] for k, v in losses.items()}
        gain = {k: l["role", k] - l["personal", k] for k in qs}
        report["action"][granularity] = {
            "nll": {arm: {k: float(l[arm, k].mean()) for k in qs} for arm in ("role", "personal")},
            "role_decomposition": {arm: role_decompose(*(l[arm, k] for k in order), m)
                                   for arm in ("role", "personal")},
            "personal_gain": {k: paired(v, m) for k, v in gain.items()},
            "ownership_gain": {arm: paired(l[arm, "off"] - l[arm, "on"], m)
                               for arm in ("role", "personal")},
            "ownership_by_personal_interaction": paired(gain["on"] - gain["off"], m),
            "old_to_on_gain": {arm: paired(l[arm, "old"] - l[arm, "on"], m)
                               for arm in ("role", "personal")}
        }
    with a.out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)


def self_test():
    q = reorder_q(np.array([[7, 5, 0]]), np.array([[0, 5, 7]]), np.array([[0, .3, .7]]))
    np.testing.assert_allclose(q, [[.7, .3, 0]])
    old = np.array([[[.12, .48], [.3, .1], [0., 0.]]])
    new = replace_actor(old, q, q > 0)
    np.testing.assert_allclose(new, [[[.14, .56], [.225, .075], [0, 0]]])
    np.testing.assert_allclose(folded(np.ones((1, 13))/13)[0, 0], 4/13)
    try:
        reorder_q(np.array([[7, 5, 0]]), np.array([[0, 5, 9]]), q)
    except AssertionError:
        pass
    else:
        raise AssertionError("Mismatched roster accepted")
    q0 = np.array([[.20, .30, .15, .35, 0]])
    q1 = np.array([[.45, .15, .30, .10, 0]])
    cross, exchangeable = role_counterfactuals(q0, q1, np.array([[1, 1, 7, 7, 7]]), q0 > 0)
    np.testing.assert_allclose(cross["q01"], [[.375, .125, .15, .35, 0]])
    np.testing.assert_allclose(cross["q10"], [[.24, .36, .30, .10, 0]])
    assert exchangeable.item()
    print("PASS: role decomposition with unknown-role singletons")
    print("PASS: order, zero padding, conditional zones, folded classes, mismatched roster")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    names = "canonical data matches archive backup-data off on out".split()
    for name in names:
        p.add_argument("--" + name, type=Path)
    args = p.parse_args()
    if args.self_test:
        self_test()
    elif all(getattr(args, name.replace("-", "_")) is not None for name in names):
        torch.set_num_threads(1)
        run(args)
    else:
        p.error("Supply all archive/input arguments or --self-test.")
