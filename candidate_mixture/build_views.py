"""Extend upstream's non-test view with visible owner IDs and supervised action labels."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


def visible_window(base, events, query, lag):
    """Same query-only window as upstream inputs(), oldest first, -1 for padding."""
    team = query.team_id.iloc[0]
    time = events.period.to_numpy() * 10000 + events.sec.to_numpy()
    qt = query.period.to_numpy() * 10000 + query.sec.to_numpy()
    own = events.team_id.to_numpy() == team
    onball = events.event_name.isin(base.ON_BALL).to_numpy() | (
        events.sub_event_name.to_numpy() == "Ground attacking duel"
    )
    early = base.preceding(np.arange(len(time)), time, qt - 2)
    recent = base.preceding(np.flatnonzero(own & onball), time, qt)
    recent[(recent < 0) | (time[recent] < qt[:, None] - 2) | lag] = -1
    return np.sort(np.concatenate((early, recent), axis=1), axis=1)[:, -base.K:]


def extend(base, original, events, query, roster, subs, lag):
    x = original(events, query, roster, subs, lag)
    w = visible_window(base, events, query, lag)
    own = events.team_id.to_numpy() == query.team_id.iloc[0]
    vocab = pd.Index(base.VOCAB["sub_event_name"])
    code = vocab.get_indexer(events.sub_event_name)
    expected = np.where(w >= 0, 1 + code[w] + len(vocab) * (~own[w]), 0)
    np.testing.assert_array_equal(x["token"], expected)
    # Only candidate-team IDs can match candidates. Padding and unknown owners are 0.
    owner = events.player_id.to_numpy()[w]
    x["window_actor"] = np.where((w >= 0) & own[w] & (owner > 0), owner, 0).astype(np.int32)
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True, help="Processed England_nontest directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--view", required=True, choices=("main", "lag"))
    p.add_argument("--reference", type=Path, help="Optional archived upstream NPZ for exact feature comparison")
    a = p.parse_args()
    sys.path.insert(0, str(a.upstream.resolve() / "onball_participation"))
    import build as base

    model = a.data / "model"
    game = pd.read_parquet(model / "matches.parquet", columns=["match_id", "block"])
    if set(game.block.unique()) != {"model_train", "model_validation"}:
        raise ValueError("Only the existing train/development split is allowed.")
    validation = set(game.loc[game.block == "model_validation", "match_id"])
    meta = pd.read_parquet(model / "targets_meta.parquet", columns=["event_id", "match_id"])
    label = pd.read_parquet(model / "targets_label.parquet", columns=["action", "player_id"])
    if len(meta) != len(label) or meta.event_id.duplicated().any():
        raise ValueError("Targets and labels are not uniquely row-aligned.")
    y = pd.Index(base.ACTIONS).get_indexer(label.action).astype(np.int8)
    if (y < 0).any():
        raise ValueError("Unknown target class; check the upstream action vocabulary.")

    a.out.mkdir(parents=True, exist_ok=True)
    dest = a.out / (a.view + ".npz")
    if dest.exists():
        raise FileExistsError(dest)
    original = base.inputs
    with tempfile.TemporaryDirectory(dir=a.out) as tmp:
        argv = ["build.py", "--data", str(a.data), "--out", tmp, "--view", a.view]
        with patch.object(sys, "argv", argv), patch.object(
            base, "inputs", lambda *args: extend(base, original, *args)
        ):
            base.main()
        src = Path(tmp) / dest.name
        complete = Path(tmp) / ("complete_" + dest.name)
        with np.load(src) as z:
            np.testing.assert_array_equal(z["event"], meta.event_id)
            np.testing.assert_array_equal(z["match"], meta.match_id)
            np.testing.assert_array_equal(z["val"], meta.match_id.isin(validation))
            reached = np.flatnonzero(z["slot"] >= 0)
            np.testing.assert_array_equal(z["player"][reached, z["slot"][reached]], label.player_id.iloc[reached])
            if a.reference:
                with np.load(a.reference) as ref:
                    if not set(ref.files) <= set(z.files):
                        raise ValueError("Archive has keys missing from rebuilt input.")
                    for name in ref.files:
                        np.testing.assert_array_equal(z[name], ref[name], err_msg=name)
            np.savez(complete, **dict(z), action=y, action_names=np.array(base.ACTIONS))
        complete.replace(dest)
    print(json.dumps({"view": a.view, "rows": len(y), "reached": len(reached), "saved": str(dest)}))


if __name__ == "__main__":
    main()
