"""Report paired development-only contrasts on a fixed, action-weighted match cohort."""
import argparse
import json
from pathlib import Path

import numpy as np

from train import METRICS

ARMS = ("off-context", "on-context", "off-mixture", "on-mixture")


def compare(delta, matches, samples):
    _, ind = np.unique(matches, return_inverse=True)
    n = np.bincount(ind)
    if len(n) < 2:
        raise ValueError("At least two matches are required for a match bootstrap.")
    mass = np.bincount(ind, weights=delta.mean(0))
    draw = np.random.default_rng(20260715).integers(len(n), size=(samples, len(n)))
    boot = mass[draw].sum(1) / n[draw].sum(1)
    return dict(gain=float(delta.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
                by_seed=delta.mean(1).tolist(), matches=len(n), actions=len(ind))


def classification(label, predicted, count):
    m = np.bincount(count * label + predicted, minlength=count * count).reshape(count, count)
    p, t = m.sum(0), m.sum(1)
    f1 = np.divide(2 * m.diagonal(), p + t, out=np.zeros(count), where=p + t > 0)
    return {"macro_f1": float(f1.mean()), "f1_by_class": f1.tolist(), "confusion": m.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, type=Path, help="A single view's run root")
    p.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    p.add_argument("--seeds", nargs="+", type=int, default=(20260715, 20260716, 20260717))
    p.add_argument("--bootstrap", type=int, default=10_000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.bootstrap < 1 or len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        p.error("Use distinct arms and seeds and a positive bootstrap count.")
    data, summaries, reference = {}, {}, None
    acc = {"actor_top1", "actor_top3", "event_acc"}
    for arm in args.arms:
        arr, classes, epochs = [], [], []
        for seed in args.seeds:
            folder = args.runs / arm / str(seed)
            run = json.loads((folder / "run.json").read_text())
            if run["seed"] != seed or run["arm"] != arm:
                raise ValueError(f"Run metadata mismatch: {folder}")
            epochs.append(run["best_epoch"])
            sig = {k: v for k, v in run.items() if k not in ("arm", "seed", "best_epoch")}
            with np.load(folder / "validation.npz", allow_pickle=False) as f:
                event, match, label, swap = (f[k] for k in ("event", "match", "label", "exchangeable"))
                if reference is None:
                    reference = sig, event, match, label, swap
                if sig != reference[0]:
                    raise ValueError("Training protocol/data SHA differs between runs.")
                for u, v in zip((event, match, label, swap), reference[1:]):
                    np.testing.assert_array_equal(u, v)
                arr.append(f["scores"].astype(float))
                classes.append(classification(label.astype(int), f["predicted"], len(sig["labels"])))
        x = data[arm] = np.stack(arr)
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite scores: {arm}")
        summaries[arm] = dict(means=dict(zip(METRICS, x.mean((0, 1)).tolist())),
                              classification_by_seed=classes, chosen_epochs=epochs)
    sig, _, matches, _, exchangeable = reference
    contrasts = {}
    for base, test in (("off-context", "on-context"), ("off-mixture", "on-mixture"),
                       ("off-context", "off-mixture"), ("on-context", "on-mixture")):
        if base not in data or test not in data:
            continue
        d = data[base] - data[test]
        contrasts[f"{test}_vs_{base}"] = {
            k: compare(-d[:, :, j] if k in acc else d[:, :, j], matches, args.bootstrap)
            for j, k in enumerate(METRICS[:7])
        }
    diagnostics = {}
    lookup = dict(zip(METRICS, range(len(METRICS))))
    for arm, x in data.items():
        if not arm.endswith("mixture") or not sig["event_weight"]:
            continue
        p_nll = x[:, :, lookup["event_nll"]]
        ds = {k: compare(x[:, :, lookup[k]] - p_nll, matches, args.bootstrap) for k in (
            "conditional_event_nll_oracle", "event_nll_hard_top1_mixture_ties", "event_nll_uniform")}
        if len(np.unique(matches[exchangeable])) >= 2:
            d = (x[:, :, lookup["event_nll_role_permuted"]] - p_nll)[:, exchangeable]
            ds["event_nll_role_permuted_exchangeable_only"] = compare(d, matches[exchangeable], args.bootstrap)
        diagnostics[arm] = ds
    report = dict(split="development_same_as_checkpoint_selection", seeds=list(args.seeds),
                  protocol=sig, exchangeable_actions=int(exchangeable.sum()), arms=summaries,
                  paired_gain_positive_is_better=contrasts, diagnostics_delta_alt_minus_prediction=diagnostics)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Saved {args.out}")
    for arm, v in summaries.items():
        m = v["means"]
        print(f"{arm:12s} actor NLL={m['actor_nll']:.4f} actor Top-1={m['actor_top1']:.2%} event NLL={m['event_nll']:.4f}")


if __name__ == "__main__":
    main()
