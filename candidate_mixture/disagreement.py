"""Measure evidence for candidate mixtures on high- vs low-disagreement events."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from blend_check import match_gain, read
from model import CandidateMLP, marginal
from train import FEATURES, fingerprint


def disagreement(q, r):
    """Actor--action mutual information implied by model probabilities, in nats."""
    pooled = marginal(q, r)
    gap = (r.exp() * (r - pooled[:, None])).sum(-1)
    return (q.exp() * gap).sum(-1).clamp_min(0)


def run(data, direct, mixture, device, per_match, batch_size):
    mcfg, event, match, label, mx = read(mixture)
    dcfg, ed, md, yd, dx = read(direct)
    excluded = {"arm", "best_epoch"}
    if {k: v for k, v in mcfg.items() if k not in excluded} != {k: v for k, v in dcfg.items() if k not in excluded}:
        raise ValueError("Checkpoint and direct arm need to use identical data, seed and schedule.")
    if not (mcfg["arm"].endswith("-mixture") and dcfg["arm"].endswith("-context")):
        raise ValueError("Pass a context and a mixture run.")
    if mcfg["data_sha256"] != fingerprint(data):
        raise ValueError("Input view does not match the saved run.")
    for x,y in ((event,ed),(match,md),(label,yd)):
        np.testing.assert_array_equal(x,y)
    with np.load(data, allow_pickle=False) as z:
        train = np.flatnonzero(~z["val"] & (z["slot"] >= 0))
        val = np.flatnonzero(z["val"] & (z["slot"] >= 0))
        for a,b in ((event,z["event"][val]),(match,z["match"][val]),(label,z["action"][val])):
            np.testing.assert_array_equal(a,b)
        mt = z["match"][train]
        if np.intersect1d(mt,match).size:
            raise ValueError("Train and development matches overlap.")
        # Fixed reproducible sample across all training matches; no observed action in the cutoff.
        rng = np.random.default_rng(20260715)
        sampled = np.sort(np.concatenate([rng.choice(train[mt == mid], min(per_match, int((mt == mid).sum())),
                                                     replace=False) for mid in np.unique(mt)]))
        rows = np.r_[sampled,val]
        feats = {k: torch.as_tensor(z[k][rows], device=device) for k in FEATURES}
    w = torch.load(mixture / "best.pt", map_location=device, weights_only=True)
    net = CandidateMLP(w["event.weight"].shape[0]).to(device)
    net.load_state_dict(w)
    net.eval()
    with torch.inference_mode():
        dv, action = [], []
        for i in range(0,len(rows),batch_size):
            x = {k: v[i:i+batch_size] for k,v in feats.items()}
            q, _, r = net(x, mcfg["arm"].startswith("on-"), "mixture")
            dv.append(disagreement(q,r).cpu().numpy())
            j = min(i+batch_size, len(rows))
            if j > len(sampled):
                s = max(len(sampled)-i, 0)
                yi = torch.as_tensor(label[max(0,i-len(sampled)):j-len(sampled)], device=device).long()
                action.append(-marginal(q[s:],r[s:]).gather(1,yi[:,None]).squeeze(1).cpu().numpy())
    np.testing.assert_allclose(np.concatenate(action), mx, rtol=1e-5, atol=3e-5)
    dv = np.concatenate(dv)
    cutoff = float(np.median(dv[:len(sampled)]))
    high = dv[len(sampled):] > cutoff
    report = dict(split="development_exploratory", train_median_nats=cutoff, train_rows=len(sampled),
                  dev_rows=len(val), total_matches=len(np.unique(match)), bins={})
    _, mi = np.unique(match,return_inverse=True)
    delta = dx - mx
    draws = np.random.default_rng(20260715).integers(mi.max()+1,size=(10_000,mi.max()+1))
    bootstrap = {}
    for key, mask in (("low",~high),("high",high)):
        count = np.bincount(mi[mask], minlength=mi.max()+1)
        mass = np.bincount(mi[mask],weights=delta[mask],minlength=len(count))
        den = count[draws].sum(1)
        bootstrap[key] = np.divide(mass[draws].sum(1),den,out=np.full(len(den),np.nan),where=den>0)
        blend = np.log(2)-np.logaddexp(-dx[mask], -mx[mask])
        report["bins"][key] = dict(events=int(mask.sum()), matches=int(np.count_nonzero(count)),
                                   direct_nll=float(dx[mask].mean()), mixture_nll=float(mx[mask].mean()),
                                   blend_nll=float(blend.mean()), mixture_gain=match_gain(delta[mask],match[mask]),
                                   blend_gain=match_gain(dx[mask]-blend,match[mask]))
    interaction = bootstrap["high"] - bootstrap["low"]
    report["interaction_high_minus_low"] = {
        "gain": float(delta[high].mean()-delta[~high].mean()),
        "ci95": np.nanquantile(interaction,[.025,.975]).tolist()}
    return report


def self_test():
    q = torch.tensor([[.5,.5],[.5,.5],[1.,0.]],dtype=torch.float64).log()
    r = torch.tensor([[[.7,.3],[.7,.3]],[[.9,.1],[.1,.9]],[[.9,.1],[.1,.9]]],dtype=torch.float64).log()
    d = disagreement(q,r)
    torch.testing.assert_close(d,torch.tensor([0.,.3680642071684971,0.],dtype=torch.float64))
    print("PASS: identical heads and a single possible actor give zero; opposing heads give positive D.")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("data","direct","mixture","out"):
        p.add_argument("--"+k,type=Path)
    p.add_argument("--device",default="cpu")
    p.add_argument("--per-match",type=int,default=32)
    p.add_argument("--batch",type=int,default=256)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        self_test()
    else:
        if not all((a.data,a.direct,a.mixture,a.out,a.per_match>0,a.batch>0)):
            p.error("Pass data, direct/mixture runs, output, and positive sampling/batch sizes.")
        torch.set_num_threads(1)
        result=run(a.data,a.direct,a.mixture,torch.device(a.device),a.per_match,a.batch)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open("x") as f:
            json.dump(result,f,indent=2,allow_nan=False)
        print(json.dumps(result,indent=2,allow_nan=False))
