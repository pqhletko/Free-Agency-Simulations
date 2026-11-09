"""Experiments.  Usage:
    python3 run_experiments.py expansion [n_seeds]
    python3 run_experiments.py policy    [n_seeds]
Outputs CSVs in ./results/ and prints paired-difference tables (same free-agent class in each arm).
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from fa_sim import build_market, ascending_auction, summarize, make_budget_demand

os.makedirs("results", exist_ok=True)
METRICS = ["spend", "top10_price", "price_per_marg_war", "player_surplus", "team_surplus",
           "team_surplus_orig", "team_surplus_net", "shortfall", "welfare", "n_signed", "payroll_max"]


def paired_table(df, arm_col, base_arm):
    rows = []
    base = df[df[arm_col] == base_arm].set_index("seed")
    for arm in df[arm_col].unique():
        if arm == base_arm:
            continue
        cur = df[df[arm_col] == arm].set_index("seed")
        for met in METRICS:
            d = (cur[met] - base[met]).dropna()
            rows.append(dict(arm=arm, metric=met, base_mean=base.loc[d.index, met].mean(), arm_mean=cur.loc[d.index, met].mean(),
                             diff=d.mean(), se=d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan))
    return pd.DataFrame(rows)


def expansion(n_seeds):
    arms = {"30 teams": dict(n_new=0), "32 teams (11/16 open)": dict(n_new=2, new_open=11)}
    sens = {"32 teams (6/16 open)": dict(n_new=2, new_open=6), "32 teams (16/16 open)": dict(n_new=2, new_open=16)}
    recs = []
    for s in range(n_seeds):
        for name, kw in {**arms, **(sens if s < 10 else {})}.items():
            mk = build_market(seed=s, **kw)
            res = ascending_auction(mk, eps=0.05)
            recs.append(dict(seed=s, arm=name, rounds=res.rounds, converged=res.converged,
                             **summarize(mk, res.p, res.alloc, first_n=30)))
        print(f"  seed {s} done", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv("results/expansion_raw.csv", index=False)
    tab = paired_table(df, "arm", "30 teams")
    tab.to_csv("results/expansion_paired.csv", index=False)
    return df, tab


def policy(n_seeds, eps=0.25, cap_q=0.85, floor_q=0.15):
    recs = []
    for s in range(n_seeds):
        t0 = time.time()
        mk = build_market(seed=s)
        base_fn = make_budget_demand(mk)
        res0 = ascending_auction(mk, demand_fn=base_fn, eps=eps, max_rounds=3000)
        sm0 = summarize(mk, res0.p, res0.alloc)
        pay0 = mk.P0 + np.array([res0.p[res0.alloc == i].sum() for i in range(mk.n)])
        cap, floor = float(np.quantile(pay0, cap_q)), float(np.quantile(pay0, floor_q))
        arms = {"baseline": (None, None), "cap": (cap, None), "floor": (None, floor), "cap+floor": (cap, floor)}
        for name, (c, f) in arms.items():
            if name == "baseline":
                res, sm = res0, sm0
            else:
                res = ascending_auction(mk, demand_fn=make_budget_demand(mk, cap=c, floor=f),
                                        eps=eps, max_rounds=3000)
                sm = summarize(mk, res.p, res.alloc, floor=f)
            pay = mk.P0 + np.array([res.p[res.alloc == i].sum() for i in range(mk.n)])
            recs.append(dict(seed=s, arm=name, cap=c, floor=f, rounds=res.rounds, converged=res.converged,
                             releases=res.releases, frac_at_cap=float((pay >= (c or np.inf) - eps).mean()),
                             frac_below_floor=float((pay < (f or -np.inf)).mean()), **sm))
        print(f"  seed {s} done in {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv("results/policy_raw.csv", index=False)
    tab = paired_table(df, "arm", "baseline")
    tab.to_csv("results/policy_paired.csv", index=False)
    return df, tab


if __name__ == "__main__":
    which = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x:,.2f}")
    df, tab = (expansion if which == "expansion" else policy)(n)
    print("\nconverged share by arm:\n", df.groupby("arm")["converged"].mean().to_string())
    print("\nPaired differences vs base arm (mean over seeds, SE):\n", tab.to_string(index=False))
