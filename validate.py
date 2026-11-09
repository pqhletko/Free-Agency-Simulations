"""Validate the ascending auction against the efficient / VCG benchmark.

Columns
  eff_gap     : optimal welfare minus auction welfare, $M (should be ~0)
  alloc_diff  : number of players allocated to a different team than in the efficient allocation
                (nonzero only when near-ties exist; check eff_gap and regret before worrying)
  max_regret  : largest utility any team could gain by switching to its best bundle at the final
                prices, $M. An eps-Walrasian equilibrium has regret <= roughly eps * bundle size.
  tot_pay     : total spend, auction vs. VCG (min-Walrasian) benchmark
"""
import sys
import time
import numpy as np
from fa_sim import (build_market, ascending_auction, vcg_payments, summarize,
                    demand_unconstrained, marginal_value)

seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
eps = 0.05
print(f"{'seed':>4} {'rounds':>6} {'conv':>5} {'releas':>6} {'eff_gap':>8} {'alloc_diff':>10} "
      f"{'max_regret':>10} {'tot_pay_auc':>11} {'tot_pay_vcg':>11} {'pay_gap%':>8} {'secs':>5}")
for s in seeds:
    t0 = time.time()
    mk = build_market(seed=s)
    res = ascending_auction(mk, eps=eps)
    W_star, alloc_star, pay_vcg = vcg_payments(mk)
    sm = summarize(mk, res.p, res.alloc)
    W_auc = sm["welfare"] + mk.base.sum()      # incl. incumbents, net of reserves (same basis as W_star)

    regret = 0.0
    for i in range(mk.n):
        T = np.flatnonzero(res.alloc == i)
        u_now = marginal_value(mk, i, T) - res.p[T].sum()
        D = np.flatnonzero(demand_unconstrained(mk, i, res.p, np.zeros(mk.m)))
        u_best = marginal_value(mk, i, D) - res.p[D].sum()
        regret = max(regret, u_best - u_now)

    pay_auc = np.array([res.p[res.alloc == i].sum() for i in range(mk.n)])
    gap = 100 * (pay_auc.sum() - pay_vcg.sum()) / pay_vcg.sum()
    print(f"{s:>4} {res.rounds:>6} {str(res.converged):>5} {res.releases:>6} {W_star - W_auc:>8.3f} "
          f"{int((res.alloc != alloc_star).sum()):>10} {regret:>10.3f} "
          f"{pay_auc.sum():>11.1f} {pay_vcg.sum():>11.1f} {gap:>8.2f} {time.time() - t0:>5.1f}")
