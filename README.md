# MLB free-agency auction simulator

Simulation companion to the LaTeX paper. Everything here runs on **synthetic data**; the
numbers are for checking the machinery and for directional intuition only.

## Files

| File | Purpose |
|---|---|
| `fa_sim.py` | Model: market generator, slot-assignment valuations, demand oracles, ascending auction, efficient benchmark + VCG |
| `validate.py` | Checks the auction against the efficient / VCG benchmark (`python3 validate.py 10`) |
| `run_experiments.py` | `expansion` (30 vs 32 teams) and `policy` (cap / floor) experiments, paired by seed |
| `results/` | CSVs written by the experiments (`*_raw.csv`, `*_paired.csv`) |

## Mapping to the paper

| Paper | Code |
|---|---|
| Teams N, free agents S, reserve r_j | `Market.n`, `Market.m`, `Market.r` |
| Slots, roster R_i | `SLOTS` (16 slots), `Market.B[i]` (incumbent value in each slot) |
| A_i(R) = best slot assignment | `assignment_value` (max-weight matching, `scipy.optimize.linear_sum_assignment`) |
| v_i(T) = A_i(R_i + T) - A_i(R_i) | `marginal_value` |
| C_i (dollars per expected win) | `Market.C` (values are stored already multiplied by C_i) |
| Gross-substitutes demand | `demand_unconstrained`: one matching problem per team per round, exact |
| Ascending auction | `ascending_auction`: Kelso-Crawford style, provisional holders, eps = $0.05M |
| Min Walrasian prices / VCG | `vcg_payments` (Clarke pivot, reserve-adjusted) |
| Cap K, floor F | `make_budget_demand(cap=..., floor=...)` |

Floor modelling choice: a team below the floor pays the shortfall anyway, so its cost of signing
players is max(spend, F - P0_i). Shortfall dollars are reported separately (`shortfall`) and are
**not** counted as player surplus; where they go (players' pool, league) is a policy choice.

## Validation (10 seeds, 30 teams, 60 free agents)

* Auction converged on every seed with zero releases (as gross substitutes predicts).
* Welfare gap to the efficient allocation: at most $0.06M.
* Maximum team regret at final prices: at most $0.14M (about 3 eps), i.e. an eps-Walrasian equilibrium.
* Total spend within 0.2% of the VCG benchmark.
* A few seeds allocate 1-7 players differently from the efficient allocation; these are near-ties
  (welfare differences under $0.07M), so per-team payments are not comparable on those seeds.
* Cap runs: 3 seeds checked, max regret $0.6M (about 2 eps), 0-2 items unsold above reserve. Under the
  cap the auction releases items (about 165 per run), the observable symptom of gross substitutes failing.

## Preliminary results (synthetic data; paired over seeds, mean +/- SE)

**Expansion, +2 teams with 11 of 16 slots open (20 seeds):** total spend +$28.2M +/- 5.7 (+2.5%),
top-10 player price +$0.8M +/- 0.3 (+1.4%), price per marginal WAR +1.9%, player surplus +$27.7M,
surplus of the original 30 teams -$26.8M +/- 5.4 (-13%), welfare +$17.4M. Effects shrink with fewer
open slots on the new teams (6/16 open: spend +$15.0M +/- 3.3, 10 seeds).

**Cap at the 85th percentile of baseline payroll (8 seeds):** spend -$78.3M +/- 11.7 (-6.8%),
top-10 price -6.7%, welfare -$79.8M +/- 16.4 (-6.2%), team surplus -$2.7M +/- 7.1 (not distinguishable
from zero), max payroll -$116M. Mostly a transfer from players to teams that destroys some welfare.

**Floor at the 15th percentile:** small price effect (spend +$4.4M +/- 2.0), welfare -$2.3M +/- 1.1,
but below-floor teams forfeit $72.9M +/- 6.1 in shortfall payments.

## Known limitations

1. **All inputs are synthetic.** Placeholders to replace: `c0` ($8M per win), `r0` (reserve $0.8M),
   `salary_frac`, the WAR distributions, position mix, team-fit noise, and open-slot counts.
2. The cap and floor levels are quantiles of the simulated baseline payroll distribution, not
   CBA numbers. Results are sensitive to these levels (sweep not yet run).
3. Single season, simultaneous bids, risk-neutral teams, expected WAR only (the distributions D_ij
   enter only through their means).
4. The budget-constrained demand oracle is exact only when a team's constraint does not bind;
   otherwise it enumerates bundles from the best 8 candidates, up to 5 signings.
5. Auction prices are accurate to roughly eps; the cap runs use eps = $0.25M.

## Suggested next steps

1. Replace synthetic inputs with data: FanGraphs $/WAR for `c0`, a projection system (Steamer/ZiPS) for
   expected WAR and positions of the actual free-agent class, current payrolls for `P0`.
2. Sweep cap and floor levels and report welfare and price curves.
3. Check price monotonicity item by item when teams are added (the theory predicts weakly higher
   minimum Walrasian prices under gross substitutes).
4. Add ex-post realized WAR (draw from D_ij on [-5, 21]) to measure bargain/bust outcomes.
5. Add the luxury tax as a soft cap (tax rate on payroll above a threshold).
