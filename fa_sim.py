"""
fa_sim.py -- MLB free agency as a multi-item market with assignment (slot) valuations.

Model mapping to the paper
--------------------------
* Teams i in N are buyers; free agents j in S are indivisible items with reserve price r_j.
* Team i has K roster slots. An incumbent/free agent j placed in slot k contributes
  C_i * a_ij * elig(pos_j, k) dollars of value (a_ij = expected WAR for team i).
* A_i(R) = max-weight matching of players in R to slots (empty slot = replacement level = 0).
* v_i(T) = A_i(R_i + T) - A_i(R_i)   (marginal value; gross-substitutes assignment valuation).
* Quasilinear utility u_i(T, p) = v_i(T) - sum_{j in T} p_j   (C_i is folded into the values).

All dollar amounts are in $M. Every number produced by `build_market` is SYNTHETIC; see README.
"""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------------------------------------------
# Roster structure
# --------------------------------------------------------------------------------------
SLOTS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "DH",
         "SP", "SP", "SP", "SP", "SP", "RP", "RP"]
K = len(SLOTS)
HITTERS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]
PRIMARY = HITTERS + ["SP", "RP"]
_SECONDARY = {("SP", "RP"): 0.6, ("CF", "LF"): 0.95, ("CF", "RF"): 0.95, ("LF", "RF"): 0.9,
              ("RF", "LF"): 0.9, ("SS", "2B"): 0.95, ("SS", "3B"): 0.95, ("2B", "SS"): 0.85,
              ("2B", "3B"): 0.85, ("3B", "2B"): 0.85, ("3B", "1B"): 0.9}


def elig(pos: str, slot: str) -> float:
    """Fraction of a player's WAR retained when placed in `slot` (0 = cannot play there)."""
    if pos == slot:
        return 1.0
    if pos in HITTERS and slot == "DH":
        return 0.7
    if pos in HITTERS and pos != "C" and slot == "1B":
        return 0.8
    return _SECONDARY.get((pos, slot), 0.0)


def elig_vec(pos: str) -> np.ndarray:
    return np.array([elig(pos, s) for s in SLOTS])


# --------------------------------------------------------------------------------------
# Market
# --------------------------------------------------------------------------------------
@dataclass
class Market:
    n: int            # number of teams
    m: int            # number of free agents
    C: np.ndarray     # (n,)       $M per expected WAR, team specific
    B: np.ndarray     # (n,K,K)    incumbent l placed in slot k, value in $M
    V: np.ndarray     # (n,m,K)    free agent j in slot k for team i, value in $M
    r: np.ndarray     # (m,)       reserve prices
    P0: np.ndarray    # (n,)       payroll committed to existing roster
    base: np.ndarray  # (n,)       A_i(R_i) in $M
    pos: np.ndarray   # (m,)
    mu: np.ndarray    # (m,)       league-average expected WAR of each free agent


def assignment_value(M: np.ndarray) -> float:
    ri, ci = linear_sum_assignment(M, maximize=True)
    return float(M[ri, ci].sum())


def build_market(seed=0, n_teams=30, m=60, n_new=0, new_open=11, c0=8.0, r0=0.8,
                 salary_frac=0.6) -> Market:
    """Synthetic market. Teams 0..n_teams-1 are identical across calls with different n_new
    (per-team RNG streams), so 30-team vs 32-team runs differ only by the added teams.

    c0 : $M per win (PLACEHOLDER -- replace with the FanGraphs $/WAR figure)
    r0 : reserve price in $M (PLACEHOLDER -- league-minimum-type outside option)
    new_open : number of the 16 slots left open on each expansion team
    """
    rf = np.random.default_rng([seed, 9999])
    probs = np.array([0.05, 0.06, 0.06, 0.06, 0.06, 0.07, 0.06, 0.07, 0.26, 0.25])
    pos = rf.choice(PRIMARY, size=m, p=probs / probs.sum())
    mu = rf.gamma(1.2, 1.3, size=m)
    stars = rf.choice(m, size=min(6, m), replace=False)
    mu[stars] = rf.uniform(4.0, 7.0, size=len(stars))
    E = np.vstack([elig_vec(p) for p in pos])                       # (m,K)
    slot_E = np.vstack([elig_vec(s) for s in SLOTS])                # incumbent at slot l has primary pos = slot

    n = n_teams + n_new
    C = np.empty(n); B = np.zeros((n, K, K)); V = np.zeros((n, m, K)); P0 = np.empty(n)
    base = np.empty(n)
    for i in range(n):
        rt = np.random.default_rng([seed, i])
        C[i] = c0 * max(0.5, 1 + 0.25 * rt.standard_normal())
        quality = max(0.6, 1 + 0.2 * rt.standard_normal())
        war = rt.gamma(2.0, 1.0, size=K) * quality
        n_open = new_open if i >= n_teams else int(rt.integers(2, 5))
        war[rt.choice(K, size=n_open, replace=False)] = 0.0          # expiring contracts / empty slots
        fit = np.maximum(0.0, 1 + 0.15 * rt.standard_normal(m))      # team-specific fit
        B[i] = C[i] * war[:, None] * slot_E
        V[i] = C[i] * (mu * fit)[:, None] * E
        P0[i] = salary_frac * c0 * war.sum() * np.exp(0.2 * rt.standard_normal())
        base[i] = assignment_value(B[i])
    return Market(n, m, C, B, V, np.full(m, r0), P0, base, pos, mu)


# --------------------------------------------------------------------------------------
# Demand oracles
# --------------------------------------------------------------------------------------
def demand_unconstrained(mk: Market, i: int, p: np.ndarray, hold: np.ndarray) -> np.ndarray:
    """Exact demand oracle for assignment valuations: one max-weight matching of
    (incumbents + free agents priced at p) to slots. `hold` breaks ties toward current holdings."""
    Vi = mk.V[i]
    W = np.where(Vi > 0, np.maximum(Vi - p[:, None] + 1e-9 * hold[:, None], 0.0), 0.0)
    M = np.vstack([mk.B[i], W])
    ri, ci = linear_sum_assignment(M, maximize=True)
    out = np.zeros(mk.m, dtype=bool)
    sel = ri >= K
    out[ri[sel] - K] = M[ri[sel], ci[sel]] > 1e-12
    return out


def marginal_value(mk: Market, i: int, T) -> float:
    """v_i(T) = A_i(R_i + T) - A_i(R_i), gross of prices."""
    T = list(T)
    if not T:
        return 0.0
    return assignment_value(np.vstack([mk.B[i], mk.V[i][T]])) - mk.base[i]


def make_budget_demand(mk: Market, cap=None, floor=None, pool=8, kmax=5):
    """Demand oracle with payroll cap and/or floor.
      cap   : payroll P0_i + sum p_j must be <= cap
      floor : a team below the floor pays the shortfall anyway, so its cost of signing is
              max(spend, floor - P0_i)  (hard floor, shortfall forfeited)

    Hybrid: if the unconstrained demand D0 is still optimal -- cap slack, and either no floor
    gap or spend(D0) >= gap (the floor only makes cheaper bundles look better, never D0 worse
    than the unconstrained optimum) -- return D0 exactly. Otherwise enumerate bundles
    (approximate: `pool` players with the best single-player marginal net value, at most
    `kmax` signings). Valuations are submodular, so a player with v_i({j}) - p_j <= 0 can never
    improve a bundle; the pool ranking is lossless apart from truncation.
    """
    mv1 = np.array([[marginal_value(mk, i, [j]) for j in range(mk.m)] for i in range(mk.n)])

    def demand(i: int, p: np.ndarray, hold: np.ndarray) -> np.ndarray:
        D0 = demand_unconstrained(mk, i, p, hold)
        spend0 = float(p[D0].sum())
        gap0 = max(0.0, floor - mk.P0[i]) if floor is not None else 0.0
        if (cap is None or mk.P0[i] + spend0 <= cap + 1e-12) and (gap0 == 0.0 or spend0 >= gap0):
            return D0
        out = np.zeros(mk.m, dtype=bool)
        net = mv1[i] - p
        cand = np.argsort(-net)[:pool]
        cand = cand[net[cand] > 0]
        gap = max(0.0, floor - mk.P0[i]) if floor is not None else 0.0
        best_u, best_T = -max(0.0, gap), ()                         # empty set
        for k in range(1, min(kmax, len(cand)) + 1):
            for T in combinations(cand, k):
                spend = float(p[list(T)].sum())
                if cap is not None and mk.P0[i] + spend > cap:
                    continue
                u = marginal_value(mk, i, T) - max(spend, gap) + 1e-9 * hold[list(T)].sum()
                if u > best_u + 1e-12:
                    best_u, best_T = u, T
        out[list(best_T)] = True
        return out
    return demand


# --------------------------------------------------------------------------------------
# Ascending auction
# --------------------------------------------------------------------------------------
@dataclass
class AuctionResult:
    p: np.ndarray
    alloc: np.ndarray        # (m,) holder team index, -1 if unsold
    rounds: int
    converged: bool
    releases: int            # times a team dropped an item it held (0 under gross substitutes)


def ascending_auction(mk: Market, demand_fn=None, eps=0.05, max_rounds=20000, seed=0) -> AuctionResult:
    """Kelso-Crawford style ascending auction with provisional holders.

    Each item has a standing bid p_j and a holder. Every round each team computes its demand at
    the prices it would have to pay (p_j for items it holds, p_j + eps for items held by someone
    else, p_j for unheld items), and bids on demanded items it does not hold. Each item with new
    bids goes to a random bidder at the raised price. Stops when there are no new bids.
    Under gross substitutes a holder never wants to drop an item; if it does (e.g. with a binding
    cap), the item is released and the event is counted.
    """
    if demand_fn is None:
        demand_fn = lambda i, q, h: demand_unconstrained(mk, i, q, h)
    rng = np.random.default_rng(seed)
    p = mk.r.copy()
    holder = -np.ones(mk.m, dtype=int)
    releases = 0
    converged = False
    for t in range(1, max_rounds + 1):
        bids = {}
        dropped = []
        for i in range(mk.n):
            held = holder == i
            q = np.where(holder == -1, p, np.where(held, p, p + eps))
            D = demand_fn(i, q, held.astype(float))
            for j in np.flatnonzero(D & ~held):
                bids.setdefault(int(j), []).append(i)
            dropped += [(i, int(j)) for j in np.flatnonzero(held & ~D)]
        changed = False
        for j, who in bids.items():
            if holder[j] != -1:
                p[j] += eps
            holder[j] = who[rng.integers(len(who))]
            changed = True
        for i, j in dropped:
            if j not in bids and holder[j] == i:
                holder[j] = -1
                releases += 1
                changed = True
        if not changed:
            converged = True
            break
    return AuctionResult(p, holder.copy(), t, converged, releases)


# --------------------------------------------------------------------------------------
# Efficient benchmark and VCG
# --------------------------------------------------------------------------------------
def efficient_allocation(mk: Market, exclude=None):
    """Welfare-maximizing allocation of free agents (net of reserves) as one big assignment
    problem. Returns (total welfare, alloc, per-team matched value)."""
    teams = [i for i in range(mk.n) if i != exclude]
    nt = len(teams)
    M = np.zeros((mk.m + nt * K, nt * K))
    for a, i in enumerate(teams):
        cs = slice(a * K, (a + 1) * K)
        M[:mk.m, cs] = np.where(mk.V[i] > 0, np.maximum(mk.V[i] - mk.r[:, None], 0.0), 0.0)
        M[mk.m + a * K: mk.m + (a + 1) * K, cs] = mk.B[i]
    ri, ci = linear_sum_assignment(M, maximize=True)
    alloc = -np.ones(mk.m, dtype=int)
    val = {i: 0.0 for i in teams}
    for r_, c_ in zip(ri, ci):
        val[teams[c_ // K]] += M[r_, c_]
        if r_ < mk.m and M[r_, c_] > 1e-12:
            alloc[r_] = teams[c_ // K]
    return float(M[ri, ci].sum()), alloc, val


def vcg_payments(mk: Market):
    """Clarke-pivot payments (reserve-adjusted). Equal to min Walrasian payments for GS valuations."""
    W, alloc, val = efficient_allocation(mk)
    pay = np.zeros(mk.n)
    for i in range(mk.n):
        W_minus, _, _ = efficient_allocation(mk, exclude=i)
        pay[i] = W_minus - (W - val[i]) + mk.r[alloc == i].sum()
    return W, alloc, pay


# --------------------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------------------
def summarize(mk: Market, p: np.ndarray, alloc: np.ndarray, first_n=None, floor=None) -> dict:
    """first_n: report team surplus of teams 0..first_n-1 separately (e.g. the original 30 teams
    when expansion teams have been appended).
    floor: payroll floor; shortfall = total dollars teams must pay to reach it without signing
    anyone (a transfer, reported separately and netted out of `team_surplus_net`)."""
    signed = alloc >= 0
    spend = float(p[signed].sum())
    gross = np.zeros(mk.n)
    spend_i = np.zeros(mk.n)
    mwar = 0.0
    for i in range(mk.n):
        T = np.flatnonzero(alloc == i)
        gross[i] = marginal_value(mk, i, T)
        spend_i[i] = p[T].sum()
        mwar += gross[i] / mk.C[i]
    k = first_n or mk.n
    top10 = np.argsort(-mk.mu)[:10]
    payroll = mk.P0 + np.array([p[alloc == i].sum() for i in range(mk.n)])
    return dict(
        n_signed=int(signed.sum()), spend=spend,
        player_surplus=float((p[signed] - mk.r[signed]).sum()),
        team_surplus=float(gross.sum() - spend),
        team_surplus_orig=float(gross[:k].sum() - spend_i[:k].sum()),
        shortfall=float(np.maximum(0.0, floor - (mk.P0 + spend_i)).sum()) if floor is not None else 0.0,
        team_surplus_net=float(gross.sum() - spend
                               - (np.maximum(0.0, floor - (mk.P0 + spend_i)).sum() if floor is not None else 0.0)),
        welfare=float(gross.sum() - mk.r[signed].sum()),
        price_per_marg_war=spend / mwar if mwar > 0 else np.nan,
        top10_price=float(p[top10].mean()),
        payroll_mean=float(payroll.mean()), payroll_max=float(payroll.max()),
        payroll_min=float(payroll.min()),
    )
