import math
import time
import random
from itertools import combinations

import numpy as np

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from constants import TIMEOUT






def search_space_at_k(n: int, k: int) -> int:
    """Number of subsets of size k: C(n, k)."""
    return math.comb(n, k)


def search_space_window(n: int, k_lo: int, k_hi: int) -> int:
    """Total subsets in [k_lo, k_hi]: Σ C(n,k)."""
    return sum(math.comb(n, k) for k in range(k_lo, k_hi + 1))


def _enumerate_weight_k(
    instance: SubsetSumInstance,
    k: int,
    start_time: float,
) -> list[int] | None:
    """
    Enumerate all subsets of size k and check if any sums to T.
    Returns solution vector or None.
    """
    n       = instance.n
    weights = instance.weights
    T       = instance.target

    for combo in combinations(range(n), k):
        if time.perf_counter() - start_time > TIMEOUT:
            return None
        s = sum(weights[i] for i in combo)
        if s == T:
            x = [0] * n
            for i in combo:
                x[i] = 1
            return x
    return None



def analyze_k_distribution(
    n: int,
    n_samples: int = 1000,
    d_min: float = 0.45,
    d_max: float = 1.50,
) -> None:
    """
    Analyze k_lo, k_hi and C(n, k_lo) distribution across densities.
    Shows when greedy extreme enumeration is feasible.
    """
    print(f"\n{'='*70}")
    print(f"Greedy bounds analysis — n={n}, {n_samples} samples")
    print(f"{'='*70}")

    BANDS = [(0.45, 0.65), (0.65, 0.85), (0.85, 1.10), (1.10, 1.50)]

    for d_lo, d_hi in BANDS:
        k_los, k_his, costs_lo, costs_hi = [], [], [], []

        for _ in range(n_samples // len(BANDS)):
            d = random.uniform(d_lo, d_hi)
            try:
                inst   = SubsetSumInstance\
                    .create_crypto_density_no_overflow_feasible(n, d)
                k_lo, k_hi = compute_greedy_bounds(inst)
                k_los.append(k_lo)
                k_his.append(k_hi)
                costs_lo.append(math.comb(n, k_lo))
                costs_hi.append(math.comb(n, k_hi))
            except Exception:
                continue

        if not k_los:
            continue

        print(f"\n  d=[{d_lo:.2f},{d_hi:.2f}]:")
        print(f"    k_lo : mean={np.mean(k_los):.1f}, "
              f"min={min(k_los)}, max={max(k_los)}")
        print(f"    k_hi : mean={np.mean(k_his):.1f}, "
              f"min={min(k_his)}, max={max(k_his)}")
        print(f"    k_window : mean={np.mean(np.array(k_his)-np.array(k_los)):.1f}")

        # Feasibility of extreme enumeration
        for limit in [10**4, 10**5, 10**6, 10**7]:
            frac_lo = (np.array(costs_lo) <= limit).mean()
            frac_hi = (np.array(costs_hi) <= limit).mean()
            frac_any = ((np.array(costs_lo) <= limit) |
                       (np.array(costs_hi) <= limit)).mean()
            print(f"    C(n,k) ≤ {limit:>8,} : "
                  f"k_lo={frac_lo:.0%}, k_hi={frac_hi:.0%}, "
                  f"either={frac_any:.0%}")


# =============================================================================
# Main
# =============================================================================

def compute_smart_window(instance, tolerance=3):
    n       = instance.n
    weights = instance.weights
    T       = instance.target
    k_lo, k_hi = compute_greedy_bounds(instance)
    mean_w = sum(weights) / n
    k_est  = max(k_lo, min(k_hi, round(T / mean_w) if mean_w > 0 else n // 2))
    k_lo_s = max(k_lo, k_est - tolerance)
    k_hi_s = min(k_hi, k_est + tolerance)
    if k_lo_s > k_hi_s:
        k_lo_s = max(k_lo, k_est - tolerance - 1)
        k_hi_s = min(k_hi, k_est + tolerance + 1)
    return k_lo_s, k_hi_s


def benchmark_smart_window(
    n: int,
    densities: list[float],
    runs: int = 30,
    timeout: float = 10.0,
    tolerances: list[int] | None = None,
) -> None:
    """
    Compare CP-SAT variants with different window strategies.

    Solvers compared:
        1. CP-SAT pure
        2. CP-SAT + greedy bounds [k_lo, k_hi]         (100% guaranteed)
        3. CP-SAT + smart window tol=2                  (~83% coverage)
        4. CP-SAT + smart window tol=3                  (~97% coverage)
        5. CP-SAT + smart window tol=5                  (~100% coverage)

    For each density prints:
        - Solve rate
        - Mean time when solved
        - k* in window rate (coverage)
        - Space reduction achieved
    """
    from solve_cpsat import solve_cpsat

    if tolerances is None:
        tolerances = [2, 3, 5]

    print(f"\n{'='*95}")
    print(f"Smart Window Benchmark — n={n}, {runs} runs, timeout={timeout}s")
    print(f"{'='*95}")

    header = f"{'Density':>8} | {'CP-SAT':>8}"
    header += f" | {'Greedy[lo,hi]':>13}"
    for tol in tolerances:
        header += f" | {'Smart tol='+str(tol):>12}"
    print(header)
    print("-" * 95)

    for d in densities:
        cpsat_rates   = []
        greedy_rates  = []
        smart_rates   = {tol: [] for tol in tolerances}
        coverage      = {tol: [] for tol in tolerances}
        reductions    = {tol: [] for tol in tolerances}

        for _ in range(runs):
            try:
                inst = SubsetSumInstance.create_crypto_density_feasible(n, d)
            except ValueError:
                continue

            k_true = sum(inst.solution) if inst.solution else None

            # 1. CP-SAT pure
            r0 = solve_cpsat(inst)
            cpsat_rates.append(int(r0.solution is not None))

            # 2. CP-SAT + greedy bounds
            rg = solve_cpsat_greedy(inst, tolerance=0, timeout=timeout)
            greedy_rates.append(int(rg.solution is not None))

            # 3. CP-SAT + smart window for each tolerance
            for tol in tolerances:
                rs = solve_cpsat_smart(inst, tolerance=tol, timeout=timeout)
                smart_rates[tol].append(int(rs.solution is not None))

                # Coverage: is k* in the smart window?
                k_lo_s, k_hi_s = compute_smart_window(inst, tol)
                in_win = int(k_true is not None and k_lo_s <= k_true <= k_hi_s)
                coverage[tol].append(in_win)

                # Space reduction
                total   = 2 ** n
                window  = search_space_window(n, k_lo_s, k_hi_s)
                reductions[tol].append(1 - window / total)

        cr  = np.mean(cpsat_rates)  * 100
        gr  = np.mean(greedy_rates) * 100

        row = f"{d:>8.2f} | {cr:>7.0f}%"
        row += f" | {gr:>12.0f}%"
        for tol in tolerances:
            sr  = np.mean(smart_rates[tol]) * 100
            row += f" | {sr:>11.0f}%"
        print(row)

        # Second row: coverage and reduction
        row2 = f"{'':>8}   {'':>8}"
        row2 += f"   {'cov=100%':>12}"
        for tol in tolerances:
            cov = np.mean(coverage[tol]) * 100
            red = np.mean(reductions[tol]) * 100
            row2 += f"   {'cov='+f'{cov:.0f}%,r='+f'{red:.0f}%':>12}"
        print(row2)

    print(f"{'='*95}")
    print("cov = k* in window rate, r = search space reduction")


import random
import numpy as np
from SubsetSumInstance import SubsetSumInstance

def analyze_bounds(
    n: int,
    n_samples: int = 2000,
    d_min: float = 0.45,
    d_max: float = 1.50,
) -> None:
    """Analyze k_lo, k_hi, costs and bound tightening across densities."""
    BANDS = [(0.45, 0.65), (0.65, 0.85), (0.85, 1.10), (1.10, 1.50)]

    print(f"\n{'='*70}")
    print(f"Bounds analysis — n={n}, {n_samples} samples")
    print(f"{'='*70}")

    for d_lo, d_hi in BANDS:
        k_los, k_his   = [], []
        costs_lo, costs_hi = [], []
        n_fixed_list   = []
        cov3_list      = []

        for _ in range(n_samples // len(BANDS)):
            d = random.uniform(d_lo, d_hi)
            try:
                inst   = SubsetSumInstance\
                    .create_crypto_density_no_overflow_feasible(n, d)
                k_lo, k_hi = compute_greedy_bounds(inst)
                k_los.append(k_lo)
                k_his.append(k_hi)
                costs_lo.append(enum_cost(n, k_lo))
                costs_hi.append(enum_cost(n, k_hi))

                # Smart window coverage
                k_lo_s, k_hi_s = compute_smart_window(inst, 3)
                k_true = sum(inst.solution) if inst.solution else None
                cov3_list.append(
                    int(k_true is not None and k_lo_s <= k_true <= k_hi_s)
                )

                # Bound tightening
                fz, fo = bound_tightening(inst, k_lo_s, k_hi_s)
                n_fixed_list.append(len(fz) + len(fo))
            except Exception:
                continue

        if not k_los:
            continue

        print(f"\n  d=[{d_lo:.2f},{d_hi:.2f}]:")
        print(f"    k_lo  : mean={np.mean(k_los):.1f}, range=[{min(k_los)},{max(k_los)}]")
        print(f"    k_hi  : mean={np.mean(k_his):.1f}, range=[{min(k_his)},{max(k_his)}]")
        print(f"    smart window tol=3 coverage : {np.mean(cov3_list):.1%}")
        print(f"    bound tightening fixed vars : {np.mean(n_fixed_list):.1f} / {n}")

        for limit in [10**5, 10**6, 10**7]:
            frac_lo  = (np.array(costs_lo) <= limit).mean()
            frac_hi  = (np.array(costs_hi) <= limit).mean()
            frac_any = ((np.array(costs_lo) <= limit) |
                        (np.array(costs_hi) <= limit)).mean()
            print(f"    enum_cost ≤ {limit:>8,} : "
                  f"k_lo={frac_lo:.0%}, k_hi={frac_hi:.0%}, "
                  f"either={frac_any:.0%}")


