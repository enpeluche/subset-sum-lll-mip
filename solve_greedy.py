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


def benchmark_greedy(
    n: int,
    densities: list[float],
    runs: int = 30,
    timeout: float = 10.0,
    max_subsets_extreme: int = 2_000_000,
) -> None:
    """
    Compare 4 solvers across densities:
        1. CP-SAT pure baseline
        2. CP-SAT + greedy window [k_lo, k_hi]
        3. Greedy extreme enumeration
        4. Greedy extreme → CP-SAT fallback

    For each density, also prints:
        - k_lo, k_hi, k_mid estimates
        - C(n, k_lo) and C(n, k_hi) — cost of extreme enumeration
        - Window size reduction vs 2^n
    """
    from solve_cpsat import solve_cpsat

    print(f"\n{'='*90}")
    print(f"Greedy Benchmark — n={n}, {runs} runs/density, timeout={timeout}s")
    print(f"{'='*90}")

    # Header
    print(f"{'Density':>8} | "
          f"{'k_lo':>5} {'k_hi':>5} {'C(n,k_lo)':>12} {'C(n,k_hi)':>12} | "
          f"{'CP-SAT':>8} {'CP+Greedy':>10} {'Extreme':>8} {'Extreme+CP':>11}")
    print("-" * 90)

    for d in densities:
        cpsat_rates    = []
        greedy_rates   = []
        extreme_rates  = []
        extcp_rates    = []

        cpsat_times    = []
        greedy_times   = []
        extreme_times  = []

        k_lo_vals, k_hi_vals = [], []

        for _ in range(runs):
            try:
                inst = SubsetSumInstance.create_crypto_density_feasible(n, d)
            except ValueError:
                continue

            k_lo, k_hi = compute_greedy_bounds(inst)
            k_lo_vals.append(k_lo)
            k_hi_vals.append(k_hi)

            # 1. CP-SAT pure
            r0 = solve_cpsat(inst)
            cpsat_rates.append(int(r0.solution is not None))
            if r0.solution is not None:
                cpsat_times.append(r0.elapsed)

            # 2. CP-SAT + greedy window
            rg = solve_cpsat_greedy(inst, tolerance=0, timeout=timeout)
            greedy_rates.append(int(rg.solution is not None))
            if rg.solution is not None:
                greedy_times.append(rg.elapsed)

            # 3. Greedy extreme
            re = solve_greedy_extreme(
                inst, max_subsets=max_subsets_extreme
            )
            extreme_rates.append(int(re.solution is not None))
            if re.solution is not None:
                extreme_times.append(re.elapsed)

            # 4. Greedy extreme + CP-SAT fallback
            if re.solution is not None:
                extcp_rates.append(1)
            else:
                # Fallback to CP-SAT with greedy window
                rfall = solve_cpsat_greedy(inst, tolerance=0, timeout=timeout)
                extcp_rates.append(int(rfall.solution is not None))

        # Stats
        k_lo_med = int(np.median(k_lo_vals)) if k_lo_vals else 0
        k_hi_med = int(np.median(k_hi_vals)) if k_hi_vals else n
        cost_lo  = math.comb(n, k_lo_med)
        cost_hi  = math.comb(n, k_hi_med)

        cr  = np.mean(cpsat_rates)   * 100
        gr  = np.mean(greedy_rates)  * 100
        er  = np.mean(extreme_rates) * 100
        ecr = np.mean(extcp_rates)   * 100

        cost_lo_str = f"{cost_lo:>12,}" if cost_lo < 10**9 else f"{'2^'+str(int(math.log2(cost_lo))):>12}"
        cost_hi_str = f"{cost_hi:>12,}" if cost_hi < 10**9 else f"{'2^'+str(int(math.log2(cost_hi))):>12}"

        print(f"{d:>8.2f} | "
              f"{k_lo_med:>5} {k_hi_med:>5} {cost_lo_str} {cost_hi_str} | "
              f"{cr:>7.0f}% {gr:>9.0f}% {er:>7.0f}% {ecr:>10.0f}%")

    print(f"{'='*90}")
    print("C(n,k_lo) = subsets to check near k_lo (small = cheap)")
    print("Extreme   = greedy extreme enumeration only")
    print("Extreme+CP= extreme first, CP-SAT fallback if failed")


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",       type=int,   default=30)
    parser.add_argument("--runs",    type=int,   default=30)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--mode",    type=str,   default="smart",
                        choices=["smart", "analyze", "full"])
    args = parser.parse_args()

    DENSITIES = [0.54, 0.65, 0.75, 0.85, 0.94, 1.00, 1.25]

    if args.mode == "analyze":
        analyze_k_distribution(args.n, n_samples=2000)

    elif args.mode == "smart":
        benchmark_smart_window(
            n=args.n,
            densities=DENSITIES,
            runs=args.runs,
            timeout=args.timeout,
            tolerances=[2, 3, 5],
        )

    elif args.mode == "full":
        analyze_k_distribution(args.n, n_samples=1000)
        benchmark_greedy(
            n=args.n, densities=DENSITIES,
            runs=args.runs, timeout=args.timeout,
        )
        benchmark_smart_window(
            n=args.n, densities=DENSITIES,
            runs=args.runs, timeout=args.timeout,
        )