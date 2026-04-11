
import random
import numpy as np
from SubsetSumInstance import SubsetSumInstance

def benchmark_all(
    n: int,
    densities: list[float],
    runs: int = 30,
    timeout: float = 10.0,
    max_subsets: int = 2_000_000,
) -> None:
    """
    Compare all greedy strategies vs CP-SAT baseline.

    Columns:
        CP-SAT    : baseline, no weight constraint
        CP+Greedy : CP-SAT + greedy bounds [k_lo,k_hi], tol=0
        CP+Smart  : CP-SAT + smart window tol=3 (~97% coverage)
        CP+SmartBT: CP-SAT + smart window + bound tightening
        Extreme   : greedy extreme symmetric enumeration
        Full      : extreme → CP-SAT+SmartBT fallback
    """
    from solve_cpsat import solve_cpsat

    print(f"\n{'='*100}")
    print(f"Greedy v2 Benchmark — n={n}, {runs} runs/density, timeout={timeout}s")
    print(f"{'='*100}")
    print(
        f"{'Density':>8} | {'k_lo':>5} {'k_hi':>5} {'cost_lo':>10} {'cost_hi':>10} | "
        f"{'Base':>6} {'Greedy':>7} {'Smart':>6} {'SmartBT':>8} "
        f"{'Extreme':>8} {'Full':>6}"
    )
    print("-" * 100)

    for d in densities:
        rates    = {k: [] for k in
                    ["base","greedy","smart","smartbt","extreme","full"]}
        times    = {k: [] for k in rates}
        branches = {k: [] for k in rates}
        k_lo_v, k_hi_v = [], []
        cov_smart = []
        n_fixed_v = []

        for _ in range(runs):
            try:
                inst = SubsetSumInstance.create_crypto_density_feasible(n, d)
            except ValueError:
                continue

            k_lo, k_hi = compute_greedy_bounds(inst)
            k_lo_v.append(k_lo)
            k_hi_v.append(k_hi)

            # Coverage of smart window
            k_lo_s, k_hi_s = compute_smart_window(inst, 3)
            k_true = sum(inst.solution) if inst.solution else None
            cov_smart.append(int(k_true is not None and k_lo_s <= k_true <= k_hi_s))

            # Bound tightening stats
            fz, fo = bound_tightening(inst, k_lo_s, k_hi_s)
            n_fixed_v.append(len(fz) + len(fo))

            def record(key, r):
                solved = r.solution is not None
                rates[key].append(int(solved))
                if solved:
                    times[key].append(r.elapsed)
                branches[key].append(r.branches)

            record("base",    solve_cpsat(inst))
            record("greedy",  solve_cpsat_greedy(inst, tolerance=0, timeout=timeout))
            record("smart",   solve_cpsat_smart(inst, tolerance=3, timeout=timeout))
            record("smartbt", solve_cpsat_smart_bt(inst, tolerance=3, timeout=timeout))
            record("extreme", solve_greedy_extreme(inst, max_subsets, timeout=timeout))
            record("full",    solve_full_greedy(inst, max_subsets, 3, timeout))

        k_lo_med  = int(np.median(k_lo_v)) if k_lo_v else 0
        k_hi_med  = int(np.median(k_hi_v)) if k_hi_v else n
        cost_lo   = enum_cost(n, k_lo_med)
        cost_hi   = enum_cost(n, k_hi_med)
        cov       = np.mean(cov_smart) * 100
        n_fix     = np.mean(n_fixed_v)

        def pct(key): return f"{np.mean(rates[key])*100:>5.0f}%"

        print(
            f"{d:>8.2f} | {k_lo_med:>5} {k_hi_med:>5} "
            f"{cost_lo:>10,} {cost_hi:>10,} | "
            f"{pct('base')} {pct('greedy')} {pct('smart')} "
            f"{pct('smartbt')} {pct('extreme')} {pct('full')}"
        )
        print(
            f"{'':>8}   {'':>5} {'':>5} {'':>10} {'':>10}   "
            f"{'':>6} {'':>7} "
            f"cov={cov:.0f}% bt_fixed={n_fix:.1f}"
        )

    print(f"{'='*100}")
    print("cov    = k* in smart window rate (should be ~97% for tol=3)")
    print("bt_fixed = mean variables fixed by bound tightening")
    print("Extreme uses symmetric enumeration: min(C(n,k), C(n,n-k))")


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


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Greedy v2 Subset Sum solver benchmark"
    )
    parser.add_argument("--n",       type=int,   default=30)
    parser.add_argument("--runs",    type=int,   default=30)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--mode",    type=str,   default="benchmark",
                        choices=["benchmark", "analyze", "both"])
    parser.add_argument("--max_subsets", type=int, default=2_000_000)
    args = parser.parse_args()

    DENSITIES = [0.54, 0.65, 0.75, 0.85, 0.94, 1.00, 1.10, 1.25]

    if args.mode in ("analyze", "both"):
        analyze_bounds(args.n, n_samples=2000)

    if args.mode in ("benchmark", "both"):
        benchmark_all(
            n=args.n,
            densities=DENSITIES,
            runs=args.runs,
            timeout=args.timeout,
            max_subsets=args.max_subsets,
        )