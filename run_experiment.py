"""
run_experiment.py
-----------------
Reproduction script for the Subset Sum hybrid solver benchmark.

Usage:
    python run_experiment.py                  # default experiment (n=30)
    python run_experiment.py --n 50           # larger dimension
    python run_experiment.py --runs 10        # more runs per density
    python run_experiment.py --out results/   # custom output directory

Results are saved as:
    <out>/records_n<n>.json   ← raw RunRecords (reloadable)
    <out>/stats_n<n>.json     ← aggregated statistics
    <out>/performance_n<n>.png
    <out>/geometry_n<n>.png
"""

import argparse
import json
import os

from benchmark import run_benchmark
from stats import compute_stats
from plotting import plot_lll_geometry, plot_performance

from solve_cpsat import solve_cpsat
from solve_lll_hybrid import solve_lll_hybrid
from solve_bkz_hybrid import solve_bkz_hybrid

from SubsetSumInstance import SubsetSumInstance

from results import save_records


# =============================================================================
# Experiment configuration
# =============================================================================

from solve_adaptative_hybrid import solve_adaptive_hybrid
from solve_mitm import solve_mitm_classic
from solve_mitm_hgj import solve_mitm_hgj
from solve_ultimate import solve_ultimate
from functools import partial
from solve_tabu import solve_tabu

SOLVERS = {
    "cpsat": solve_cpsat,
    "ultimate": partial(solve_ultimate, mitm_max_subsets=2 ** 21),
}


# =============================================================================
# Entry point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Subset Sum hybrid solver benchmark")
    parser.add_argument("--n", type=int, default=30, help="Instance dimension")
    parser.add_argument("--runs", type=int, default=5, help="Runs per density")
    parser.add_argument("--out", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    json_path = os.path.join(args.out, f"records_n{args.n}.json")
    stats_path = os.path.join(args.out, f"stats_n{args.n}.json")
    perf_path = os.path.join(args.out, f"performance_n{args.n}.png")
    geom_path = os.path.join(args.out, f"geometry_n{args.n}.png")

    DENSITIES: list[float] = [
        round(SubsetSumInstance.get_min_safe_density(args.n) + 0.01 + 0.025 * i, 3)
        for i in range(80)
    ]

    print(f"Experiment: n={args.n}, runs={args.runs}, densities={DENSITIES}")
    print(f"Solvers: {list(SOLVERS.keys())}")
    print(f"Output: {args.out}/\n")

    # --- Run benchmark ---
    records = run_benchmark(
        n=args.n,
        densities=DENSITIES,
        n_runs=args.runs,
        solvers=SOLVERS,
        json_file=json_path,
    )

    # --- Compute stats ---
    stats = compute_stats(records, DENSITIES)

    with open(stats_path, "w") as f:
        # Convert float keys to str for JSON serialization
        json.dump({str(k): v for k, v in stats.items()}, f, indent=2)

    print(f"\nRecords saved → {json_path}")
    print(f"Stats saved   → {stats_path}")

    # --- Plots ---
    plot_performance(
        stats,
        DENSITIES,
        n=args.n,
        save_path=perf_path,
    )
    # plot_lll_geometry(records, DENSITIES, solver="adaptive", save_path=geom_path)

    print(f"Performance plot → {perf_path}")
    # print(f"Geometry plot    → {geom_path}")


if __name__ == "__main__":
    main()
