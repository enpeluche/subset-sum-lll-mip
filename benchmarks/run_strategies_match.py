#!/usr/bin/env python3
"""
run_strategies_match.py — Subset Sum Lattice Benchmark

Make sure to always use same --nn and --densities parameters --timeout

python -m benchmarks.run_strategies_match --n 2:50:1 --runs 50  --timeout 5 --densities 0.1:5:0.05 --blocks 10,20,30 --suite block --gen crypto_big --out bkz_study
python -m benchmarks.run_strategies_match --n 2:50:1 --runs 50  --timeout 5 --densities 0.1:5:0.05 --deltas 0.5,0.75,0.99 --suite delta --gen crypto_big --out lll_study_0
python -m benchmarks.run_strategies_match --n 2:50:1 --runs 50  --timeout 5 --densities 0.1:5:0.05 --deltas 0.99,0.995,0.999 --suite delta --gen crypto_big --out lll_study_1
python -m benchmarks.run_strategies_match --n 2:50:1 --runs 50 --timeout 5 --densities 0.1:5:0.05 --deltas 0.99 --blocks 20 --suite hybrid_comp --gen crypto_big --out hybrid_lll_bkz_inter_seq_study
python -m benchmarks.run_strategies_match --n 2:50:2 --runs 20 --timeout 5 --densities 0.1:5:0.1 --deltas 0.99 --blocks 20 --suite arch --gen crypto_big --out architecture_comparison


# --- CP-SAT : comparaison des 6 variantes ---
python -m benchmarks.run_strategies_match \
  --n 2:50:2 --densities 0.1:5:0.1 --runs 10 --timeout 10 \
  --suite cpsat_comp --gen crypto_big --out cpsat_comparison

# --- Lattice : tes 5 runs existants (ajustés au même pas) ---
python -m benchmarks.run_strategies_match \
  --n 2:50:2 --densities 0.1:5:0.1 --runs 10 --timeout 10 \
  --blocks 10,20,30 --suite block --gen crypto_big --out bkz_study

python -m benchmarks.run_strategies_match \
  --n 2:50:2 --densities 0.1:5:0.1 --runs 10 --timeout 10 \
  --deltas 0.5,0.75,0.99 --suite delta --gen crypto_big --out lll_study_0

python -m benchmarks.run_strategies_match \
  --n 2:50:2 --densities 0.1:5:0.1 --runs 10 --timeout 10 \
  --deltas 0.99,0.995,0.999 --suite delta --gen crypto_big --out lll_study_1

python -m benchmarks.run_strategies_match \
  --n 2:50:2 --densities 0.1:5:0.1 --runs 10 --timeout 10 \
  --deltas 0.99 --blocks 20 --suite hybrid_comp --gen crypto_big --out hybrid_seq_vs_indep

python -m benchmarks.run_strategies_match \
  --n 2:50:2 --densities 0.1:5:0.1 --runs 10 --timeout 10 \
  --deltas 0.99 --blocks 20 --suite arch --gen crypto_big --out architecture_comparison
"""

import os
import time
import matplotlib
matplotlib.use("Agg")

from benchmark.cli import build_parser, parse_all_ranges
from benchmark.suites import build_solvers
from benchmark.runner import run_benchmark
from benchmark.visual import (
    plot_all_heatmaps,
    plot_all_time_heatmaps,
    plot_success_rates,
    generate_all_diffs_for_base,
    generate_all_speedups_for_base,
)
from SubsetSumInstance import SubsetSumInstance

GENERATORS = {
    "uniform":    SubsetSumInstance.create_uniform_feasible,
    "super_inc":  SubsetSumInstance.create_super_increasing_feasible,
    "crypto":     SubsetSumInstance.create_crypto_density_feasible,
    "crypto_big": SubsetSumInstance.create_crypto_density_no_overflow_feasible,
}


def main():
    args = build_parser().parse_args()

    # --- Output: always under results/<out>/ ---
    base_dir = os.path.join("results", args.out)
    os.makedirs(base_dir, exist_ok=True)

    # --- Ranges & solvers ---
    all_ranges = parse_all_ranges(args)
    solvers = build_solvers(all_ranges, args.suite, args.timeout)

    instance_ranges = {
        "n":       all_ranges["n"],
        "density": all_ranges["density"],
    }

    json_path = os.path.join(base_dir, f"bench_{args.gen}_{time.strftime('%Y%m%d_%H%M%S')}.json")

    # --- Run ---
    records = run_benchmark(
        runs=args.runs,
        solvers=solvers,
        json_file=json_path,
        timeout=args.timeout,
        generator=GENERATORS[args.gen],
        ranges=instance_ranges,
    )

    # --- Plot ---
    ref_solver = list(solvers)[0]

    if len(all_ranges["density"]) > 1 and len(all_ranges["n"]) > 1:
        for sub in ["success", "time", "diffs", "speedups"]:
            os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

        plot_all_heatmaps(records, os.path.join(base_dir, "success"))
        plot_all_time_heatmaps(records, os.path.join(base_dir, "time"))
        generate_all_diffs_for_base(records, ref_solver, os.path.join(base_dir, "diffs"))
        generate_all_speedups_for_base(records, ref_solver, os.path.join(base_dir, "speedups"))
    else:
        plot_success_rates(records, os.path.join(base_dir, "curves.png"))


if __name__ == "__main__":
    main()