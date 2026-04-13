#!/usr/bin/env python3
"""
run_strategies_match.py — Subset Sum Lattice Benchmark

Usage:
    python -m benchmark.run_strategies_match --suite delta --out lll_study
    python -m benchmark.run_strategies_match --suite cpsat_formulation --runs 30

Reload and replot:
    from benchmark import Benchmark
    exp = Benchmark.load("results/lll_study/bench.json")
    exp.plot()
    df = exp.to_dataframe()
"""

import matplotlib
matplotlib.use("Agg")

from benchmark.cli import build_parser
from benchmark.experiment import Benchmark


def main():
    args = build_parser().parse_args()

    exp = Benchmark(
        suite=args.suite,
        n=args.n,
        densities=args.densities,
        deltas=args.deltas,
        blocks=args.blocks,
        runs=args.runs,
        timeout=args.timeout,
        gen=args.gen,
        out=args.out,
    )

    print(exp.summary())
    exp.run()
    exp.plot()


if __name__ == "__main__":
    main()