"""
benchmark/experiment.py
-----------------------
Benchmark class: encapsulates config, execution, persistence, and visualization.

Usage — run new experiment:
    exp = Benchmark(suite="delta", n="2:50:2", densities="0.1:5:0.1",
                    runs=10, timeout=10, gen="crypto_big", out="lll_study")
    exp.run()
    exp.plot()

Usage — reload from JSON:
    exp = Benchmark.load("results/lll_study/bench.json")
    exp.plot()
    df = exp.to_dataframe()
"""

import os
import time
import json
from dataclasses import dataclass, field

from .cli import parse_range
from .io import (
    RunRecord, AggRecord,
    aggregate_runs, save_agg_records, agg_records_to_df,
)
from .runner import run_benchmark
from .visual import (
    plot_success_rates,
    generate_success_mosaic,
    generate_time_mosaic,
    plot_all_success_rate_heatmaps_mosaic,
    generate_speedup_mosaic,
)
from .suites import build_solvers


# =====================================================================
# Instance generators registry
# =====================================================================

def _get_generators():
    """Lazy import to avoid circular dependencies."""
    from SubsetSumInstance import SubsetSumInstance
    return {
        "uniform":    SubsetSumInstance.create_uniform_feasible,
        "super_inc":  SubsetSumInstance.create_super_increasing_feasible,
        "crypto":     SubsetSumInstance.create_crypto_density_feasible,
        "crypto_big": SubsetSumInstance.create_crypto_density_no_overflow_feasible,
    }


# =====================================================================
# Benchmark class
# =====================================================================

class Benchmark:
    """
    Self-contained benchmark experiment.

    Holds configuration, aggregated results, and plotting methods.
    Can be created fresh (then .run()), or loaded from JSON (.load()).
    """

    def __init__(
        self,
        suite: str = "arch",
        n: str = "2:50:2",
        densities: str = "0.1:5:0.1",
        deltas: str = "0.99",
        blocks: str = "20",
        runs: int = 10,
        timeout: float = 10.0,
        gen: str = "crypto_big",
        out: str = "default",
        workers: int | None = None,
    ):
        # --- Config ---
        self.suite = suite
        self.runs = runs
        self.timeout = timeout
        self.gen = gen
        self.base_dir = os.path.join("results", out)
        self.workers = workers or max(1, (os.cpu_count() or 6) // 2)

        # --- Parse ranges ---
        self.all_ranges = {
            "n":          parse_range(n, int),
            "density":    parse_range(densities, float),
            "delta":      parse_range(deltas, float),
            "block_size": parse_range(blocks, int),
        }

        # Instance axes (what varies per row in the benchmark table)
        self.instance_ranges = {
            "n":       self.all_ranges["n"],
            "density": self.all_ranges["density"],
        }

        # --- Build solvers ---
        self.solvers = build_solvers(self.all_ranges, suite, timeout)
        self.solver_names = list(self.solvers.keys())
        self.ref_solver = self.solver_names[0]

        # --- Results (populated by run() or load()) ---
        self.agg_records: list[AggRecord] = []
        self.json_path = os.path.join(self.base_dir, "bench.json")
        self._ran = False

    # -----------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------

    def run(self) -> "Benchmark":
        """Execute the benchmark and store aggregated results."""
        os.makedirs(self.base_dir, exist_ok=True)

        generators = _get_generators()
        generator = generators[self.gen]

        self.agg_records = run_benchmark(
            runs=self.runs,
            solvers=self.solvers,
            json_file=self.json_path,
            timeout=self.timeout,
            generator=generator,
            ranges=self.instance_ranges,
            workers=self.workers,
        )

        self._ran = True
        return self

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        """Save aggregated results to JSON."""
        path = path or self.json_path
        save_agg_records(self.agg_records, path)
        print(f"Saved {len(self.agg_records)} agg records to {path}")

    @classmethod
    def load(cls, json_path: str) -> "Benchmark":
        """
        Load a Benchmark from a saved JSON file.

        Creates a minimal Benchmark instance with agg_records populated.
        Solver names and ref_solver are inferred from the data.
        """
        with open(json_path) as f:
            raw = json.load(f)

        # Reconstruct AggRecords
        agg_records = []
        for entry in raw:
            agg_records.append(AggRecord(
                n=entry["n"],
                density=entry["density"],
                n_runs=entry["n_runs"],
                results=entry["results"],
            ))

        # Infer metadata from data
        solver_names = list(agg_records[0].results.keys()) if agg_records else []
        ns = sorted(set(r.n for r in agg_records))
        ds = sorted(set(r.density for r in agg_records))

        # Create a shell Benchmark
        exp = cls.__new__(cls)
        exp.agg_records = agg_records
        exp.solver_names = solver_names
        exp.ref_solver = solver_names[0] if solver_names else ""
        exp.json_path = json_path
        exp.base_dir = os.path.dirname(json_path)
        exp.suite = "loaded"
        exp.runs = agg_records[0].n_runs if agg_records else 0
        exp.timeout = 0
        exp.gen = "unknown"
        exp.workers = 0
        exp.all_ranges = {
            "n": ns,
            "density": ds,
            "delta": [],
            "block_size": [],
        }
        exp.instance_ranges = {"n": ns, "density": ds}
        exp.solvers = {}
        exp._ran = True

        print(f"Loaded {len(agg_records)} points, "
              f"solvers: {solver_names}, "
              f"n: {ns[0]}-{ns[-1]}, "
              f"d: {ds[0]}-{ds[-1]}")

        return exp

    # -----------------------------------------------------------------
    # Data access
    # -----------------------------------------------------------------

    def to_dataframe(self):
        """Return a tidy DataFrame of aggregated results."""
        return agg_records_to_df(self.agg_records)

    @property
    def is_2d(self) -> bool:
        """True if both n and density have multiple values (heatmap-worthy)."""
        return (len(self.all_ranges["density"]) > 1 and
                len(self.all_ranges["n"]) > 1)

    def summary(self) -> str:
        """Print a quick summary of the experiment."""
        n_r = self.all_ranges["n"]
        d_r = self.all_ranges["density"]
        lines = [
            f"Suite:    {self.suite}",
            f"Solvers:  {', '.join(self.solver_names)}",
            f"Ref:      {self.ref_solver}",
            f"n:        {n_r[0]}..{n_r[-1]} ({len(n_r)} values)",
            f"density:  {d_r[0]}..{d_r[-1]} ({len(d_r)} values)",
            f"Runs:     {self.runs}",
            f"Points:   {len(self.agg_records)}",
            f"Output:   {self.base_dir}",
        ]
        return "\n".join(lines)

    # -----------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------

    def plot(self, ref_solver: str | None = None) -> None:
        """Generate all plots appropriate for this experiment."""
        if not self.agg_records:
            print("[Warning] No data to plot. Run .run() or .load() first.")
            return

        ref = ref_solver or self.ref_solver

        if self.is_2d:
            self.plot_success(ref)
            self.plot_time()
            self.plot_speedups(ref)
        else:
            self.plot_curves()

        print(f"\nAll plots saved to {self.base_dir}/")

    def plot_success(self, ref_solver: str | None = None) -> None:
        """Generate success rate heatmaps + diff mosaic."""
        ref = ref_solver or self.ref_solver
        generate_success_mosaic(self.agg_records, self.base_dir)
        plot_all_success_rate_heatmaps_mosaic(self.agg_records, ref, self.base_dir)

    def plot_time(self) -> None:
        """Generate time heatmaps."""
        generate_time_mosaic(self.agg_records, self.base_dir)

    def plot_speedups(self, ref_solver: str | None = None) -> None:
        """Generate speedup heatmaps."""
        ref = ref_solver or self.ref_solver
        generate_speedup_mosaic(self.agg_records, ref, self.base_dir)

    def plot_curves(self) -> None:
        """Generate 1D line plots (when only one axis varies)."""
        plot_success_rates(self.agg_records, os.path.join(self.base_dir, "curves.png"))

    # -----------------------------------------------------------------
    # Repr
    # -----------------------------------------------------------------

    def __repr__(self) -> str:
        status = "ran" if self._ran else "pending"
        return (f"Benchmark(suite={self.suite!r}, "
                f"solvers={len(self.solver_names)}, "
                f"points={len(self.agg_records)}, "
                f"status={status})")