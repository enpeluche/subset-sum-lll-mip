"""Data structures, serialization, and aggregation for benchmark results."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class RunRecord:
    """One run of one parameter combination (all solvers)."""

    density: float
    n: int
    run_idx: int
    results: dict  # str -> SolveResult

    def to_dict(self) -> dict:
        return {
            "density": self.density,
            "n": self.n,
            "run_idx": self.run_idx,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


@dataclass
class AggRecord:
    """One (n, density) point, aggregated over all runs."""

    n: int
    density: float
    n_runs: int
    results: dict[str, dict]  # solver_name -> stats dict

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "density": self.density,
            "n_runs": self.n_runs,
            "results": self.results,
        }


# =====================================================================
# Aggregation
# =====================================================================

def aggregate_runs(records: list[RunRecord], n_runs: int) -> list[AggRecord]:
    """Collapse individual runs into per-point statistics.

    For each (n, density) x solver, computes:
        success_rate, n_success,
        t_mean, t_median, t_std, t_min, t_max (successful runs only),
        labels distribution.
    """
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for rec in records:
        key = (rec.n, rec.density)
        for name, res in rec.results.items():
            groups[key][name].append(res)

    agg = []
    for (n, d), solvers in groups.items():
        solver_stats = {}
        for name, results in solvers.items():
            times = [r.elapsed for r in results if r.solution is not None]
            successes = sum(1 for r in results if r.solution is not None)

            solver_stats[name] = {
                "success_rate": successes / len(results),
                "n_success": successes,
                "t_mean": float(np.mean(times)) if times else None,
                "t_median": float(np.median(times)) if times else None,
                "t_std": float(np.std(times)) if times else None,
                "t_min": float(np.min(times)) if times else None,
                "t_max": float(np.max(times)) if times else None,
                "labels": dict(Counter(r.label for r in results)),
            }

        agg.append(AggRecord(n=n, density=d, n_runs=n_runs, results=solver_stats))

    return agg


# =====================================================================
# Persistence
# =====================================================================

def save_agg_records(records: list[AggRecord], path: str) -> None:
    """Save aggregated records to JSON (lightweight)."""
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def save_records(records: list[RunRecord], path: str) -> None:
    """Save raw run records to JSON (heavy -- prefer save_agg_records)."""
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def load_records(path: str) -> list[dict]:
    """Load raw JSON data from disk."""
    with open(path) as f:
        return json.load(f)


def load_agg_records(path: str) -> list[AggRecord]:
    """Load AggRecords from a saved JSON file."""
    with open(path) as f:
        raw = json.load(f)
    return [
        AggRecord(n=e["n"], density=e["density"], n_runs=e["n_runs"], results=e["results"])
        for e in raw
    ]


# =====================================================================
# DataFrame conversion
# =====================================================================

def records_to_df(records: list[RunRecord]) -> pd.DataFrame:
    """Flatten raw RunRecords into a tidy DataFrame."""
    rows = []
    for rec in records:
        for solver_name, res in rec.results.items():
            rows.append({
                "n": rec.n,
                "density": rec.density,
                "solver": solver_name,
                "success": 1 if res.solution is not None else 0,
                "time": res.elapsed,
                "best_res": res.best_res,
            })
    return pd.DataFrame(rows)


def agg_records_to_df(agg_records: list[AggRecord]) -> pd.DataFrame:
    """Flatten AggRecords into a tidy DataFrame (success in %)."""
    rows = []
    for rec in agg_records:
        for solver_name, stats in rec.results.items():
            rows.append({
                "n": rec.n,
                "density": rec.density,
                "solver": solver_name,
                "success": stats["success_rate"] * 100,
                "time": stats["t_mean"],
            })
    return pd.DataFrame(rows)