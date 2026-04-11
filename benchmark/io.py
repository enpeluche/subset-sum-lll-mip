"""Data structures and serialization for benchmark results."""

import json
from dataclasses import dataclass

import pandas as pd


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


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_records(records: list[RunRecord], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def records_to_df(records: list[RunRecord]) -> pd.DataFrame:
    """Flatten a list of RunRecord into a tidy DataFrame."""
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