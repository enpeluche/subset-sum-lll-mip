from dataclasses import dataclass, asdict
import json


@dataclass
class SolveResult:
    elapsed: float
    branches: int
    conflicts: int
    status: int
    solution: list[int] | None
    label: str
    best_res: int | None
    best_ham: int | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunRecord:
    density: float
    run_idx: int
    n: int
    results: dict[str, SolveResult]

    def to_dict(self) -> dict:
        return {
            "density": self.density,
            "run_idx": self.run_idx,
            "n": self.n,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }


def save_records(records: list[RunRecord], path: str) -> None:
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def load_records(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)
