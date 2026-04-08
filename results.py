from dataclasses import dataclass


@dataclass
class SolveResult:
    elapsed: float
    branches: int
    conflicts: int
    status: int
    solution: list[int] | None
    label: str
    best_res: int
    best_ham: int | None
