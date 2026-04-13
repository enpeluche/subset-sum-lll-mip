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
        d = asdict(self)
        d["status"] = int(d["status"])  # force int pour JSON
        return d

    @classmethod
    def trivially_infeasible(cls, elapsed_time: float) -> "SolveResult":
        """Factory method to generate a standard infeasible result."""
        return cls(
            elapsed=elapsed_time,
            branches=0,
            conflicts=0,
            status=2,
            solution=None,
            label="Trivially_Infeasible",
            best_res=None,
            best_ham=None,
        )

    @classmethod
    def timeout(
        cls, 
        elapsed: float, 
        label: str, 
        res: int | None = None, 
        ham: int | None = None
    ) -> "SolveResult":
        """Generic factory for any timeout scenario."""
        return cls(
            elapsed=elapsed,
            branches=0,
            conflicts=0,
            status=0,  # 0 = UNKNOWN / TIMEOUT
            solution=None,
            label=label,
            best_res=res,
            best_ham=ham,
        )
    
    @classmethod
    def skipped(cls, label: str) -> "SolveResult":
        """Factory for skipped solvers (e.g., MITM when N is too large)."""
        return cls(
            elapsed=0.0,
            branches=0,
            conflicts=0,
            status=3, # Souvent utilisé pour 'SKIPPED'
            solution=None,
            label=label,
            best_res=None,
            best_ham=None,
        )
    

