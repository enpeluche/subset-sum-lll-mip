import numpy as np
from results import RunRecord


def compute_stats(records: list[RunRecord], densities: list[float]) -> dict:
    """
    Aggregate benchmark results by density and solver.

    Args:
        records:   List of RunRecord from run_benchmark.
        densities: Ordered list of density values to aggregate.

    Returns:
        Nested dict: stats[density][solver_name] → {
            t_mean, t_std,
            speedup_mean, speedup_std,  ← vs first solver in records
            succ_pct,
            b_mean, c_mean,
        }
    """
    stats = {}

    for d in densities:
        runs = [r for r in records if r.density == d]
        if not runs:
            continue

        solver_names = list(runs[0].results.keys())
        stats[d] = {}

        # Référence pour le speedup = premier solveur
        ref_name = solver_names[0]
        t_ref = [r.results[ref_name].elapsed for r in runs]

        for name in solver_names:
            elapsed = [r.results[name].elapsed for r in runs]
            branches = [r.results[name].branches for r in runs]
            conflicts = [r.results[name].conflicts for r in runs]
            solved = [r.results[name].solution is not None for r in runs]
            speedups = [tb / max(th, 1e-5) for tb, th in zip(t_ref, elapsed)]

            stats[d][name] = {
                "t_mean": np.mean(elapsed),
                "t_std": np.std(elapsed),
                "speedup_mean": np.mean(speedups),
                "speedup_std": np.std(speedups),
                "succ_pct": np.mean(solved) * 100,
                "b_mean": np.mean(branches),
                "c_mean": np.mean(conflicts),
            }

    return stats
