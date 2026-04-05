"""
Hint quality analysis for the LLL+CP-SAT hybrid solver.

Covers:
    - Solution enumeration via CP-SAT
    - Hamming quality measurement against all solutions
    - Projected consensus analysis
    - Direct candidate analysis
"""

from collections import defaultdict
from ortools.sat.python import cp_model

from util import get_instance, knapsack_matrix, extract_coefficient_submatrix
from util import filter_binary_candidates
from lll_analysis import project_to_binary, min_hamming_over_solutions


# =============================================================================
# Solution enumeration
# =============================================================================


def count_solutions(
    weights: list[int],
    T: int,
    max_solutions: int = 1000,
    timeout: float = 10.0,
) -> list[list[int]]:
    """
    Enumerate all feasible solutions to a Subset Sum instance using CP-SAT.

    Args:
        weights:       Instance weights.
        T:             Target sum.
        max_solutions: Stop after finding this many solutions.
        timeout:       Maximum solve time in seconds.

    Returns:
        List of all {0,1} solution vectors found before timeout or max_solutions.
    """
    from fpylll import LLL

    n = len(weights)
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    model.add(sum(weights[i] * x[i] for i in range(n)) == T)

    solutions = []

    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            solutions.append([self.value(x[i]) for i in range(n)])
            if len(solutions) >= max_solutions:
                self.stop_search()

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 1
    solver.solve(model, SolutionCollector())

    return solutions


def min_hamming_over_solutions(
    hint: list[int],
    solutions: list[list[int]],
) -> int | None:
    """
    Minimum Hamming distance from a hint to any known solution.

    Args:
        hint:      Binary or integer vector to evaluate.
        solutions: List of valid solution vectors.

    Returns:
        Minimum Hamming distance, or None if solutions is empty.
    """
    if not solutions:
        return None
    return min(
        sum(1 for i in range(len(hint)) if hint[i] != sol[i]) for sol in solutions
    )


# =============================================================================
# Hint quality benchmark
# =============================================================================


def run_benchmark_with_taux(
    n: int,
    density: float,
    n_runs: int = 50,
    thresholds: list[int] = [3, 5, 7],
) -> None:
    """
    Measure projected hint quality against all enumerated solutions.

    For each instance:
        - Enumerates up to 100 solutions.
        - Computes Hamming distance of each LLL vector and its projection
          against both the reference solution and the nearest known solution.
        - Tracks the best hint per instance and reports threshold rates.

    Args:
        n:          Instance dimension.
        density:    Target density.
        n_runs:     Number of instances to benchmark.
        thresholds: Hamming thresholds for rate reporting.
    """
    from fpylll import LLL

    ham_ref_orig_all, ham_ref_proj_all = [], []
    ham_min_orig_all, ham_min_proj_all = [], []
    best_hint_per_instance = []
    skipped = 0

    for _ in range(n_runs):
        weights, T, solution = get_instance(n, density)
        all_solutions = count_solutions(weights, T, max_solutions=100, timeout=10.0)

        if not all_solutions:
            skipped += 1
            continue

        B = knapsack_matrix(weights, T)
        LLL.reduction(B)
        sub = extract_coefficient_submatrix(B)

        best_this = n
        for v in sub:
            p = project_to_binary(v)

            ham_ref_orig = sum(1 for i in range(n) if v[i] != solution[i])
            ham_ref_proj = sum(1 for i in range(n) if p[i] != solution[i])
            ham_min_orig = min_hamming_over_solutions(v, all_solutions)
            ham_min_proj = min_hamming_over_solutions(p, all_solutions)

            if ham_min_orig is None or ham_min_proj is None:
                continue

            ham_ref_orig_all.append(ham_ref_orig)
            ham_ref_proj_all.append(ham_ref_proj)
            ham_min_orig_all.append(ham_min_orig)
            ham_min_proj_all.append(ham_min_proj)

            if ham_min_proj < best_this:
                best_this = ham_min_proj

        best_hint_per_instance.append(best_this)

    total = len(best_hint_per_instance)
    if total == 0:
        print("No valid instances found.")
        return

    def avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    print(f"\n{'='*60}")
    print(f"n={n}, density={density}, {n_runs} runs ({skipped} skipped)")
    print(f"{'='*60}")
    print(f"  {'':30} | {'original':>10} | {'projeté':>10}")
    print(f"  {'-'*55}")
    print(
        f"  {'Ham. moy. vs référence':30} | {avg(ham_ref_orig_all):>10.2f} | {avg(ham_ref_proj_all):>10.2f}"
    )
    print(
        f"  {'Ham. moy. min (toutes sols)':30} | {avg(ham_min_orig_all):>10.2f} | {avg(ham_min_proj_all):>10.2f}"
    )
    biais_orig = avg(ham_ref_orig_all) - avg(ham_min_orig_all)
    biais_proj = avg(ham_ref_proj_all) - avg(ham_min_proj_all)
    print(f"  {'Biais (ref - min)':30} | {biais_orig:>10.2f} | {biais_proj:>10.2f}")
    print(f"  {'n/2':30} | {n/2:>10} | {n/2:>10}")

    print(f"\n  --- Meilleur hint projeté par instance ---")
    print(f"  Hamming min moyen   : {avg(best_hint_per_instance):.2f}")
    print(f"  Hamming min médiane : {sorted(best_hint_per_instance)[total//2]}")

    print(f"\n  --- Taux d'instances avec au moins un hint proche ---")
    for t in thresholds:
        count = sum(1 for h in best_hint_per_instance if h <= t)
        print(f"  hamming_min <= {t:2d} : {count/total*100:6.1f}%  ({count}/{total})")


# =============================================================================
# Projected consensus analysis
# =============================================================================


def analyze_projected_consensus(
    n: int,
    density: float,
    n_runs: int = 100,
) -> None:
    """
    Measure reliability of unanimous projected positions across LLL vectors.

    A position is unanimous if all projected vectors agree (all 0 or all 1).
    Reports what fraction of unanimous positions are correct.

    Args:
        n:       Instance dimension.
        density: Target density.
        n_runs:  Number of instances.
    """
    from fpylll import LLL

    n_unanimes_all = []
    taux_correct_all = []
    instances_avec_unanime = 0

    for _ in range(n_runs):
        weights, T, solution = get_instance(n, density)
        B = knapsack_matrix(weights, T)
        LLL.reduction(B)
        sub = extract_coefficient_submatrix(B)
        projections = [project_to_binary(v) for v in sub]

        positions_unanimes = []
        positions_correctes = 0

        for i in range(n):
            vals = [p[i] for p in projections]
            if all(x == 1 for x in vals):
                positions_unanimes.append((i, 1))
                if solution[i] == 1:
                    positions_correctes += 1
            elif all(x == 0 for x in vals):
                positions_unanimes.append((i, 0))
                if solution[i] == 0:
                    positions_correctes += 1

        n_unanimes_all.append(len(positions_unanimes))
        if positions_unanimes:
            instances_avec_unanime += 1
            taux_correct_all.append(positions_correctes / len(positions_unanimes))

    avec = instances_avec_unanime
    print(f"\n{'='*55}")
    print(f"n={n}, density={density}, {n_runs} runs")
    print(f"{'='*55}")
    print(
        f"  Instances avec positions unanimes : {avec}/{n_runs} ({avec/n_runs*100:.1f}%)"
    )
    print(f"  Nb moy. positions unanimes        : {sum(n_unanimes_all)/n_runs:.2f}")
    if taux_correct_all:
        print(
            f"  Taux correct sur unanimes         : "
            f"{sum(taux_correct_all)/len(taux_correct_all)*100:.1f}%"
            f"  (aléatoire = 50%)"
        )


# =============================================================================
# Direct candidate analysis
# =============================================================================


def analyze_direct_candidates(
    n: int,
    density: float,
    n_runs: int = 200,
) -> None:
    """
    Analyze quality of binary candidates found directly by LLL (residual = 0).

    Measures whether direct candidates are the reference solution or
    alternative solutions, and how far they are from any valid solution.

    Args:
        n:       Instance dimension.
        density: Target density.
        n_runs:  Number of instances.
    """
    from fpylll import LLL

    ham_direct_all = []
    instances_avec_direct = 0
    instances_direct_est_solution = 0

    for _ in range(n_runs):
        weights, T, solution = get_instance(n, density)
        all_solutions = count_solutions(weights, T, max_solutions=100, timeout=2.0)

        if not all_solutions:
            continue

        B = knapsack_matrix(weights, T)
        LLL.reduction(B)
        sub = extract_coefficient_submatrix(B)

        directs = [
            h
            for h in filter_binary_candidates(sub)
            if abs(sum(weights[i] * h[i] for i in range(n)) - T) == 0
        ]

        if not directs:
            continue

        instances_avec_direct += 1
        for h in directs:
            ham_min = min_hamming_over_solutions(h, all_solutions)
            ham_direct_all.append(ham_min)
            if ham_min == 0:
                instances_direct_est_solution += 1

    print(f"\n{'='*55}")
    print(f"n={n}, density={density}, {n_runs} runs")
    print(f"{'='*55}")
    print(
        f"  Instances avec candidat direct résidu=0 : "
        f"{instances_avec_direct}/{n_runs} "
        f"({instances_avec_direct/n_runs*100:.1f}%)"
    )
    if ham_direct_all:
        print(
            f"  Hamming min moyen sur toutes solutions  : {sum(ham_direct_all)/len(ham_direct_all):.2f}"
        )
        print(
            f"  Hamming min = 0 (solution exacte)       : "
            f"{instances_direct_est_solution}/{len(ham_direct_all)} "
            f"({instances_direct_est_solution/len(ham_direct_all)*100:.1f}%)"
        )
        print(
            f"  Hamming min médiane                     : {sorted(ham_direct_all)[len(ham_direct_all)//2]}"
        )
