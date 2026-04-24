"""
solve_tabu.py
-------------
Classic tabu search for Subset Sum.

O(n) per iteration: evaluates all n single-bit flips,
picks the best non-tabu move (with aspiration criterion).

Takes a hint (starting point) from the caller — no internal LLL.
The lattice warm start belongs in solve_ultimate, not here.

Supports restarts with random perturbation of the hint.
"""

import time
import random

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult


def _tabu_search(
    instance: SubsetSumInstance,
    x_init: list[int],
    n_iter: int,
    tenure: int,
    deadline: float,
) -> tuple[list[int] | None, int, int]:
    """
    Core tabu loop.

    Returns: (solution_or_None, branches, best_residual)
    """
    n = instance.n
    w = instance.weights

    x = x_init.copy()
    residual = instance.residual(x)

    best_x = x.copy()
    best_res = abs(residual)

    tabu_until = [0] * n
    branches = 0

    for it in range(n_iter):
        if time.perf_counter() > deadline:
            break

        best_j, best_move_res = -1, float("inf")

        for j in range(n):
            branches += 1
            delta = w[j] if x[j] == 0 else -w[j]
            new_res = abs(residual + delta)
            is_tabu = tabu_until[j] > it
            aspiration = new_res < best_res

            if (not is_tabu or aspiration) and new_res < best_move_res:
                best_j, best_move_res = j, new_res

        if best_j == -1:
            break

        # Apply move
        delta = w[best_j] if x[best_j] == 0 else -w[best_j]
        x[best_j] = 1 - x[best_j]
        residual += delta
        tabu_until[best_j] = it + tenure

        if abs(residual) < best_res:
            best_res = abs(residual)
            best_x = x.copy()

        if residual == 0:
            return x, branches, 0

    return (best_x if best_res == 0 else None), branches, best_res


def solve_tabu(
    instance: SubsetSumInstance,
    hint: list[int] | None = None,
    n_iter: int = 10_000,
    tenure: int | None = None,
    n_restarts: int = 5,
    timeout: float = 10.0,
    workers: int = 1,
) -> SolveResult:
    """
    Tabu search with restarts.

    Args:
        instance:    SubsetSumInstance to solve.
        hint:        Starting binary vector (e.g. from lattice hybrid).
                     Falls back to random if None.
        n_iter:      Max iterations per restart.
        tenure:      Tabu tenure (default n//4).
        n_restarts:  Number of restarts.
        timeout:     Wall clock budget in seconds.
    """
    start = time.perf_counter()
    n = instance.n

    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)

    if tenure is None:
        tenure = max(2, n // 4)

    # Starting point
    x_init = hint.copy() if hint else [random.randint(0, 1) for _ in range(n)]

    if instance.is_solution(x_init):
        return SolveResult.found(
            elapsed=time.perf_counter() - start,
            solution=x_init,
            label="Tabu_HintDirect",
            hamming_to_ground_solution=instance.hamming_to_solution(x_init),
        )

    total_branches = 0
    best_residual = abs(instance.residual(x_init))
    best_hint = x_init.copy()
    budget_per_restart = timeout / n_restarts

    for restart in range(n_restarts):
        remaining = timeout - (time.perf_counter() - start)
        if remaining <= 0:
            break

        deadline = time.perf_counter() + min(budget_per_restart, remaining)

        # First restart: use hint. Others: random perturbation.
        if restart == 0:
            x_start = x_init.copy()
        else:
            x_start = x_init.copy()
            k = random.randint(1, max(1, n // 5))
            for j in random.sample(range(n), k):
                x_start[j] = 1 - x_start[j]

        solution, branches, res = _tabu_search(
            instance, x_start, n_iter, tenure, deadline,
        )
        total_branches += branches

        if res < best_residual:
            best_residual = res

        if solution is not None and instance.is_solution(solution):
            return SolveResult.found(
                elapsed=time.perf_counter() - start,
                solution=solution,
                label=f"Tabu_r{restart + 1}",
                branches=total_branches,
                hamming_to_ground_solution=instance.hamming_to_solution(solution),
            )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        status=3,
        label="Tabu_NotFound",
        solution=None,
        branches=total_branches,
        smallest_residual=int(best_residual),
        hint=best_hint,
    )