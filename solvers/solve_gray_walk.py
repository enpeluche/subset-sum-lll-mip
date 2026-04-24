"""
solve_gray_walk.py
------------------
Gray code walk on {0,1}^n from a given starting point.

This is NOT a tabu search — every step is taken unconditionally.
The Gray code guarantees a Hamiltonian path on the hypercube:
each of the 2^n vertices is visited exactly once.

Exact solver for n ≤ 21 (~2M iterations, ~800ms).
For larger n, partial walk from the hint provided by lattice reduction.

Complexity: O(min(2^n, max_iter)) time, O(1) space.
"""

import time

from SubsetSumInstance import SubsetSumInstance
from results import SolveResult
from gray import gray_bit, gray_budget


def solve_gray_walk(
    instance: SubsetSumInstance,
    hint: list[int] | None = None,
    max_iter: int = 2_000_000,
    timeout: float = 10.0,
    workers: int = 1,
) -> SolveResult:
    """
    Gray code walk from hint (or zeros if no hint).

    The walk visits hint ⊕ gray(1), hint ⊕ gray(2), ...
    This is a shifted Hamiltonian path — bijection on {0,1}^n.

    Args:
        instance:   SubsetSumInstance to solve.
        hint:       Starting binary vector (e.g. from lattice reduction).
        max_iter:   Iteration cap (default 2M, exact for n ≤ 21).
        timeout:    Wall clock budget.
    """
    start = time.perf_counter()
    n = instance.n

    if instance.is_trivially_infeasible:
        return SolveResult.trivially_infeasible(time.perf_counter() - start)

    # Start from hint or zero vector
    x = (hint.copy() if hint else [0] * n)
    residual = instance.residual(x)

    if residual == 0 and instance.is_solution(x):
        return SolveResult.found(
            elapsed=time.perf_counter() - start,
            solution=x,
            label="GrayWalk_HintDirect",
            hamming_to_ground_solution=instance.hamming_to_solution(x),
        )

    best_x = x.copy()
    best_res = abs(residual)
    branches = 0
    limit = gray_budget(n, max_iter)
    deadline = start + timeout
    w = instance.weights

    for it in range(1, limit + 1):
        if time.perf_counter() > deadline:
            break

        j = gray_bit(it)
        if j >= n:
            break

        branches += 1

        if x[j] == 0:
            x[j] = 1
            residual += w[j]
        else:
            x[j] = 0
            residual -= w[j]

        ar = abs(residual)
        if ar < best_res:
            best_res = ar
            best_x = x.copy()

        if residual == 0:
            is_exact = (limit >= (1 << n) - 1)
            return SolveResult.found(
                elapsed=time.perf_counter() - start,
                solution=x,
                label=f"GrayWalk_{'Exact' if is_exact else 'Found'}",
                branches=branches,
                hamming_to_ground_solution=instance.hamming_to_solution(x),
            )

    return SolveResult(
        elapsed=time.perf_counter() - start,
        status=3,
        label="GrayWalk_NotFound",
        solution=None,
        branches=branches,
        smallest_residual=int(best_res),
        hint=best_x,
    )