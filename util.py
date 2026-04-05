"""
Core utilities for the LLL+CP-SAT hybrid Subset Sum benchmark.

Covers:
    - Instance generation with 64-bit overflow protection
    - Knapsack lattice matrix construction
    - LLL basis post-processing (coefficient extraction, binary filtering)
"""

import math
import random

from fpylll import IntegerMatrix


# =============================================================================
# Instance generation
# =============================================================================


def get_min_density(n: int) -> float:
    """
    Compute the minimum allowed density for dimension n.

    Ensures that generated weights stay within CP-SAT's 64-bit integer limit.
    The bound derives from MAX = 2^(n/d) < 2^63, giving d > n / (63 - log2(n) - 2).

    Args:
        n: Number of elements.

    Returns:
        Minimum safe density value for dimension n.
    """
    return n / int(63 - math.log2(n) - 2)


def get_instance(n: int, density: float) -> tuple[list[int], int, list[int]]:
    """
    Generate a random Subset Sum instance at a given density.

    Weights are sampled uniformly in [1, 2^(n/density)].
    A random binary solution is drawn and the target T is computed from it,
    guaranteeing the instance is feasible by construction.

    Args:
        n:       Number of elements.
        density: Target density d = n / log2(max weight).

    Returns:
        (weights, T, solution) where solution is the reference binary vector.

    Raises:
        AssertionError: If density is below the safe minimum for n.
    """
    min_density = get_min_density(n)
    assert (
        density >= min_density
    ), f"Density too low: for n={n}, d must be > {min_density:.2f}"

    MAX = int(2 ** (n / density))
    weights = [random.randint(1, MAX) for _ in range(n)]
    solution = [random.randint(0, 1) for _ in range(n)]
    T = sum(weights[i] * solution[i] for i in range(n))

    return weights, T, solution


# =============================================================================
# Knapsack lattice matrix
# =============================================================================


def knapsack_matrix(weights: list[int], T: int, M: int = 1) -> IntegerMatrix:
    """
    Build the (n+1) x (n+1) knapsack lattice matrix for Subset Sum.

    The matrix encodes the problem so that short vectors in the reduced basis
    correspond to candidate solutions. The scaling factor M amplifies the
    weight column relative to the identity block, steering LLL toward
    binary solutions.

    Structure:
        [ a_1*M   1  0  ...  0 ]
        [ a_2*M   0  1  ...  0 ]
        [  ...                 ]
        [ a_n*M   0  0  ...  1 ]
        [ -T*M    0  0  ...  0 ]

    Args:
        weights: List of n positive integers.
        T:       Target sum.
        M:       Scaling factor (default 1; set to 2^n for standard reduction).

    Returns:
        An fpylll IntegerMatrix of shape (n+1) x (n+1).
    """
    n = len(weights)
    A = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n):
        A[i][0] = weights[i] * M
        A[i][i + 1] = 1
    A[n][0] = -T * M
    return IntegerMatrix.from_matrix(A)


# =============================================================================
# LLL basis post-processing
# =============================================================================


def extract_coefficient_submatrix(matrix: IntegerMatrix) -> list[list[int]]:
    """
    Extract the coefficient submatrix from a reduced knapsack basis.

    Drops the first column (which encodes the weighted sum) and returns
    the remaining n columns, which encode candidate {-k,...,k}-vectors.

    Args:
        matrix: An LLL-reduced (n+1) x (n+1) IntegerMatrix.

    Returns:
        List of n+1 vectors of length n.
    """
    return [
        [matrix[j][i + 1] for i in range(matrix.ncols - 1)] for j in range(matrix.nrows)
    ]


def filter_binary_candidates(sub: list[list[int]]) -> list[list[int]]:
    """
    Filter vectors whose coefficients are all in {0, 1}.

    These are the only vectors that can directly encode a valid subset,
    and are used as hints for CP-SAT or as direct solutions.

    Args:
        sub: List of integer vectors (output of extract_coefficient_submatrix).

    Returns:
        Subset of vectors with all entries in {0, 1}.
    """
    return [row for row in sub if all(x in (0, 1) for x in row)]


def is_solution(weights: list[int], T: int, candidate: list[int]) -> bool:
    """
    Check whether a binary vector is a valid solution to the Subset Sum instance.

    Args:
        weights:   List of n positive integers.
        T:         Target sum.
        candidate: Binary vector of length n.

    Returns:
        True if sum(weights[i] * candidate[i]) == T, False otherwise.
    """
    return sum(w * x for w, x in zip(weights, candidate)) == T
