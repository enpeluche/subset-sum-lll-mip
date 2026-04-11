from fpylll import IntegerMatrix
import math

def extract_vectors_from_basis(matrix: IntegerMatrix) -> list[list[int]]:
    """
    Extract the coefficient submatrix from a reduced knapsack basis.

    Drops the first column (which encodes the weighted sum) and returns
    the remaining n columns, which encode candidate {-k,...,k}-vectors.

    Args:
        matrix: An LLL-reduced (n+1) x (n+1) IntegerMatrix.

    Returns:
        List of n+1 vectors of length n.
    """

    n = matrix.ncols - 1
    extracted = []

    for j in range(matrix.nrows):
        v = [matrix[j][i + 1] for i in range(n)]
        extracted.append(v)

    extracted.sort(key=lambda v: sum(x*x for x in v))
    
    return extracted

def filter_binary_vectors(sub: list[list[int]]) -> list[list[int]]:
    """
    Filter vectors whose coefficients are all in {0, 1}.

    These are the only vectors that can directly encode a valid subset,
    and are used as hints for CP-SAT or as direct solutions.

    Args:
        sub: List of integer vectors (output of extract_vectors_from_basis).

    Returns:
        Subset of vectors with all entries in {0, 1}.
    """
    binary_set = {0, 1}

    return [v for v in sub if all(x in binary_set for x in v)]


def search_space_window(n: int, k_lo: int, k_hi: int) -> int:
    """Total subsets in [k_lo, k_hi]."""
    return sum(math.comb(n, k) for k in range(k_lo, k_hi + 1))


def enum_cost(n: int, k: int) -> int:
    """Cost of enumerating subsets of size k — uses symmetry."""
    return min(math.comb(n, k), math.comb(n, n - k))