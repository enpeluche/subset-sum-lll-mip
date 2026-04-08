from fpylll import IntegerMatrix


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
    return [
        [matrix[j][i + 1] for i in range(matrix.ncols - 1)] for j in range(matrix.nrows)
    ]


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
    return [row for row in sub if all(x in (0, 1) for x in row)]
