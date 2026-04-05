"""
LLL hint analysis utilities.

Covers projection strategies, Hamming metrics, local search,
and combinatorial hint construction from LLL reduced bases.
"""

from itertools import combinations


# =============================================================================
# Projection strategies
# =============================================================================


def project_to_binary(v: list[int]) -> list[int]:
    """
    Clip each coordinate to {0, 1}: negative values → 0, values > 1 → 1.

    This is the tightest possible projection — it can only maintain or
    improve Hamming distance to any {0,1} vector, never worsen it.

    Args:
        v: Integer vector with arbitrary coefficients.

    Returns:
        Binary vector of the same length.
    """
    return [max(0, min(1, x)) for x in v]


def project_majority(sub: list[list[int]], n: int) -> list[int]:
    """
    For each position, vote 1 if more vectors are positive than negative.

    Args:
        sub: List of integer vectors (LLL reduced basis coefficients).
        n:   Vector length.

    Returns:
        Binary consensus vector of length n.
    """
    hint = []
    for i in range(n):
        vals = [v[i] for v in sub]
        positifs = sum(1 for x in vals if x > 0)
        negatifs = sum(1 for x in vals if x < 0)
        hint.append(1 if positifs > negatifs else 0)
    return hint


def project_weighted(sub: list[list[int]], n: int) -> list[int]:
    """
    Weighted vote: earlier (shorter) vectors have higher weight (1/(rank+1)).

    Args:
        sub: List of integer vectors sorted by norm (LLL order).
        n:   Vector length.

    Returns:
        Binary consensus vector of length n.
    """
    hint = []
    for i in range(n):
        score = sum(v[i] / (r + 1) for r, v in enumerate(sub))
        hint.append(1 if score > 0 else 0)
    return hint


def project_consensus(sub: list[list[int]], n: int) -> list[int | None]:
    """
    Strict consensus: assign a value only when all vectors agree in sign.
    Ambiguous positions are marked None (CP-SAT remains free on those).

    Args:
        sub: List of integer vectors.
        n:   Vector length.

    Returns:
        Vector of length n with values in {0, 1, None}.
    """
    hint = []
    for i in range(n):
        vals = [v[i] for v in sub]
        if all(x >= 0 for x in vals):
            hint.append(1 if sum(vals) > 0 else 0)
        elif all(x <= 0 for x in vals):
            hint.append(0)
        else:
            hint.append(None)
    return hint


# =============================================================================
# Hamming metrics
# =============================================================================


def hamming(v: list[int], solution: list[int]) -> int:
    """Hamming distance between two equal-length binary vectors."""
    return sum(1 for i in range(len(solution)) if v[i] != solution[i])


def hamming_consensus(v: list[int | None], solution: list[int]) -> int:
    """Hamming distance ignoring None positions."""
    return sum(
        1 for i in range(len(solution)) if v[i] is not None and v[i] != solution[i]
    )


def hamming_consensus_details(
    cons: list[int | None], solution: list[int]
) -> tuple[int, int]:
    """
    Hamming distance restricted to covered (non-None) positions.

    Returns:
        (hamming_on_covered, number_of_covered_positions)
    """
    covered = [(i, cons[i]) for i in range(len(solution)) if cons[i] is not None]
    ham_covered = sum(1 for i, v in covered if v != solution[i])
    return ham_covered, len(covered)


def coverage_consensus(v: list[int | None]) -> float:
    """Fraction of non-None positions in a consensus vector."""
    return sum(1 for x in v if x is not None) / len(v)


def count_non_binary(v: list[int]) -> int:
    """Count coordinates not in {0, 1}."""
    return sum(1 for x in v if x not in (0, 1))


# =============================================================================
# Local search
# =============================================================================


def local_search_around_projection(
    v_proj: list[int],
    weights: list[int],
    T: int,
    max_flips: int = 2,
) -> list[int] | None:
    """
    Search for an exact solution by flipping up to max_flips bits of v_proj.

    This is an exact k-opt search guided by the residual structure:
    since all a_i are known, we can check each neighbor in O(n^k).

    Args:
        v_proj:    Binary starting point (projected LLL vector).
        weights:   Instance weights.
        T:         Target sum.
        max_flips: Maximum number of simultaneous bit flips (1, 2, or 3).

    Returns:
        A valid solution vector if found, None otherwise.
    """
    n = len(weights)

    def residual(v):
        return abs(sum(weights[i] * v[i] for i in range(n)) - T)

    if residual(v_proj) == 0:
        return v_proj

    # 1-flip
    for i in range(n):
        v = v_proj.copy()
        v[i] = 1 - v[i]
        if residual(v) == 0:
            return v

    if max_flips < 2:
        return None

    # 2-flip
    for i in range(n):
        for j in range(i + 1, n):
            v = v_proj.copy()
            v[i] = 1 - v[i]
            v[j] = 1 - v[j]
            if residual(v) == 0:
                return v

    if max_flips < 3:
        return None

    # 3-flip
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                v = v_proj.copy()
                v[i] = 1 - v[i]
                v[j] = 1 - v[j]
                v[k] = 1 - v[k]
                if residual(v) == 0:
                    return v

    return None


# =============================================================================
# Combinatorial hint construction
# =============================================================================


def find_binary_from_combinations(
    sub: list[list[int]],
    weights: list[int],
    T: int,
    max_k: int = 3,
) -> list[list[int]]:
    """
    Search for binary solution vectors by taking signed sums of LLL vectors.

    Tries all pairs (k=2) and optionally triplets (k=3) with all sign
    combinations. Returns vectors that are binary AND have residual 0.

    Args:
        sub:    LLL coefficient vectors.
        weights: Instance weights.
        T:      Target sum.
        max_k:  Maximum combination size (2 or 3).

    Returns:
        List of valid binary solution vectors found by combination.
    """
    found = []
    n = len(weights)

    def check(s):
        if all(x in (0, 1) for x in s):
            if abs(sum(weights[i] * s[i] for i in range(n)) - T) == 0:
                found.append(s[:])

    for v1, v2 in combinations(sub, 2):
        for signs in [(1, 1), (1, -1), (-1, 1)]:
            check([signs[0] * v1[i] + signs[1] * v2[i] for i in range(n)])

    if max_k >= 3:
        for v1, v2, v3 in combinations(sub, 3):
            for signs in [
                (1, 1, 1),
                (1, 1, -1),
                (1, -1, 1),
                (-1, 1, 1),
                (1, -1, -1),
                (-1, 1, -1),
                (-1, -1, 1),
            ]:
                check(
                    [
                        signs[0] * v1[i] + signs[1] * v2[i] + signs[2] * v3[i]
                        for i in range(n)
                    ]
                )

    return found


# =============================================================================
# Display helpers
# =============================================================================


def display_vectors_with_residual(
    vectors: list[list[int]],
    weights: list[int],
    T: int,
    solution: list[int] | None = None,
) -> list[list[int]]:
    """
    Print each vector with its residual and optionally its Hamming distance.

    Args:
        vectors:  List of integer vectors to display.
        weights:  Instance weights.
        T:        Target sum.
        solution: If provided, also print Hamming distance to this vector.

    Returns:
        The input vectors unchanged (for chaining).
    """
    n = len(weights)
    for v in vectors:
        residual = abs(sum(weights[i] * v[i] for i in range(n)) - T)
        if solution is not None:
            ham = sum(1 for i in range(n) if v[i] != solution[i])
            print(
                f"[ {' '.join(f'{x:2}' for x in v)} ]"
                f"  résidu = {residual}  hamming = {ham}"
            )
        else:
            print(f"[ {' '.join(f'{x:2}' for x in v)} ]  résidu = {residual}")
    return vectors
