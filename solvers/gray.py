"""
gray.py
-------
Gray code utilities for subset sum solvers.

A Gray code is a Hamiltonian path on the hypercube {0,1}^n:
each step flips exactly one bit, visiting all 2^n vertices.

Used by:
    - solve_mitm_gray.py   (enumeration of half-sums)
    - solve_gray_walk.py   (structured local search)
    - solve_tabu.py        (Gray-ordered bit selection)
"""


def gray_bit(i: int) -> int:
    """Return the index of the bit that flips between Gray(i-1) and Gray(i)."""
    return (i & -i).bit_length() - 1


def gray_budget(n: int, max_iter: int = 2_000_000) -> int:
    """Full enumeration if feasible, capped otherwise."""
    return min((1 << n) - 1, max_iter)


def mask_to_bits(mask: int, length: int) -> list[int]:
    """Convert a bitmask to a list of 0/1 of given length."""
    return [(mask >> j) & 1 for j in range(length)]