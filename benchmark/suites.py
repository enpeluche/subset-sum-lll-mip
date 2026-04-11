"""Solver factory: builds a named dict of solvers for each experiment suite."""

from functools import partial
from solvers.solve_lattice_hybrid import solve_lattice_hybrid
from solvers.solve_greedy_cpsat import (
    solve_cpsat_greedy_bound,
    solve_cpsat_smart_window,
    solve_cpsat_smart_tightened,
)
from solvers.solve_cpsat import solve_cpsat
from solvers.solve_greedy_extreme import solve_greedy_extreme
from solvers.solve_greedy_full import solve_full_greedy

def build_solvers(ranges: dict, suite: str, timeout: float) -> dict:
    """
    Return {name: partial(solve_lattice_hybrid, ...)} for the chosen suite.

    Suites:
        delta      — one solver per delta value (LLL only)
        block      — one solver per block_size (BKZ only)
        arch       — LLL / BKZ / SEQ / INDEP head-to-head
        hybrid_comp — SEQ vs INDEP focus
        cpsat_comp  — all CP-SAT variants (vanilla → full greedy)
        mega_test  — cartesian product delta × block_size
    """
    deltas = ranges.get("delta", [0.99])
    blocks = ranges.get("block_size", [20])
    base_delta = deltas[0]
    base_block = int(blocks[0])

    builders = {
        "delta":     _suite_delta,
        "block":     _suite_block,
        "arch":      _suite_arch,
        "hybrid_comp": _suite_hybrid_comp,
        "mega_test": _suite_mega,
        "cpsat_comp":  _suite_cpsat_comp,
    }
    return builders[suite](deltas, blocks, base_delta, base_block, timeout)


# ------------------------------------------------------------------

def _suite_delta(deltas, _blocks, _bd, _bb, timeout):
    return {
        f"LLL-{d}": partial(solve_lattice_hybrid, strategy="LLL_ONLY", delta=d, timeout=timeout)
        for d in deltas
    }


def _suite_block(_deltas, blocks, _bd, _bb, timeout):
    return {
        f"BKZ-{int(b)}": partial(solve_lattice_hybrid, strategy="BKZ_ONLY", block_size=int(b), timeout=timeout)
        for b in blocks
    }


def _suite_arch(_deltas, _blocks, base_delta, base_block, timeout):
    kw = dict(delta=base_delta, block_size=base_block, timeout=timeout)
    return {
        "LLL":            partial(solve_lattice_hybrid, strategy="LLL_ONLY", **kw),
        "BKZ":            partial(solve_lattice_hybrid, strategy="BKZ_ONLY", **kw),
        "SEQ (LLL->BKZ)": partial(solve_lattice_hybrid, strategy="SEQ_LLL_BKZ", **kw),
        "INDEP":          partial(solve_lattice_hybrid, strategy="INDEP_LLL_BKZ", **kw),
    }


def _suite_hybrid_comp(_deltas, _blocks, base_delta, base_block, timeout):
    """Focus strict sur la comparaison des deux architectures hybrides."""
    kw = dict(delta=base_delta, block_size=base_block, timeout=timeout)
    return {
        "SEQ_LLL_BKZ":   partial(solve_lattice_hybrid, strategy="SEQ_LLL_BKZ", **kw),
        "INDEP_LLL_BKZ": partial(solve_lattice_hybrid, strategy="INDEP_LLL_BKZ", **kw),
    }


def _suite_mega(_deltas, _blocks, _bd, _bb, timeout):
    # Uses the full lists (not base values)
    from itertools import product as xprod
    return {
        f"SEQ-d{d}-b{int(b)}": partial(
            solve_lattice_hybrid, strategy="SEQ", delta=d, block_size=int(b), timeout=timeout
        )
        for d, b in xprod(_deltas, _blocks)  # BUG FIX: was using wrong vars
    }

# ------------------------------------------------------------------
# CP-SAT suites
# ------------------------------------------------------------------
 
def _suite_cpsat_comp(_deltas, _blocks, _bd, _bb, timeout):
    return {
        "CP-SAT Vanilla":   partial(solve_cpsat, timeout=timeout, workers=8),
        "Greedy Bound":     partial(solve_cpsat_greedy_bound, tolerance=0, timeout=timeout, workers=8),
        "Smart Window":     partial(solve_cpsat_smart_window, tolerance=3, timeout=timeout, workers=8),
        "Smart+Tightening": partial(solve_cpsat_smart_tightened, tolerance=3, timeout=timeout, workers=8),
        "Greedy Extreme":   partial(solve_greedy_extreme, max_subsets=2_000_000, timeout=timeout),
        "Full Greedy":      partial(solve_full_greedy, tolerance_smart=3, timeout=timeout, workers=8),
    }