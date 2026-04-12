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

from solvers.solve_tabu import solve_tabu
from solvers.solve_mitm import solve_mitm_classic

def build_solvers(ranges: dict, suite: str, timeout: float) -> dict:
    """
    Return {name: partial(solve_lattice_hybrid, ...)} for the chosen suite.

    Suites:
        delta      — one solver per delta value (LLL only)
        block      — one solver per block_size (BKZ only)
        arch       — LLL / BKZ / SEQ / INDEP head-to-head
        hybrid_comp — SEQ vs INDEP focus
        tabu_comp   — tabu engines × warm starts
        exact_comp  — MITM vs CP-SAT vs Greedy Extreme
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
        "tabu_comp":   _suite_tabu_comp,
        "exact_comp":  _suite_exact_comp,
        "cpsat_comp":  _suite_cpsat_comp,
        "scaling": _suite_scaling,
        "gray": _suite_gray_study,
        "gray_landscape": _suite_gray_landscape,
    }
    return builders[suite](deltas, blocks, base_delta, base_block, timeout)


# ------------------------------------------------------------------

def _suite_delta(deltas, _blocks, _bd, _bb, timeout):
    return {
        f"LLL-{d}": partial(solve_lattice_hybrid, strategy="LLL_ONLY", delta=d, timeout=timeout)
        for d in deltas
    }

def _suite_scaling(_deltas, _blocks, _bd, _bb, timeout):
    from solvers.solve_lattice_hybrid import solve_lattice_hybrid
    return {
        "M=2^n":        partial(solve_lattice_hybrid, strategy="LLL_ONLY", scaling="2n", timeout=timeout),
        "M=sqrt(n)":    partial(solve_lattice_hybrid, strategy="LLL_ONLY", scaling="sqrt_n", timeout=timeout),
        "M=n":          partial(solve_lattice_hybrid, strategy="LLL_ONLY", scaling="n", timeout=timeout),
        "M=sum(w)":     partial(solve_lattice_hybrid, strategy="LLL_ONLY", scaling="sum_w", timeout=timeout),
        "M=2^(n_div_2)":    partial(solve_lattice_hybrid, strategy="LLL_ONLY", scaling="2n2", timeout=timeout),
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


# ------------------------------------------------------------------
# Tabu suites
# ------------------------------------------------------------------
 
def _suite_tabu_comp(_deltas, _blocks, _bd, _bb, timeout):
    return {
        # Engine comparison (all with LLL warm start)
        "Tabu Classic":  partial(solve_tabu, engine="classic", warm_start="lll", timeout=timeout),
        "Tabu Gray":     partial(solve_tabu, engine="gray", warm_start="lll", n_iter=500_000, timeout=timeout),
        #"Tabu Beckett":  partial(solve_tabu, engine="beckett", warm_start="lll", n_iter=500_000, timeout=timeout),
        # Warm start comparison (all with classic engine)
        #"Classic+Random": partial(solve_tabu, engine="classic", warm_start="random", timeout=timeout),
        #"Classic+Sign":   partial(solve_tabu, engine="classic", warm_start="sign", timeout=timeout),
    }
 
 

def _suite_gray_study(_deltas, _blocks, _bd, _bb, timeout):
    return {
        "Gray Walk":  partial(solve_tabu, engine="gray_walk", warm_start="lll",
                              n_iter=2_000_000, timeout=timeout),
        "Gray Tabu":  partial(solve_tabu, engine="gray", warm_start="lll",
                              n_iter=2_000_000, timeout=timeout),
        "Classic":    partial(solve_tabu, engine="classic", warm_start="lll",
                              n_iter=10_000, timeout=timeout),
    }

def _suite_gray_landscape(_deltas, _blocks, _bd, _bb, timeout):
    from solvers.solve_gray_landscape import solve_gray_landscape
    from solvers.solve_tabu import solve_tabu
    from solvers.solve_cpsat import solve_cpsat
    return {
        "CP-SAT Vanilla":      partial(solve_cpsat, timeout=timeout, workers=6),
        "Gray Walk":            partial(solve_tabu, engine="gray_walk", warm_start="lll",
                                        n_iter=2_000_000, timeout=timeout),
        "Gray Tabu":            partial(solve_tabu, engine="gray", warm_start="lll",
                                        n_iter=2_000_000, timeout=timeout),
        "Landscape Hints":      partial(solve_gray_landscape, confidence=0.3,
                                        fix_variables=False, timeout=timeout),
        "Landscape Fix":        partial(solve_gray_landscape, confidence=0.3,
                                        fix_variables=True, timeout=timeout),
    }
# ------------------------------------------------------------------
# Exact methods comparison
# ------------------------------------------------------------------
 
def _suite_exact_comp(_deltas, _blocks, _bd, _bb, timeout):
    return {
        "CP-SAT Vanilla": partial(solve_cpsat, timeout=timeout, workers=8),
        "MITM":           partial(solve_mitm_classic, max_subsets=1 << 22),
        "Greedy Extreme":  partial(solve_greedy_extreme, max_subsets=2_000_000, timeout=timeout),
        "Full Greedy":     partial(solve_full_greedy, tolerance_smart=3, timeout=timeout, workers=8),
    }