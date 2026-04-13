"""Solver factory: builds a named dict of solvers for each experiment suite."""

from functools import partial
from typing import Dict, Callable, Any


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
        lattice_delta       — Evaluates LLL performance across different delta values
        lattice_block       — Assesses BKZ performance by varying the block size
        lattice_arch        — Compares standalone solvers (LLL/BKZ) vs. hybrid architectures
        lattice_hybrid_comp — Strict head-to-head of Sequential vs. Independent hybrids
        lattice_scaling     — Tests dynamic scaling strategies for LLL and BKZ
        tabu_comp   — tabu engines × warm starts
        exact_comp  — MITM vs CP-SAT vs Greedy Extreme
        cpsat_comp  — all CP-SAT variants (vanilla → full greedy)
    """
    deltas = ranges.get("delta", [0.99])
    blocks = ranges.get("block_size", [20])
    base_delta = deltas[0]
    base_block = int(blocks[0])

    builders = {
        "lattice_delta":     _suite_lattice_delta,
        "lattice_block":     _suite_lattice_block,
        "lattice_arch":      _suite_lattice_arch,
        "lattice_hybrid_comp": _suite_lattice_hybrid_comp,
        "lattice_scaling": _suite_lattice_scaling,
        "tabu_comp":   _suite_tabu_comp,
        "exact_comp":  _suite_exact_comp,
        "cpsat_comp":  _suite_cpsat_comp,
        "gray": _suite_gray_study,
        "gray_landscape": _suite_gray_landscape,
        "cpsat_formulation": _suite_cpsat_formulation,
    }
    return builders[suite](deltas, blocks, base_delta, base_block, timeout)





def _suite_lattice_delta(deltas, blocks, base_delta, base_block, timeout) -> Dict[str, Callable]:
    """Fait varier le paramètre delta pour LLL."""
    return {
        f"LLL-{d}": partial(solve_lattice_hybrid, strategy="LLL_ONLY", delta=d, timeout=timeout)
        for d in deltas
    }


def _suite_lattice_block(deltas, blocks, base_delta, base_block, timeout) -> Dict[str, Callable]:
    """Fait varier la taille de bloc pour BKZ."""
    return {
        f"BKZ-{int(b)}": partial(solve_lattice_hybrid, strategy="BKZ_ONLY", block_size=int(b), timeout=timeout)
        for b in blocks
    }


def _suite_lattice_scaling(deltas, blocks, base_delta, base_block, timeout) -> Dict[str, Callable]:
    """Génère les stratégies de scaling dynamiquement pour LLL ET BKZ."""
    scalings = {
        "2^n": "2n",
        "sqrt(n)": "sqrt_n",
        "n": "n",
        "sum(w)": "sum_w",
        "2^(n_div_2)": "2n2"
    }
    
    suite = {}
    for label, scale_val in scalings.items():
        # Version LLL
        suite[f"LLL (M={label})"] = partial(
            solve_lattice_hybrid, strategy="LLL_ONLY", scaling=scale_val, delta=base_delta, timeout=timeout
        )
        # Version BKZ
        suite[f"BKZ (M={label})"] = partial(
            solve_lattice_hybrid, strategy="BKZ_ONLY", scaling=scale_val, block_size=base_block, timeout=timeout
        )
    return suite


def _suite_lattice_arch(deltas, blocks, base_delta, base_block, timeout) -> Dict[str, Callable]:
    """Compare toutes les architectures avec les paramètres par défaut."""
    kw = dict(delta=base_delta, block_size=base_block, timeout=timeout)
    return {
        "LLL":            partial(solve_lattice_hybrid, strategy="LLL_ONLY", **kw),
        "BKZ":            partial(solve_lattice_hybrid, strategy="BKZ_ONLY", **kw),
        "SEQ (LLL->BKZ)": partial(solve_lattice_hybrid, strategy="SEQ_LLL_BKZ", **kw),
        "INDEP":          partial(solve_lattice_hybrid, strategy="INDEP_LLL_BKZ", **kw),
    }


def _suite_lattice_hybrid_comp(deltas, blocks, base_delta, base_block, timeout) -> Dict[str, Callable]:
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

def _suite_cpsat_formulation(_deltas, _blocks, _bd, _bb, timeout):
    from solvers.solve_cpsat import solve_cpsat
    from solvers.solve_cpsat_optim import (
        solve_cpsat_satisfy,
        solve_cpsat_minimize,
        solve_cpsat_minimize_lll,
        solve_cpsat_dual_lll,
    )
    return {
        # Baseline
        "Satisfy":       partial(solve_cpsat, timeout=timeout),
        # Pure optimization (no LLL)
        "Minimize":      partial(solve_cpsat_minimize, timeout=timeout),
        # Optimization with LLL warm start + residual bound
        "Minimize+LLL":  partial(solve_cpsat_minimize_lll, timeout=timeout),
        # Dual phase: scout (optimize 30%) → solve (satisfy 70%)
        "Dual+LLL":      partial(solve_cpsat_dual_lll, timeout=timeout, scout_ratio=0.3),
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