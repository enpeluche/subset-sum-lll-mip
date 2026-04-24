"""Solver factory: builds a named dict of solvers for each experiment suite."""

from functools import partial
from typing import Dict, Callable, Any


from solvers.solve_lattice_hybrid import solve_lattice_hybrid
from solvers.cpsat.solve_cpsat_greedy import (
    solve_cpsat_greedy_bound,
    solve_cpsat_smart_window,
    solve_cpsat_smart_tightened,
)

from solvers.cpsat.solve_greedy_full import solve_full_greedy

from solvers.solve_tabu import solve_tabu
from solvers.cpsat.solve_cpsat import solve_cpsat

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
        "lattice_bkz_adaptative": _suite_lattice_bkz_adaptative,
        "lattice_scaling_bkz": _suite_lattice_scaling_bkz,
        "lattice_scaling_lll": _suite_lattice_scaling_lll,
        "tabu_comp":   _suite_tabu_comp,
        "exact_comp":  _suite_exact_comp,
        "cpsat_comp":  _suite_cpsat_comp,
        "gray": _suite_gray_study,
        "gray_landscape": _suite_gray_landscape,
        "cpsat_formulation": _suite_cpsat_formulation,
        "cpsat_min_sat": _suite_cpsat_min_sat,
        "smart_lattice_cpsat": _suite_smart_lattice_cpsat,
    }
    return builders[suite](deltas, blocks, base_delta, base_block, timeout)

# satisfy vs minimize, vanilla
def _suite_cpsat_objective_impact(timeout) -> Dict[str, Callable]:
    from solvers.cpsat.solve_cpsat_minimize import solve_cpsat_minimize
    from solvers.cpsat.solve_cpsat import solve_cpsat

    return {
        "CPSAT-MIN": partial(solve_cpsat_minimize, timeout),
        "CPSAT-SAT": partial(solve_cpsat, timeout),
    }


# dual vs minimize & satisfy, vanilla
def _suite_cpsat_scout_efficiency(timeout) -> Dict[str, Callable]:
    from solvers.cpsat.solve_cpsat_minimize import solve_cpsat_dual, solve_cpsat_minimize
    from solvers.cpsat.solve_cpsat import solve_cpsat

    return {
        "CPSAT-DUAL": partial(solve_cpsat_dual, timeout=timeout),
        "CPSAT-MIN": partial(solve_cpsat_minimize, timeout),
        "CPSAT-SAT": partial(solve_cpsat, timeout),
    }


#solve_cpsat vs solve_cpsat_greedy_bound vs solve_cpsat_smart_tightened.
def _suite_cpsat_search_reduction(timeout) -> Dict[str, Callable]:
    from solvers.cpsat.solve_cpsat_greedy import solve_cpsat_greedy_bound
    from solvers.cpsat.solve_cpsat import solve_cpsat
    
    return {
        "CPSAT-VANILLA": partial(solve_cpsat, timeout=timeout),
        "CPSAT-GREEDY":  partial(solve_cpsat_greedy_bound, timeout=timeout),
        "CPSAT-SMART":   partial(solve_cpsat_smart_window, timeout=timeout),
        "CPSAT-TIGHT":   partial(solve_cpsat_smart_tightened, timeout=timeout),
    }

# differten grredy bound de cpsat
def _suite_cpsat_bound_tolerance(timeout) -> Dict[str, Callable]:
    from solvers.cpsat.solve_cpsat_greedy import solve_cpsat_greedy_bound

    return {
        "CPSAT-TOL-0": partial(solve_cpsat_greedy_bound, timeout=timeout, tolerance=0),
        "CPSAT-TOL-1": partial(solve_cpsat_greedy_bound, timeout=timeout, tolerance=1),
        "CPSAT-TOL-2": partial(solve_cpsat_greedy_bound, timeout=timeout, tolerance=2),
        "CPSAT-TOL-3": partial(solve_cpsat_greedy_bound, timeout=timeout, tolerance=3),
    }

Bennell, J.A., Song, X. A beam search implementation for the irregular shape packing problem. J Heuristics 16, 167–188 (2010). 
https://doi-org.sid2nomade-2.grenet.fr/10.1007/s10732-008-9095-x
C. Bierwirth, J. Kuhpfahl (2017), Extended GRASP for the job shop scheduling problem with total weighted tardiness objective, European Journal of Operational Research, 
olume 261, Issue 3, Pages 835-848, https://doi.org/10.1016/j.ejor.2017.03.030. 
E.G. Birgin, J.E. Ferreira, D.P. Ronconi (2020), A filtered beam search method for the m-machine permutation flowshop scheduling problem minimizing the earliness and tardiness penalties and the waiting time of the jobs,
Computers & Operations Research, Volume 114, 104824, ISSN 0305-0548, https://doi.org/10.1016/j.cor.2019.104824.
Bürgy, R. A neighborhood for complex job shop scheduling problems with regular objectives. J Sched 20, 391–422 (2017).
Mao Chen, Yajing Yang, Zeyu Zeng, Xiangyang Tang, Xicheng Peng, Sannuya Liu (2024), A filtered beam search based heuristic algorithm for packing unit circles into a circular container, Computers & Operations Research, Volume 1
Victor Fernandez-Viagas, Jose M. Framinan (2017), A beam-search-based constructive heuristic for the PFSP to minimise total flowtime, Computers & Operations Research, 
Volume 81, Pages 167-177, https://doi.org/10.1016/j.cor.2016.12.020. 
Victor Fernandez-Viagas, Jorge M.S. Valente, Jose M. Framinan (2018), Iterated-greedy-based algorithms with beam search initialization for the permutation flowshop to minimise total tardiness,
 Expert Systems with Applications, Volume 94, Pages 58-69, https://doi.org/10.1016/j.eswa.2017.10.050.
 J. Kuhpfahl, C. Bierwirth (2016), A study on local search neighborhoods for the job shop scheduling problem with total weighted tardiness objective, Computers & Operations Research, Volume 66, Pages 44-57, https://doi.org/10.1016/j.cor.2015.07.011. Li, Z., Janardhanan, M. N., & Rahman, H. F. (2021). Enhanced beam search heuristic for U-shaped assembly line balancing problems. Engineering Optimization, 53(4), 594–608. https://doi.org/10.1080/0305215X.2020.1741569 Zixiang Li, Ibrahim Kucukkoc, Qiuhua Tang (2021), Enhanced branch-bound-remember and iterative beam search algorithms for type II assembly line balancing problem, Computers & Operations Research, Volume 131, 2021, 105235, ISSN 0305-0548, https://doi.org/10.1016/j.cor.2021.105235. Luc Libralesso, Pablo Andres Focke, Aurélien Secardin, Vincent Jost (2022), Iterative beam search algorithms for the permutation flowshop, European Journal of Operational Research, Volume 301, Issue 1, Pages 217-234, ISSN 0377-2217, https://doi.org/10.1016/j.ejor.2021.10.015. Yazid Mati, Stèphane Dauzère-Pérès, Chams Lahlou (2011), A general approach for optimizing regular criteria in the job-shop scheduling problem, European Journal of Operational Research, Volume 212, Issue 1, Pages 33-42, https://doi.org/10.1016/j.ejor.2011.01.046. Rafael Morais, Teobaldo Bulhões, Anand Subramanian (2024), Exact and heuristic algorithms for minimizing the makespan on a single machine scheduling problem with sequence-dependent setup times and release dates, European Journal of Operational Research, Volume 315, Issue 2, Pages 442-453, ISSN 0377-2217, https://doi.org/10.1016/j.ejor.2023.11.024. Consuelo Parreño-Torres, Ramon Alvarez-Valdes, Francisco Parreño (2022), A beam search algorithm for minimizing crane times in premarshalling problems, European Journal of Operational Research, Volume 302, Issue 3, Pages 1063-1078, ISSN 0377-2217, https://doi.org/10.1016/j.ejor.2022.01.038. Rossi, F.L., Nagano, M.S. Beam search-based heuristics for the mixed no-idle flowshop with total flowtime criterion. OR Spectrum 44, 1311–1346 (2022). https://doi.org.sid2nomade-2.grenet.fr/10.1007/s00291-022-00678-9 Oleh Sobeyko, Lars Mönch (2016), Heuristic approaches for scheduling jobs in large-scale flexible job shops, Computers & Operations Research, Volume 68, Pages 97-109, https://doi.org/10.1016/j.cor.2015.11.004. 




# solve_lattice_hybrid vs solve_mitm_gray vs solve_tabu

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

def _suite_lattice_bkz_adaptative(deltas, blocks, base_delta, base_block, timeout) -> Dict[str, Callable]:
    suite = {
        "BKZ-Adaptative": partial(
            solve_lattice_hybrid, 
            strategy="ADAPTATIVE_BKZ", 
            block_size=30, 
            timeout=timeout
        )
    }

    for b in blocks:
        suite[f"BKZ-{int(b)}"] = partial(
            solve_lattice_hybrid, 
            strategy="BKZ_ONLY", 
            block_size=int(b), 
            timeout=timeout
        )

    return suite


def _suite_lattice_scaling_lll(deltas, blocks, base_delta, base_block, timeout) -> Dict[str, Callable]:
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
       
        # Version BKZ
        suite[f"LLL (M={label})"] = partial(
            solve_lattice_hybrid, strategy="LLL_ONLY", scaling=scale_val, block_size=base_block, timeout=timeout
        )
    return suite

def _suite_lattice_scaling_bkz(deltas, blocks, base_delta, base_block, timeout) -> Dict[str, Callable]:
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
    }

# Hybrid comparaison suites

def _suite_smart_lattice_cpsat(deltas, blocks, base_delta, base_block, timeout) -> Dict[str, Callable]:
    kw = dict(delta=base_delta, block_size=base_block, timeout=timeout)
    return {
        "Smart LLL->BKZ->Satisfy":            partial(solve_lattice_hybrid, strategy="SMART", **kw),
        "Satisfy":       partial(solve_cpsat, timeout=timeout),
        "Minimize":      partial(solve_cpsat_minimize, timeout=timeout),
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
        "Full Greedy":      partial(solve_full_greedy, tolerance_smart=3, timeout=timeout, workers=8),
    }

def _suite_cpsat_min_sat(_deltas, _blocks, _bd, _bb, timeout):
    return {
        "Satisfy":       partial(solve_cpsat, timeout=timeout),
        "Minimize":      partial(solve_cpsat_minimize, timeout=timeout),
    }  
    

def _suite_cpsat_formulation(_deltas, _blocks, _bd, _bb, timeout):

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
    from solvers.cpsat.solve_cpsat import solve_cpsat
    return {
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