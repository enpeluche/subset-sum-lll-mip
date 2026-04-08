# compare_lll_bkz.py
"""
Compare LLL vs BKZ sur le taux de candidats binaires,
le temps de réduction et la qualité des vecteurs courts.

Réducteurs testés :
    LLL           — standard, delta=0.99
    BKZ(10)       — bloc 10, bon compromis vitesse/qualité
    BKZ(20)       — bloc 20, meilleur qualité, plus lent
    BKZ(30)       — bloc 30, très fort, potentiellement lent
"""

import time
import numpy as np
from fpylll import LLL, BKZ, IntegerMatrix
from utils import (
    get_instance,
    knapsack_matrix,
    extract_coefficient_submatrix,
    filter_binary_candidates,
)
from scipy import stats


# =============================================================================
# Helpers
# =============================================================================


def significance_marker(p):
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "~"
    return "="


def ttest_log(a, b):
    _, p = stats.ttest_ind(
        [np.log1p(x) for x in a],
        [np.log1p(x) for x in b],
        equal_var=False,
    )
    return p


def ztest_proportions(n_a, s_a, n_b, s_b):
    p_a = s_a / n_a
    p_b = s_b / n_b
    p_pool = (s_a + s_b) / (n_a + n_b)
    if p_pool in (0, 1):
        return 1.0
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = (p_a - p_b) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))


# =============================================================================
# Réducteurs
# =============================================================================

REDUCERS = {
    "LLL": lambda B: LLL.reduction(B, delta=0.99),
    "BKZ(10)": lambda B: BKZ.reduction(B, BKZ.Param(block_size=10)),
    "BKZ(20)": lambda B: BKZ.reduction(B, BKZ.Param(block_size=20)),
    "BKZ(30)": lambda B: BKZ.reduction(B, BKZ.Param(block_size=30)),
}


def run_reducer(B, name):
    """Lance le réducteur et mesure le temps."""
    start = time.perf_counter()
    REDUCERS[name](B)
    elapsed = time.perf_counter() - start
    return elapsed


# =============================================================================
# Benchmark
# =============================================================================


def run_comparison(n, densities, n_runs=50):
    """
    Pour chaque densité et chaque réducteur, mesure :
        - taux de candidats binaires
        - temps de réduction
        - résidu du meilleur vecteur
        - nombre moyen de candidats binaires
    """
    reducer_names = list(REDUCERS.keys())
    ref = "LLL"

    print(f"\nn={n}, {n_runs} runs per density")
    print(f"Legend: * p<0.05  ~ p<0.10  = not significant  (référence = LLL)")
    print(f"{'='*100}")

    header = (
        f"{'Density':>8} | {'Metric':<10} "
        + "".join(f"{r:>10}" for r in reducer_names)
        + "  "
        + "  ".join(f"{r}vLLL" for r in reducer_names[1:])
    )
    print(header)
    print(f"{'-'*100}")

    for d in densities:
        data = {
            r: {
                "has_binary": [],
                "n_binary": [],
                "time": [],
                "best_res": [],
            }
            for r in reducer_names
        }

        for _ in range(n_runs):
            weights, T, sol = get_instance(n, d)

            for name in reducer_names:
                # Construire une nouvelle matrice pour chaque réducteur
                B = knapsack_matrix(weights, T, 2 ** n)

                # Réduction + temps
                elapsed = run_reducer(B, name)
                sub = extract_coefficient_submatrix(B)

                # Candidats binaires
                candidates = filter_binary_candidates(sub)
                n_bin = len(candidates)

                # Meilleur résidu
                best_res = (
                    min(abs(sum(weights[i] * v[i] for i in range(n)) - T) for v in sub)
                    if sub
                    else float("inf")
                )

                data[name]["has_binary"].append(1 if n_bin > 0 else 0)
                data[name]["n_binary"].append(n_bin)
                data[name]["time"].append(elapsed)
                data[name]["best_res"].append(best_res)

        # Tests statistiques vs LLL
        p_binary = {
            r: ztest_proportions(
                n_runs,
                sum(data[r]["has_binary"]),
                n_runs,
                sum(data[ref]["has_binary"]),
            )
            for r in reducer_names[1:]
        }
        p_time = {
            r: ttest_log(data[r]["time"], data[ref]["time"]) for r in reducer_names[1:]
        }
        p_res = {
            r: ttest_log(data[r]["best_res"], data[ref]["best_res"])
            for r in reducer_names[1:]
        }

        def sm(p):
            return significance_marker(p)

        def avg(key, r):
            return np.mean(data[r][key])

        # --- Taux binaire ---
        line_bin = f"{d:>8.2f} | {'BINAIRE%':<10} "
        line_bin += "".join(f"{avg('has_binary', r)*100:>9.1f}%" for r in reducer_names)
        line_bin += "  " + "  ".join(f"{sm(p_binary[r]):>6}" for r in reducer_names[1:])
        print(line_bin)

        # --- Nombre moyen de candidats ---
        line_n = f"{'':>8}   {'N_CANDS':<10} "
        line_n += "".join(f"{avg('n_binary', r):>10.2f}" for r in reducer_names)
        print(line_n)

        # --- Temps de réduction ---
        line_t = f"{'':>8}   {'TIME(s)':<10} "
        line_t += "".join(f"{avg('time', r):>10.4f}" for r in reducer_names)
        line_t += "  " + "  ".join(f"{sm(p_time[r]):>6}" for r in reducer_names[1:])
        print(line_t)

        # --- Meilleur résidu ---
        line_r = f"{'':>8}   {'BEST_RES':<10} "
        line_r += "".join(f"{avg('best_res', r):>10.2e}" for r in reducer_names)
        line_r += "  " + "  ".join(f"{sm(p_res[r]):>6}" for r in reducer_names[1:])
        print(line_r)

        print(f"{'-'*100}")

    print(f"{'='*100}")
    print(f"Legend: * p<0.05  ~ p<0.10  = not significant")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    from utils import get_min_density
    from constants import N

    N = 30
    d_min = get_min_density(N)
    densities = [d_min + 0.10 * i for i in range(12)]

    run_comparison(n=N, densities=densities, n_runs=50)
