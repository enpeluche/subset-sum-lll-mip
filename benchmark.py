"""
Benchmarking utilities for the LLL+CP-SAT hybrid solver.

Four main functions:
    - format_pct:        Format a percentage change between two values.
    - print_run_results: Pretty-print a single run comparison.
    - run_benchmark:     Run the full benchmark loop across densities.
    - compute_stats:     Aggregate raw results into summary statistics.
"""

import csv
import numpy as np
from ortools.sat.python import cp_model

from util import get_instance
from solve_lll import solve_lll
from solve_ss_cpsat import solve_subset_sum_cpsat


# =============================================================================
# Constants
# =============================================================================

STATUS_MAP = {
    cp_model.UNKNOWN: "UNKNOWN",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.OPTIMAL: "OPTIMAL",
}

SOLVED_STATUSES = {cp_model.OPTIMAL, cp_model.FEASIBLE}


# =============================================================================
# Formatting helpers
# =============================================================================


def format_pct(new: float, base: float) -> str:
    """
    Return a signed percentage change string between two values.

    Args:
        new:  The new value (e.g. hybrid time).
        base: The reference value (e.g. baseline time).

    Returns:
        A formatted string like '+12.3%' or '-5.1%', or 'N/A' if base is zero.
    """
    if base == 0:
        return "N/A"
    return f"{((new - base) / base) * 100:+.1f}%"


def print_run_results(
    i: int,
    time_b: float,
    branch_b: int,
    conf_b: int,
    status_b: int,
    time_h: float,
    branch_h: int,
    conf_h: int,
    status_h: int,
    solution,
) -> None:
    """
    Pretty-print the comparison between baseline and hybrid for a single run.

    Args:
        i:        Run index.
        time_b:   Baseline solve time.
        branch_b: Baseline branch count.
        conf_b:   Baseline conflict count.
        status_b: Baseline CP-SAT status code.
        time_h:   Hybrid solve time.
        branch_h: Hybrid branch count.
        conf_h:   Hybrid conflict count.
        status_h: Hybrid CP-SAT status code.
        solution: Solution vector found by the hybrid, or None.
    """
    str_stat_b = STATUS_MAP.get(status_b, "ERR")
    str_stat_h = STATUS_MAP.get(status_h, "ERR")

    # Special label when LLL solved directly without any CP-SAT branching
    if time_h == 0.0 and status_h == cp_model.OPTIMAL:
        str_stat_h = "EXACT_LLL"

    print(
        f"Run {i}\n"
        f" Statut   -> Base: {str_stat_b:<10} | Hybride: {str_stat_h}\n"
        f"Temps    -> Base: {time_b:.3f}s       | Hybride: {time_h:.3f}s"
        f"      ({format_pct(time_h, time_b)})\n"
        f"Branches -> Base: {branch_b:<10} | Hybride: {branch_h}"
        f"    ({format_pct(branch_h, branch_b)})\n"
        f"Conflits -> Base: {conf_b:<10} | Hybride: {conf_h}"
        f"    ({format_pct(conf_h, conf_b)})\n"
        f"{solution}\n"
    )


# =============================================================================
# Benchmark loop
# =============================================================================


def run_benchmark(
    n: int,
    densities: list[float],
    n_runs: int,
    csv_file: str,
) -> tuple[dict, float, float]:
    """
    Run the full benchmark comparing vanilla CP-SAT against the hybrid LLL solver.

    For each density and each run:
        - Generates a random Subset Sum instance of dimension n.
        - Solves it with vanilla CP-SAT (baseline).
        - Solves it with the hybrid LLL+CP-SAT solver.
        - Records all metrics in memory and appends a row to the CSV file.

    Args:
        n:         Instance dimension (number of elements).
        densities: List of density values to benchmark.
        n_runs:    Number of runs per density.
        csv_file:  Path to the CSV file for persistent logging.

    Returns:
        A tuple (results, total_time_cpsat, total_time_hybrid) where results
        maps each density to its collected metrics:
        {
            density: {
                "time_base":     list[float],
                "branches_base": list[int],
                "conflicts_base":list[int],
                "success_base":  list[int],
                "time_hy":       list[float],
                "branches_hy":   list[int],
                "conflicts_hy":  list[int],
                "success_hy":    list[int],
                "residual":      list[float],
                "hamming":       list[int],
            }
        }
    """
    results: dict = {}
    total_time_cpsat: float = 0.0
    total_time_hybrid: float = 0.0

    for d in densities:
        print(f"\nDensity: {d}")

        results[d] = {
            "time_base": [],
            "branches_base": [],
            "conflicts_base": [],
            "success_base": [],
            "time_hy": [],
            "branches_hy": [],
            "conflicts_hy": [],
            "success_hy": [],
            "residual": [],
            "hamming": [],
        }

        for i in range(n_runs):
            # --- Instance generation ---
            a, T, sol = get_instance(n, d)

            # --- Baseline: vanilla CP-SAT ---
            time_b, branch_b, conf_b, status_b, _ = solve_subset_sum_cpsat(a, T)
            total_time_cpsat += time_b

            # --- Hybrid: LLL + CP-SAT ---
            time_h, branch_h, conf_h, status_h, s, lab, residual, hamming = solve_lll(
                a, T, sol
            )
            total_time_hybrid += time_h

            # --- Console output ---
            print_run_results(
                i,
                time_b,
                branch_b,
                conf_b,
                status_b,
                time_h,
                branch_h,
                conf_h,
                status_h,
                s,
            )

            # --- In-memory recording ---
            results[d]["time_base"].append(time_b)
            results[d]["branches_base"].append(branch_b)
            results[d]["conflicts_base"].append(conf_b)
            results[d]["success_base"].append(1 if status_b in SOLVED_STATUSES else 0)
            results[d]["time_hy"].append(time_h)
            results[d]["branches_hy"].append(branch_h)
            results[d]["conflicts_hy"].append(conf_h)
            results[d]["success_hy"].append(1 if status_h in SOLVED_STATUSES else 0)
            results[d]["residual"].append(residual)
            results[d]["hamming"].append(hamming)

            # --- Persistent CSV logging ---
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        d,
                        i,
                        time_b,
                        branch_b,
                        conf_b,
                        time_h,
                        branch_h,
                        conf_h,
                    ]
                )

    print(f"\nTotal CP-SAT time : {total_time_cpsat:.2f}s")
    print(f"Total Hybrid time : {total_time_hybrid:.2f}s")

    return results, total_time_cpsat, total_time_hybrid


# =============================================================================
# Statistics aggregation
# =============================================================================


def compute_stats(results: dict, densities: list[float]) -> dict:
    """
    Compute aggregate statistics for each density from benchmark results.

    Args:
        results:   Dict mapping density → collected metrics (from run_benchmark).
        densities: List of density values in the order they were benchmarked.

    Returns:
        Dict mapping density → {
            t_base_mean, t_base_std,
            t_hy_mean,   t_hy_std,
            b_base_mean, b_hy_mean,
            c_base_mean, c_hy_mean,
            speedup_mean, speedup_std,
            succ_base_mean, succ_hy_mean,
        }
    """
    stats = {}
    for d in densities:
        r = results[d]
        speedups = [
            t_b / max(t_h, 1e-5) for t_b, t_h in zip(r["time_base"], r["time_hy"])
        ]
        stats[d] = {
            "t_base_mean": np.mean(r["time_base"]),
            "t_base_std": np.std(r["time_base"]),
            "t_hy_mean": np.mean(r["time_hy"]),
            "t_hy_std": np.std(r["time_hy"]),
            "b_base_mean": np.mean(r["branches_base"]),
            "b_hy_mean": np.mean(r["branches_hy"]),
            "c_base_mean": np.mean(r["conflicts_base"]),
            "c_hy_mean": np.mean(r["conflicts_hy"]),
            "speedup_mean": np.mean(speedups),
            "speedup_std": np.std(speedups),
            "succ_base_mean": np.mean(r["success_base"]) * 100,
            "succ_hy_mean": np.mean(r["success_hy"]) * 100,
        }
    return stats
