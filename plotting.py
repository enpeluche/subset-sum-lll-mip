"""
Visualization utilities for the LLL+CP-SAT hybrid benchmark.

Two main functions:
    - plot_performance:    solve time, branches, conflicts, speedup, success rate
    - plot_lll_geometry:   residual vs density, residual vs search tree, Hamming precision
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_performance(
    stats: dict,
    densities: list,
    total_time_cpsat: float,
    total_time_hybrid: float,
    save_path: str | None = None,
) -> None:
    """
    Plot the five performance metrics comparing CP-SAT baseline vs hybrid solver.

    Panels:
        [0,0] Solve time (mean)
        [0,1] Number of branches explored
        [0,2] Number of conflicts (backtracks)
        [1,0] Hybrid speedup factor
        [1,1] Success rate before timeout

    Args:
        stats:              Output of compute_stats().
        densities:          Ordered list of density values.
        total_time_cpsat:   Cumulative wall-clock time for CP-SAT baseline.
        total_time_hybrid:  Cumulative wall-clock time for hybrid solver.
        save_path:          If provided, save the figure to this path instead
                            of displaying it interactively.
    """
    dens = densities
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ax_time, ax_branch, ax_conf = axes[0]
    ax_speed, ax_succ, ax_empty = axes[1]
    fig.delaxes(ax_empty)

    # --- Solve time ---
    ax_time.plot(
        dens,
        [stats[d]["t_base_mean"] for d in dens],
        "-o",
        markersize=3,
        linewidth=1.2,
        color="blue",
        alpha=0.7,
        label=f"CP-SAT (Base) - Total: {total_time_cpsat:.2f}s",
    )
    ax_time.plot(
        dens,
        [stats[d]["t_hy_mean"] for d in dens],
        "-s",
        markersize=3,
        linewidth=1.2,
        color="red",
        alpha=0.8,
        label=f"Hybride - Total: {total_time_hybrid:.2f}s",
    )
    ax_time.set_ylim(bottom=0)
    ax_time.set_xlabel("Density")
    ax_time.set_ylabel("Solve time (s)")
    ax_time.set_title("Temps de résolution (Moyenne)")
    ax_time.grid(axis="y", alpha=0.3)
    ax_time.legend()

    # --- Branches ---
    ax_branch.plot(
        dens,
        [stats[d]["b_base_mean"] for d in dens],
        "-o",
        markersize=2,
        linewidth=0.8,
        color="blue",
        alpha=0.6,
        label="CP-SAT (Base)",
    )
    ax_branch.plot(
        dens,
        [stats[d]["b_hy_mean"] for d in dens],
        "-s",
        markersize=2,
        linewidth=0.8,
        color="red",
        alpha=0.8,
        label="Hybride",
    )
    ax_branch.set_ylim(bottom=0)
    ax_branch.set_xlabel("Density")
    ax_branch.set_ylabel("Branches")
    ax_branch.set_title("Nombre de branches explorées")
    ax_branch.grid(axis="y", alpha=0.5)
    ax_branch.legend()

    # --- Conflicts ---
    ax_conf.plot(
        dens,
        [stats[d]["c_base_mean"] for d in dens],
        "-o",
        markersize=2,
        linewidth=0.8,
        color="blue",
        alpha=0.6,
        label="CP-SAT (Base)",
    )
    ax_conf.plot(
        dens,
        [stats[d]["c_hy_mean"] for d in dens],
        "-s",
        markersize=2,
        linewidth=0.8,
        color="red",
        alpha=0.8,
        label="Hybride",
    )
    ax_conf.set_ylim(bottom=0)
    ax_conf.set_xlabel("Density")
    ax_conf.set_ylabel("Conflicts")
    ax_conf.set_title("Nombre de conflits (Backtracks)")
    ax_conf.grid(axis="y", alpha=0.5)
    ax_conf.legend()

    # --- Speedup ---
    ax_speed.plot(
        dens,
        [stats[d]["speedup_mean"] for d in dens],
        "-^",
        markersize=5,
        linewidth=1.8,
        color="green",
        alpha=0.9,
        label="Facteur d'accélération (Moyen)",
    )
    ax_speed.axhline(
        y=1,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label="Référence (x1 - Égalité)",
    )
    ax_speed.set_ylim(bottom=0)
    ax_speed.set_xlabel("Density")
    ax_speed.set_ylabel("Speedup (Base Time / Hybrid Time)")
    ax_speed.set_title("Accélération de l'Hybride (Speedup)")
    ax_speed.grid(axis="y", alpha=0.3)
    ax_speed.legend()

    # --- Success rate ---
    ax_succ.plot(
        dens,
        [stats[d]["succ_base_mean"] for d in dens],
        "-o",
        markersize=4,
        linewidth=1.2,
        color="blue",
        alpha=0.6,
        label="CP-SAT (Base)",
    )
    ax_succ.plot(
        dens,
        [stats[d]["succ_hy_mean"] for d in dens],
        "-s",
        markersize=4,
        linewidth=1.2,
        color="red",
        alpha=0.8,
        label="Hybride (LLL+CP-SAT)",
    )
    ax_succ.set_ylim(-5, 105)
    ax_succ.set_xlabel("Density")
    ax_succ.set_ylabel("Success Rate (%)")
    ax_succ.set_title("Taux de résolution avant Timeout")
    ax_succ.grid(axis="y", alpha=0.5)
    ax_succ.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def plot_lll_geometry(
    results: dict,
    densities: list,
    save_path: str | None = None,
) -> None:
    """
    Plot four LLL geometry diagnostics.

    Panels:
        [0,0] Density vs mean LLL residual
        [0,1] Residual vs CP-SAT branches (correlation)
        [1,0] Residual vs CP-SAT conflicts (correlation)
        [1,1] Hamming distance to solution vs residual (hint precision)

    Args:
        results:   Dict mapping density → collected metrics (from run_benchmark).
        densities: Ordered list of density values.
        save_path: If provided, save the figure instead of displaying it.
    """
    dens = densities

    def mean(data):
        return np.mean(data) if isinstance(data, list) else data

    res_vals = [max(mean(results[d]["residual"]), 1e-5) for d in dens]
    branch_vals = [mean(results[d]["branches_hy"]) for d in dens]
    conf_vals = [mean(results[d]["conflicts_hy"]) for d in dens]
    ham_vals = [mean(results[d]["hamming"]) for d in dens]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_dens_res, ax_res_branch, ax_res_conf, ax_ham_res = axes.flatten()

    # --- Density vs residual ---
    ax_dens_res.plot(dens, res_vals, marker="o", color="purple", linewidth=1.5)
    ax_dens_res.set_xlabel("Densité")
    ax_dens_res.set_ylabel("Résidu moyen |R|")
    ax_dens_res.set_title("Densité vs Résidu LLL")
    ax_dens_res.set_yscale("log")
    ax_dens_res.grid(True, alpha=0.5)

    # --- Residual vs branches ---
    ax_res_branch.scatter(res_vals, branch_vals, color="blue", s=60, alpha=0.7)
    ax_res_branch.set_ylim(bottom=0)
    ax_res_branch.set_xlabel("Résidu |R| (Échelle Log)")
    ax_res_branch.set_ylabel("Branches explorées par l'Hybride")
    ax_res_branch.set_title("Corrélation : Qualité géométrique vs Arbre de recherche")
    ax_res_branch.set_xscale("log")
    ax_res_branch.grid(True, alpha=0.5)

    # --- Residual vs conflicts ---
    ax_res_conf.scatter(res_vals, conf_vals, color="orange", s=60, alpha=0.7)
    ax_res_conf.set_ylim(bottom=0)
    ax_res_conf.set_xlabel("Résidu |R| (Échelle Log)")
    ax_res_conf.set_ylabel("Conflits générés par l'Hybride")
    ax_res_conf.set_title("Corrélation : Résidu vs Backtracks (Conflits)")
    ax_res_conf.set_xscale("log")
    ax_res_conf.grid(True, alpha=0.5)

    # --- Hamming vs residual ---
    ax_ham_res.scatter(ham_vals, res_vals, color="red", s=60, alpha=0.7)
    for d_val, ham, res in zip(dens, ham_vals, res_vals):
        ax_ham_res.annotate(
            f"d={d_val:.2f}",
            (ham, res),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            alpha=0.8,
        )
    ax_ham_res.set_xlabel("Distance de Hamming à la solution exacte")
    ax_ham_res.set_ylabel("Résidu |R| (Échelle Log)")
    ax_ham_res.set_title("Précision des Hints vs Résidu LLL")
    ax_ham_res.set_yscale("log")
    ax_ham_res.grid(True, alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
