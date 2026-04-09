import numpy as np
import matplotlib.pyplot as plt
from .style import get_style


def plot_performance(
    stats: dict,
    densities: list[float],
    n: int,
    save_path: str | None = None,
) -> None:
    """
    Plot performance metrics for all solvers in stats.

    Panels:
        [0,0] Taux de résolution
        [0,1] Temps de résolution (log)
        [1,0] Facteur d'accélération vs premier solveur
        [1,1] Branches explorées (log)

    Args:
        stats:     Output of compute_stats() — stats[density][solver_name].
        densities: Ordered list of density values.
        save_path: If provided, save figure to this path.
    """
    solver_names = list(stats[densities[0]].keys())
    ref = solver_names[0]  # référence pour le speedup

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_succ, ax_time, ax_speed, ax_branch = axes.flatten()

    for name in solver_names:
        style = get_style(name)
        kwargs = dict(
            marker=style["marker"],
            color=style["color"],
            label=style["label"],
            linewidth=1.5,
            markersize=5,
        )

        ax_succ.plot(
            densities, [stats[d][name]["succ_pct"] for d in densities], **kwargs
        )

        ax_time.plot(
            densities,
            [max(stats[d][name]["t_mean"], 1e-5) for d in densities],
            **kwargs,
        )

        ax_branch.plot(
            densities, [max(stats[d][name]["b_mean"], 1) for d in densities], **kwargs
        )

        if name != ref:
            speedups = [
                stats[d][ref]["t_mean"] / max(stats[d][name]["t_mean"], 1e-5)
                for d in densities
            ]
            ax_speed.plot(densities, speedups, **kwargs)

    # Axes config
    ax_succ.set(title="Taux de résolution", xlabel="Densité", ylabel="%")
    ax_succ.set_ylim(-5, 105)

    ax_time.set(title="Temps de résolution", xlabel="Densité", ylabel="Temps (s, log)")
    ax_time.set_yscale("log")

    ax_speed.axhline(y=1, color="gray", linestyle="--", linewidth=1, label="speedup=1")
    ax_speed.set(
        title="Facteur d'accélération",
        xlabel="Densité",
        ylabel=f"Speedup vs {get_style(ref)['label']}",
    )
    ax_speed.set_yscale("log")

    ax_branch.set(
        title="Branches explorées par CP-SAT", xlabel="Densité", ylabel="Branches (log)"
    )
    ax_branch.set_yscale("log")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.suptitle(
        f"CP-SAT vs LLL+CP-SAT vs BKZ(30)+CP-SAT — n={n}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


def _infer_n(stats: dict, densities: list) -> str:
    """Best-effort extraction of n from stats for the title."""
    try:
        return str(stats[densities[0]].get("n", "?"))
    except Exception:
        return "?"
