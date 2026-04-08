import numpy as np
import matplotlib.pyplot as plt
from results import RunRecord


def plot_lll_geometry(
    records: list[RunRecord],
    densities: list[float],
    solver: str = "lll",
    save_path: str | None = None,
) -> None:
    """
    Plot LLL geometry diagnostics for a given solver.

    Panels:
        [0,0] Densité vs résidu moyen
        [0,1] Résidu vs branches
        [1,0] Résidu vs conflits
        [1,1] Distance de Hamming vs résidu

    Args:
        records:   List of RunRecord from run_benchmark.
        densities: Ordered list of density values.
        solver:    Solver name to extract geometry metrics from.
        save_path: If provided, save figure to this path.
    """

    def mean_attr(d, attr):
        vals = [
            getattr(r.results[solver], attr)
            for r in records
            if r.density == d and r.results[solver].best_res is not None
        ]
        return np.mean(vals) if vals else 0

    res_vals = [max(mean_attr(d, "best_res"), 1e-5) for d in densities]
    branch_vals = [mean_attr(d, "branches") for d in densities]
    conf_vals = [mean_attr(d, "conflicts") for d in densities]
    ham_vals = [mean_attr(d, "best_ham") for d in densities]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_dr, ax_rb, ax_rc, ax_hr = axes.flatten()

    ax_dr.plot(densities, res_vals, "o-", color="purple", linewidth=1.5)
    ax_dr.set(
        title="Densité vs Résidu LLL", xlabel="Densité", ylabel="Résidu moyen |R|"
    )
    ax_dr.set_yscale("log")

    ax_rb.scatter(res_vals, branch_vals, color="blue", s=60, alpha=0.7)
    ax_rb.set(title="Résidu vs Branches", xlabel="Résidu |R|", ylabel="Branches")
    ax_rb.set_xscale("log")

    ax_rc.scatter(res_vals, conf_vals, color="orange", s=60, alpha=0.7)
    ax_rc.set(title="Résidu vs Conflits", xlabel="Résidu |R|", ylabel="Conflits")
    ax_rc.set_xscale("log")

    ax_hr.scatter(ham_vals, res_vals, color="red", s=60, alpha=0.7)
    for d, ham, res in zip(densities, ham_vals, res_vals):
        ax_hr.annotate(
            f"d={d:.2f}",
            (ham, res),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
        )
    ax_hr.set(title="Hamming vs Résidu", xlabel="Distance Hamming", ylabel="Résidu |R|")
    ax_hr.set_yscale("log")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
