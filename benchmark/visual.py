"""Visualization: success rates, heatmaps, diffs, performance, LLL geometry."""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from .io import RunRecord, records_to_df
from .style import get_style


# ===================================================================
# Success-rate line plot
# ===================================================================

def plot_success_rates(records: list[RunRecord], output_path="results/success_plot.png"):
    df = records_to_df(records)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="n", y="success", hue="solver", marker="o")
    plt.title("Évolution de la robustesse en fonction de n")
    plt.ylabel("Taux de Succès")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graphique sauvegardé : {output_path}")


# ===================================================================
# Heatmaps
# ===================================================================

def plot_all_heatmaps(records: list[RunRecord], output_dir="results"):
    df = records_to_df(records)
    df["density"] = df["density"].round(2)
    solvers = df["solver"].unique()
    print(f"Génération de {len(solvers)} heatmaps...")

    for name in solvers:
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = f"{output_dir}/heatmap_{safe}.png"
        subset = df[df["solver"] == name]
        pivot = subset.groupby(["n", "density"])["success"].mean().unstack() * 100

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot, linewidths=0, annot=False, cmap="RdYlGn",
            vmin=0, vmax=100,
            yticklabels=max(1, len(pivot) // 8),
            xticklabels=max(1, len(pivot.columns) // 6),
            cbar_kws={"label": "Taux de Succès (%)"},
        )
        plt.title(f"Transition de Phase : {name}", pad=15)
        plt.xlabel("Densité $d$")
        plt.ylabel("Dimension $n$")
        plt.xticks(rotation=45)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  > [OK] {path}")


def plot_heatmap_diff(
    records: list[RunRecord],
    solver_base: str,
    solver_amelio: str,
    output_path="results/heatmap_diff.png",
):
    df = records_to_df(records)
    df["density"] = df["density"].round(2)

    pivot_b = df[df["solver"] == solver_base].groupby(["n", "density"])["success"].mean().unstack() * 100
    pivot_a = df[df["solver"] == solver_amelio].groupby(["n", "density"])["success"].mean().unstack() * 100
    diff = pivot_a.sub(pivot_b, fill_value=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        diff, annot=False, cmap="RdBu", center=0, vmin=-100, vmax=100,
        linewidths=0,
        yticklabels=max(1, len(diff) // 8),
        xticklabels=max(1, len(diff.columns) // 6),
        cbar_kws={"label": "Gain de Succès (%)"},
    )
    plt.title(f"Gain Absolu : {solver_amelio} vs {solver_base}", pad=20, fontsize=14)
    plt.xlabel("Densité $d$", fontsize=12)
    plt.ylabel("Dimension $n$", fontsize=12)
    plt.xticks(rotation=45)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  > [OK] Diff-Heatmap : {output_path}")


def generate_all_diffs_for_base(records: list[RunRecord], solver_base: str, output_dir="results"):
    df = records_to_df(records)
    solvers = df["solver"].unique()
    if solver_base not in solvers:
        print(f"[Attention] '{solver_base}' absent des résultats.")
        return

    targets = [s for s in solvers if s != solver_base]
    print(f"\n--- Diff-Heatmaps (base : {solver_base}) ---")
    for t in targets:
        safe_b = solver_base.replace(" ", "_").replace("(", "").replace(")", "").replace("->", "_to_")
        safe_t = t.replace(" ", "_").replace("(", "").replace(")", "").replace("->", "_to_")
        path = f"{output_dir}/diff_{safe_b}_vs_{safe_t}.png"
        try:
            plot_heatmap_diff(records, solver_base, t, output_path=path)
        except Exception as e:
            print(f"  > [Erreur] {solver_base} vs {t}: {e}")
    print(f"--- Terminé ---")


# ===================================================================
# Time heatmaps
# ===================================================================

def plot_all_time_heatmaps(records: list[RunRecord], output_dir="results"):
    """One heatmap per solver showing median solve time (log scale)."""
    os.makedirs(output_dir, exist_ok=True)

    df = records_to_df(records)
    df["density"] = df["density"].round(2)
    solvers = df["solver"].unique()
    print(f"Génération de {len(solvers)} time-heatmaps...")

    for name in solvers:
        safe = name.replace(" ", "_").replace("(", "").replace(")", "")
        path = f"{output_dir}/time_{safe}.png"
        subset = df[df["solver"] == name]
        pivot = subset.groupby(["n", "density"])["time"].median().unstack()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot, linewidths=0, annot=False, cmap="YlOrRd",
            norm=mcolors.LogNorm(vmin=max(pivot.min().min(), 1e-4), vmax=pivot.max().max()),
            yticklabels=max(1, len(pivot) // 8),
            xticklabels=max(1, len(pivot.columns) // 6),
            cbar_kws={"label": "Temps médian (s, log)"},
        )
        plt.title(f"Temps de résolution : {name}", pad=15)
        plt.xlabel("Densité $d$")
        plt.ylabel("Dimension $n$")
        plt.xticks(rotation=45)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  > [OK] {path}")


# ===================================================================
# Speedup heatmaps
# ===================================================================

def plot_speedup_heatmap(
    records: list[RunRecord],
    solver_base: str,
    solver_fast: str,
    output_path="results/speedup.png",
):
    """Heatmap of speedup = median_time(base) / median_time(fast), log-scale diverging at 1."""
    df = records_to_df(records)
    df["density"] = df["density"].round(2)

    t_base = df[df["solver"] == solver_base].groupby(["n", "density"])["time"].median().unstack()
    t_fast = df[df["solver"] == solver_fast].groupby(["n", "density"])["time"].median().unstack()

    # Avoid division by zero
    speedup = t_base / t_fast.replace(0, np.nan)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        speedup, annot=False, linewidths=0,
        cmap="RdBu",
        norm=mcolors.LogNorm(vmin=0.1, vmax=10),
        yticklabels=max(1, len(speedup) // 8),
        xticklabels=max(1, len(speedup.columns) // 6),
        cbar_kws={"label": "Speedup (log, >1 = plus rapide)"},
    )
    plt.title(f"Speedup : {solver_fast} vs {solver_base}", pad=20, fontsize=14)
    plt.xlabel("Densité $d$", fontsize=12)
    plt.ylabel("Dimension $n$", fontsize=12)
    plt.xticks(rotation=45)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  > [OK] Speedup-Heatmap : {output_path}")


def generate_all_speedups_for_base(records: list[RunRecord], solver_base: str, output_dir="results"):
    """Generate a speedup heatmap for every solver vs solver_base."""
    os.makedirs(output_dir, exist_ok=True)

    df = records_to_df(records)
    solvers = df["solver"].unique()
    if solver_base not in solvers:
        print(f"[Attention] '{solver_base}' absent des résultats.")
        return

    targets = [s for s in solvers if s != solver_base]
    print(f"\n--- Speedup-Heatmaps (base : {solver_base}) ---")
    for t in targets:
        safe_b = solver_base.replace(" ", "_").replace("(", "").replace(")", "").replace("->", "_to_")
        safe_t = t.replace(" ", "_").replace("(", "").replace(")", "").replace("->", "_to_")
        path = f"{output_dir}/speedup_{safe_b}_vs_{safe_t}.png"
        try:
            plot_speedup_heatmap(records, solver_base, t, output_path=path)
        except Exception as e:
            print(f"  > [Erreur] speedup {solver_base} vs {t}: {e}")
    print(f"--- Terminé ---")


# ===================================================================
# Multi-solver performance (4 panels)
# ===================================================================

def plot_performance(
    stats: dict,
    densities: list[float],
    n: int,
    save_path: str | None = None,
):
    """Success rate, solve time (log), speedup vs ref, branches (log)."""
    solver_names = list(stats[densities[0]].keys())
    ref = solver_names[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_succ, ax_time, ax_speed, ax_branch = axes.flatten()

    for name in solver_names:
        st = get_style(name)
        kw = dict(marker=st["marker"], color=st["color"], label=st["label"],
                   linewidth=1.5, markersize=5)

        ax_succ.plot(densities, [stats[d][name]["succ_pct"] for d in densities], **kw)
        ax_time.plot(densities, [max(stats[d][name]["t_mean"], 1e-5) for d in densities], **kw)
        ax_branch.plot(densities, [max(stats[d][name]["b_mean"], 1) for d in densities], **kw)

        if name != ref:
            speedups = [stats[d][ref]["t_mean"] / max(stats[d][name]["t_mean"], 1e-5) for d in densities]
            ax_speed.plot(densities, speedups, **kw)

    ax_succ.set(title="Taux de résolution", xlabel="Densité", ylabel="%")
    ax_succ.set_ylim(-5, 105)
    ax_time.set(title="Temps de résolution", xlabel="Densité", ylabel="Temps (s, log)")
    ax_time.set_yscale("log")
    ax_speed.axhline(1, color="gray", ls="--", lw=1, label="speedup=1")
    ax_speed.set(title="Facteur d'accélération", xlabel="Densité",
                 ylabel=f"Speedup vs {get_style(ref)['label']}")
    ax_speed.set_yscale("log")
    ax_branch.set(title="Branches CP-SAT", xlabel="Densité", ylabel="Branches (log)")
    ax_branch.set_yscale("log")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.axvline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)

    fig.suptitle(f"Benchmark solveurs — n={n}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()


# ===================================================================
# LLL geometry diagnostics (4 panels)
# ===================================================================

def plot_lll_geometry(
    records: list[RunRecord],
    densities: list[float],
    solver: str = "lll",
    save_path: str | None = None,
):
    """Residual vs density, branches, conflicts, Hamming distance."""

    def _mean_attr(d, attr):
        vals = [
            getattr(r.results[solver], attr)
            for r in records
            if r.density == d and solver in r.results and r.results[solver].best_res is not None
        ]
        return np.mean(vals) if vals else 0

    res = [max(_mean_attr(d, "best_res"), 1e-5) for d in densities]
    bra = [_mean_attr(d, "branches") for d in densities]
    con = [_mean_attr(d, "conflicts") for d in densities]
    ham = [_mean_attr(d, "best_ham") for d in densities]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_dr, ax_rb, ax_rc, ax_hr = axes.flatten()

    ax_dr.plot(densities, res, "o-", color="purple", lw=1.5)
    ax_dr.set(title="Densité vs Résidu LLL", xlabel="Densité", ylabel="Résidu moyen |R|")
    ax_dr.set_yscale("log")

    ax_rb.scatter(res, bra, color="blue", s=60, alpha=0.7)
    ax_rb.set(title="Résidu vs Branches", xlabel="Résidu |R|", ylabel="Branches")
    ax_rb.set_xscale("log")

    ax_rc.scatter(res, con, color="orange", s=60, alpha=0.7)
    ax_rc.set(title="Résidu vs Conflits", xlabel="Résidu |R|", ylabel="Conflits")
    ax_rc.set_xscale("log")

    ax_hr.scatter(ham, res, color="red", s=60, alpha=0.7)
    for d, h, r in zip(densities, ham, res):
        ax_hr.annotate(f"d={d:.2f}", (h, r), textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax_hr.set(title="Hamming vs Résidu", xlabel="Distance Hamming", ylabel="Résidu |R|")
    ax_hr.set_yscale("log")

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()