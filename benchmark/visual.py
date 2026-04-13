"""Visualization: heatmap mosaics for success, time, speedup, and diffs."""

import copy
import math
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .io import AggRecord, RunRecord, agg_records_to_df
from .style import get_style


# =====================================================================
# Helpers
# =====================================================================

def _make_grid(n_panels: int, max_cols: int = 3, panel_size: tuple = (8, 6)):
    """Create a figure with a grid of subplots, return (fig, axes_list)."""
    cols = min(max_cols, n_panels)
    rows = math.ceil(n_panels / cols)
    fig, raw_axes = plt.subplots(rows, cols, figsize=(cols * panel_size[0], rows * panel_size[1]))

    if isinstance(raw_axes, np.ndarray):
        axes_list = raw_axes.flatten().tolist()
    else:
        axes_list = [raw_axes]

    return fig, axes_list


def _cleanup_and_save(fig, axes_list, n_used: int, cmap, norm, cbar_label: str, output_path: str):
    """Hide unused axes, add global colorbar, save figure."""
    for j in range(n_used, len(axes_list)):
        axes_list[j].axis("off")

    fig.tight_layout(rect=(0, 0, 0.9, 1))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    fig.colorbar(sm, cax=cax, label=cbar_label)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  > [OK] {output_path}")


def _prepare_df(agg_records: list[AggRecord]) -> pd.DataFrame:
    """Convert to DataFrame with rounded density."""
    df = agg_records_to_df(agg_records)
    df["density"] = df["density"].round(2)
    return df


def _tick_skip(pivot, axis: str = "both") -> dict:
    """Compute yticklabels/xticklabels skip for readability."""
    kw = {}
    if axis in ("both", "y"):
        kw["yticklabels"] = max(1, len(pivot) // 8)
    if axis in ("both", "x"):
        kw["xticklabels"] = max(1, len(pivot.columns) // 6)
    return kw


# =====================================================================
# 1D plot (when only one axis varies)
# =====================================================================

def plot_success_rates(agg_records: list[AggRecord], output_path="results/success_plot.png"):
    """Line plot of success rate vs n (for single-density sweeps)."""
    df = agg_records_to_df(agg_records)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="n", y="success", hue="solver", marker="o")
    plt.title("Taux de succès en fonction de n")
    plt.ylabel("Taux de Succès (%)")
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  > [OK] {output_path}")


# =====================================================================
# Success rate heatmaps
# =====================================================================

def generate_success_mosaic(agg_records: list[AggRecord], output_dir="results"):
    """One heatmap per solver: absolute success rate (0-100%)."""
    os.makedirs(output_dir, exist_ok=True)
    df = _prepare_df(agg_records)
    solvers = df["solver"].unique()
    n_solvers = len(solvers)
    if n_solvers == 0:
        return

    fig, axes = _make_grid(n_solvers)
    cmap = copy.copy(plt.get_cmap("RdYlGn"))
    cmap.set_bad(color="#333333")

    print(f"\n--- Mosaic: Success Rates ---")
    for i, name in enumerate(solvers):
        pivot = df[df["solver"] == name].groupby(["n", "density"])["success"].mean().unstack()
        sns.heatmap(
            pivot, annot=False, linewidths=0, cmap=cmap,
            vmin=0, vmax=100, cbar=False, ax=axes[i],
            **_tick_skip(pivot),
        )
        axes[i].set_title(name, pad=15, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Densité $d$")
        axes[i].set_ylabel("Dimension $n$")
        axes[i].tick_params(axis="x", rotation=45)

    _cleanup_and_save(
        fig, axes, n_solvers, cmap,
        mcolors.Normalize(vmin=0, vmax=100),
        "Taux de Succès (%)",
        f"{output_dir}/mosaic_success_all.png",
    )


# =====================================================================
# Success rate diff heatmaps (vs reference)
# =====================================================================

def plot_all_success_rate_heatmaps_mosaic(
    agg_records: list[AggRecord], solver_base: str, output_dir="results",
):
    """One heatmap per solver: success rate gain vs reference (-100 to +100%)."""
    os.makedirs(output_dir, exist_ok=True)
    df = _prepare_df(agg_records)
    solvers = df["solver"].unique()

    if solver_base not in solvers:
        print(f"[Warning] '{solver_base}' not in results.")
        return

    targets = [s for s in solvers if s != solver_base]
    if not targets:
        return

    fig, axes = _make_grid(len(targets))
    pivot_base = df[df["solver"] == solver_base].groupby(["n", "density"])["success"].mean().unstack()

    print(f"\n--- Mosaic: Success Gain vs {solver_base} ---")
    for i, name in enumerate(targets):
        pivot_t = df[df["solver"] == name].groupby(["n", "density"])["success"].mean().unstack()
        diff = pivot_t.sub(pivot_base, fill_value=0)

        sns.heatmap(
            diff, annot=False, linewidths=0, cmap="RdBu",
            center=0, vmin=-100, vmax=100, cbar=False, ax=axes[i],
            **_tick_skip(diff),
        )
        axes[i].set_title(f"{name} vs {solver_base}", pad=15, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Densité $d$")
        axes[i].set_ylabel("Dimension $n$")
        axes[i].tick_params(axis="x", rotation=45)

    _cleanup_and_save(
        fig, axes, len(targets), "RdBu",
        mcolors.Normalize(vmin=-100, vmax=100),
        "Gain de Succès (%)",
        f"{output_dir}/SR_heatmaps_vs_{solver_base}.png",
    )


# =====================================================================
# Time heatmaps
# =====================================================================

def generate_time_mosaic(agg_records: list[AggRecord], output_dir="results", timeout_val=10):
    """One heatmap per solver: median solve time (log scale)."""
    os.makedirs(output_dir, exist_ok=True)
    df = _prepare_df(agg_records)
    solvers = df["solver"].unique()
    n_solvers = len(solvers)
    if n_solvers == 0:
        return

    # Global log scale bounds
    valid_times = df["time"][df["time"] > 0]
    global_min = max(valid_times.min(), 1e-4) if not valid_times.empty else 1e-4
    global_max = timeout_val
    norm = mcolors.LogNorm(vmin=global_min, vmax=global_max)

    fig, axes = _make_grid(n_solvers)

    print(f"\n--- Mosaic: Solve Time (log) ---")
    for i, name in enumerate(solvers):
        pivot = df[df["solver"] == name].groupby(["n", "density"])["time"].median().unstack()
        pivot = pivot.fillna(timeout_val)

        sns.heatmap(
            pivot, annot=False, linewidths=0, cmap="YlOrRd",
            norm=norm, cbar=False, ax=axes[i],
            **_tick_skip(pivot),
        )
        axes[i].set_title(name, pad=15, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Densité $d$")
        axes[i].set_ylabel("Dimension $n$")
        axes[i].tick_params(axis="x", rotation=45)

    _cleanup_and_save(
        fig, axes, n_solvers, "YlOrRd", norm,
        "Temps médian (s, log)",
        f"{output_dir}/mosaic_time_all.png",
    )


# =====================================================================
# Speedup heatmaps (vs reference)
# =====================================================================

def generate_speedup_mosaic(
    agg_records: list[AggRecord], solver_base: str, output_dir="results", timeout_val=10,
):
    """One heatmap per solver: speedup vs reference (log scale, >1 = faster)."""
    os.makedirs(output_dir, exist_ok=True)
    df = _prepare_df(agg_records)
    solvers = df["solver"].unique()

    if solver_base not in solvers:
        print(f"[Warning] '{solver_base}' not in results.")
        return

    targets = [s for s in solvers if s != solver_base]
    if not targets:
        return

    fig, axes = _make_grid(len(targets))
    norm = mcolors.LogNorm(vmin=0.1, vmax=10)
    t_base = df[df["solver"] == solver_base].groupby(["n", "density"])["time"].median().unstack().fillna(timeout_val)

    print(f"\n--- Mosaic: Speedup vs {solver_base} ---")
    for i, name in enumerate(targets):
        t_fast = df[df["solver"] == name].groupby(["n", "density"])["time"].median().unstack().fillna(timeout_val)
        speedup = t_base / t_fast.replace(0, np.nan)

        if speedup.isna().all().all():
            axes[i].set_title(f"{name} (no data)")
            axes[i].axis("off")
            continue

        sns.heatmap(
            speedup, annot=False, linewidths=0, cmap="RdBu",
            norm=norm, cbar=False, ax=axes[i],
            **_tick_skip(speedup),
        )
        axes[i].set_title(f"{name} vs {solver_base}", pad=15, fontsize=13, fontweight="bold")
        axes[i].set_xlabel("Densité $d$")
        axes[i].set_ylabel("Dimension $n$")
        axes[i].tick_params(axis="x", rotation=45)

    safe_base = solver_base.replace(" ", "_").replace("(", "").replace(")", "")
    _cleanup_and_save(
        fig, axes, len(targets), "RdBu", norm,
        "Speedup (log, >1 = plus rapide)",
        f"{output_dir}/mosaic_speedup_vs_{safe_base}.png",
    )


# =====================================================================
# Multi-solver performance (4 panels, for 1D density sweeps)
# =====================================================================

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


# =====================================================================
# LLL geometry diagnostics (4 panels)
# =====================================================================

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