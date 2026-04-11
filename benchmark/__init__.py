"""Benchmark toolkit for Subset-Sum solvers."""

from .io import RunRecord, save_records, load_records, records_to_df
from .runner import run_benchmark
from .visual import (
    plot_success_rates,
    plot_all_heatmaps,
    plot_heatmap_diff,
    generate_all_diffs_for_base,
    plot_performance,
    plot_lll_geometry,
    plot_all_time_heatmaps,          # <--- Nouveau
    plot_speedup_heatmap,            # <--- Nouveau
    generate_all_speedups_for_base,  # <--- Nouveau
)
from .style import get_style, SOLVER_STYLES

__all__ = [
    "RunRecord",
    "save_records",
    "load_records",
    "records_to_df",
    "run_benchmark",
    "plot_success_rates",
    "plot_all_heatmaps",
    "plot_heatmap_diff",
    "generate_all_diffs_for_base",
    "plot_performance",
    "plot_lll_geometry",
    "get_style",
    "SOLVER_STYLES",
    "plot_all_time_heatmaps",          # <--- Nouveau
    "plot_speedup_heatmap",            # <--- Nouveau
    "generate_all_speedups_for_base",  # <--- Nouveau
]