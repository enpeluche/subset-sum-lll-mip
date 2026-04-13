"""Benchmark toolkit for Subset-Sum solvers."""

from .io import RunRecord, AggRecord, save_records, load_records, records_to_df, agg_records_to_df
from .runner import run_benchmark
from .experiment import Benchmark
from .visual import (
    plot_success_rates,
    generate_speedup_mosaic,
    generate_success_mosaic,
    generate_time_mosaic,
    plot_all_success_rate_heatmaps_mosaic,
    plot_performance,
    plot_lll_geometry,
)
from .style import get_style, SOLVER_STYLES

__all__ = [
    "Benchmark",
    "RunRecord",
    "AggRecord",
    "save_records",
    "load_records",
    "records_to_df",
    "agg_records_to_df",
    "run_benchmark",
    "plot_success_rates",
    "generate_success_mosaic",
    "generate_time_mosaic",
    "generate_speedup_mosaic",
    "plot_all_success_rate_heatmaps_mosaic",
    "plot_performance",
    "plot_lll_geometry",
    "get_style",
    "SOLVER_STYLES",
]