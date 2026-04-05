# main.py
import csv
import os

from constants import N, N_RUNS, DENSITIES, CSV_FILE, FIGURE_PREFIX
from benchmark import run_benchmark, compute_stats
from plotting import plot_performance, plot_lll_geometry

# --- CSV initialization ---
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "density",
                "run_id",
                "time_base",
                "branches_base",
                "conflicts_base",
                "time_hy",
                "branches_hy",
                "conflicts_hy",
            ]
        )

# --- Run benchmark ---
results, total_time_cpsat, total_time_hybrid = run_benchmark(
    n=N,
    densities=DENSITIES,
    n_runs=N_RUNS,
    csv_file=CSV_FILE,
)

# --- Compute statistics ---
stats = compute_stats(results, DENSITIES)

# --- Plot ---
plot_performance(
    stats,
    DENSITIES,
    total_time_cpsat=total_time_cpsat,
    total_time_hybrid=total_time_hybrid,
    save_path=f"{FIGURE_PREFIX}_0.png",
)

plot_lll_geometry(
    results,
    DENSITIES,
    save_path=f"{FIGURE_PREFIX}_1.png",
)
