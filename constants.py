"""
Global configuration constants for the LLL+CP-SAT hybrid benchmark.

All tuneable parameters are defined here so they can be changed in one place
without hunting through the codebase.
"""

from SubsetSumInstance import SubsetSumInstance

# =============================================================================
# Solver configuration
# =============================================================================

# Maximum wall-clock time (in seconds) allocated per solve call.
# Applies to both the vanilla CP-SAT baseline and the hybrid fallback.
TIMEOUT: float = 5.0

# Time budget (in seconds) allocated to each individual hint attempt
# in the hybrid solver. Kept short: a good hint converges in milliseconds;
# a bad hint should be abandoned quickly to preserve fallback budget.
MICRO_TIMEOUT: float = 2.0

# =============================================================================
# Benchmark configuration
# =============================================================================

# Instance dimension (number of elements in the subset sum problem)
N: int = 30

# Number of runs per density value
N_RUNS: int = 100

# Density grid: starts just above the minimum safe density for N,
# then increments by 0.05 up to 40 steps.
# The minimum density ensures weights stay within CP-SAT's 64-bit integer limit.
DENSITIES: list[float] = [
    SubsetSumInstance.get_min_safe_density(N) + 0.025 * i for i in range(80)
]

# =============================================================================
# Output configuration
# =============================================================================

# Path to the CSV file for persistent logging of raw benchmark results.
# If the file already exists, new rows are appended (no overwrite).
CSV_FILE: str = "benchmark_results.csv"

# Path prefix for saved figures.
# plot_performance saves to f"{FIGURE_PREFIX}_0.png"
# plot_lll_geometry saves to f"{FIGURE_PREFIX}_1.png"
FIGURE_PREFIX: str = f"lll_{N}"
