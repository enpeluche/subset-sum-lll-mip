"""Benchmark runner with live terminal progress display."""

import sys
import time
import inspect
import itertools
from functools import partial

from scipy.stats import fisher_exact, mannwhitneyu

from .io import RunRecord, AggRecord, aggregate_runs, save_agg_records


# =====================================================================
# ANSI colors (single source of truth)
# =====================================================================

class _C:
    """ANSI escape codes for terminal UI."""
    RST      = "\033[0m"
    BOLD     = "\033[1m"
    DIM      = "\033[2m"
    RED      = "\033[91m"
    GRN      = "\033[92m"
    YEL      = "\033[93m"
    BLU      = "\033[94m"
    GRY      = "\033[90m"
    BOLD_RED = "\033[1;91m"

    # Separators
    SEP_RED  = f"{BOLD_RED}|{RST}"
    SEP_GRY  = f"{DIM}|{RST}"


# Column widths
_WIDTHS = {"n": 3, "density": 7}
_SR_W = 6
_T_W = 10
_BAR_W = 24


# =====================================================================
# Formatting
# =====================================================================

def format_time_adaptive(seconds: float) -> str:
    """Format time dynamically as s, ms, µs or ns."""
    if seconds == 0.0:
        return "0 ns"
    elif seconds >= 1.0:
        return f"{seconds:.2f} s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.2f} ms"
    elif seconds >= 1e-6:
        return f"{seconds * 1e6:.2f} µs"
    else:
        return f"{seconds * 1e9:.0f} ns"


# =====================================================================
# Helpers
# =====================================================================

def _safe_call(func, **kwargs):
    """Call *func* forwarding only the kwargs it accepts (respects partial)."""
    fixed = func.keywords.keys() if isinstance(func, partial) else []
    sig = inspect.signature(func)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and k not in fixed}
    return func(**filtered)


def _fmt_params(cfg, axes):
    """Format parameter values for display (centered, gray separator)."""
    parts = []
    for ax in axes:
        v = cfg[ax]
        w = _WIDTHS.get(ax, 10)
        parts.append(f"{v:^{w}}" if isinstance(v, int) else f"{v:^{w}.2f}")
    return f" {_C.SEP_GRY} ".join(parts)


def _raw_params_width(axes):
    """Width of the params column without ANSI codes."""
    return len(" | ".join(f"{'':^{_WIDTHS.get(ax, 10)}}" for ax in axes))


# =====================================================================
# Header builder
# =====================================================================

def _build_header(solver_names, display_axes):
    """Build the three header lines (names, SR|Time, separator)."""
    pw = _raw_params_width(display_axes)

    # Line 1: solver names
    h1_parts = []
    for i, name in enumerate(solver_names):
        width = _SR_W + _T_W + 6 - (5 if i == 0 else 0)
        h1_parts.append(f"{_C.BOLD_RED}{name.upper():^{width}}{_C.RST}")
    h1 = f"{_C.SEP_RED} {'':^{pw}} {_C.SEP_RED} " + f" {_C.SEP_RED} ".join(h1_parts) + f" {_C.SEP_RED}"

    # Line 2: SR | Time sub-headers
    h_params = f" {_C.SEP_GRY} ".join(f"{ax:^{_WIDTHS.get(ax, 10)}}" for ax in display_axes)
    h2_parts, sep_parts = [], []
    for i in range(len(solver_names)):
        sw = _SR_W - 3 if i == 0 else _SR_W - 1
        tw = _T_W - 3 if i == 0 else _T_W
        h2_parts.append(f"{'SR':^{sw + 2}}{_C.SEP_GRY}{'Time':^{tw + 4}}")
        sep_parts.append(f"{'-' * (sw + 3)}{_C.SEP_GRY}{'-' * (tw + 5)}")

    h2 = f"{_C.SEP_RED} {h_params} {_C.SEP_RED} " + f" {_C.SEP_RED} ".join(h2_parts) + f" {_C.SEP_RED}"
    sep = f"{_C.SEP_RED}{'-' * (pw + 2)}{_C.SEP_RED}" + f"{_C.SEP_RED}".join(sep_parts) + f"{_C.SEP_RED}"

    return h1, h2, sep


# =====================================================================
# Public API
# =====================================================================

def run_benchmark(runs, solvers, json_file, timeout, generator, ranges: dict, workers: int):
    """Run all solvers over a parameter grid and persist aggregated results."""
    keys = list(ranges.keys())
    combos = list(itertools.product(*ranges.values()))
    solver_names = list(solvers.keys())

    records: list[RunRecord] = []
    agg_records: list[AggRecord] = []
    total = len(combos) * runs
    done = 0

    # Display config
    varying = [k for k, v in ranges.items() if len(v) > 1]
    display_axes = varying or ["n", "density"]
    fixes = [f"{k}={v[0]}" for k, v in ranges.items() if k not in varying]

    print("\nCONFIGURATION")
    print(f"Axes variables   : {', '.join(varying) if varying else 'Aucun (Point unique)'}")
    print(f"Paramètres fixes : {', '.join(fixes)}")
    print(f"Runs par point   : {runs} | Total points : {len(combos)}")

    h1, h2, sep = _build_header(solver_names, display_axes)
    t_start = time.monotonic()
    last_n = None

    for params in combos:
        cfg = dict(zip(keys, params))
        succ = {name: 0 for name in solver_names}
        times = {name: [] for name in solver_names}

        # Print header when n changes
        current_n = cfg.get("n")
        if current_n != last_n:
            print(f"\n{h1}\n{h2}\n{sep}")
            last_n = current_n

        for i in range(runs):
            instance = _safe_call(generator, **cfg)
            tmp = {}

            for name, solver in solvers.items():
                _progress_bar(done, total, t_start, cfg, display_axes)
                res = _safe_call(solver, instance=instance, workers=workers)
                tmp[name] = res
                if res.solution is not None:
                    succ[name] += 1
                    times[name].append(res.elapsed)

            records.append(RunRecord(
                density=cfg.get("density", 0.0),
                n=cfg.get("n", 0),
                run_idx=i,
                results=tmp,
            ))
            done += 1

        # Aggregate this (n, density) point and free raw records
        agg_records.extend(aggregate_runs(records, runs))
        records.clear()

        _freeze_row(cfg, display_axes, succ, times, solver_names, runs)

    save_agg_records(agg_records, json_file)
    return agg_records


# =====================================================================
# Terminal UI
# =====================================================================

def _progress_bar(done, total, t_start, cfg, axes):
    """Overwrite current line with a progress bar + ETA."""
    pct = done / total if total else 0
    filled = int(_BAR_W * pct)
    bar = "█" * filled + "░" * (_BAR_W - filled)

    elapsed = time.monotonic() - t_start
    eta_str = format_time_adaptive(elapsed / done * (total - done)) if done > 0 else "..."

    ps = _fmt_params(cfg, axes)
    line = (
        f"\r{_C.BLU}  {bar} {pct * 100:5.1f}%{_C.RST}"
        f"  {_C.GRY}{done}/{total}  ETA {eta_str}  [{ps}]{_C.RST}"
    )
    sys.stdout.write(f"{line}\033[K")
    sys.stdout.flush()


def _freeze_row(cfg, axes, succ, times, names, n_runs):
    """Print the final row for one (n, density) point with statistical significance."""
    ps = _fmt_params(cfg, axes)
    ref = names[0]
    t_ref = times[ref]
    avg_ref = sum(t_ref) / len(t_ref) if t_ref else 0.0

    parts = []
    for name in names:
        t_cur = times[name]
        has_data = len(t_cur) > 0
        avg = sum(t_cur) / len(t_cur) if t_cur else 0.0
        rate = succ[name] / n_runs
        is_ref = (name == ref)

        # --- Statistical tests vs reference ---
        sr_sym, sr_col = " ", _C.GRY
        if not is_ref:
            _, p = fisher_exact([
                [succ[ref], n_runs - succ[ref]],
                [succ[name], n_runs - succ[name]],
            ])
            if p < 0.001: #type: ignore
                sr_sym = ">>>" if succ[name] > succ[ref] else "<<<"
                sr_col = _C.GRN if succ[name] > succ[ref] else _C.RED
            elif p < 0.01: #type: ignore
                sr_sym = ">>" if succ[name] > succ[ref] else "<<"
                sr_col = _C.GRN if succ[name] > succ[ref] else _C.RED
            elif p < 0.05: #type: ignore
                sr_sym = ">" if succ[name] > succ[ref] else "<"
                sr_col = _C.GRN if succ[name] > succ[ref] else _C.RED
            elif p < 0.15: #type: ignore
                sr_sym = ">=" if succ[name] > succ[ref] else "<="
                sr_col = _C.GRN if succ[name] > succ[ref] else _C.RED
            

        t_sym, t_col = " ", _C.GRY
        if not is_ref and t_ref and t_cur:
            try:
                if t_ref != t_cur:
                    res = mannwhitneyu(t_ref, t_cur)
                    if res.pvalue < 0.001:
                        t_sym = ">>>" if avg < avg_ref else "<<<"
                        t_col = _C.GRN if avg < avg_ref else _C.RED
                    elif res.pvalue < 0.01:
                        t_sym = ">>" if avg < avg_ref else "<<"
                        t_col = _C.GRN if avg < avg_ref else _C.RED
                    elif  res.pvalue < 0.05:
                        t_sym = ">" if avg < avg_ref else "<"
                        t_col = _C.GRN if avg < avg_ref else _C.RED
                    elif  res.pvalue < 0.05:
                        t_sym = ">=" if avg < avg_ref else "<="
                        t_col = _C.GRN if avg < avg_ref else _C.RED
            except Exception:
                pass

        # --- Format cell ---
        txt_col = _C.GRN if rate == 1.0 else (_C.YEL if rate > 0 else _C.GRY)
        t_label = format_time_adaptive(avg) if has_data else f"{'-':^{_T_W}}"

        if is_ref:
            cell = (f" {txt_col}{rate * 100:>3.0f}%{_C.RST}"
                    f" {_C.SEP_GRY} "
                    f"{txt_col}{t_label:>{_T_W}}{_C.RST} ")
        else:
            cell = (f" {txt_col}{rate * 100:>3.0f}%{_C.RST}"
                    f"{'':>{_SR_W - 5}}{sr_col}{sr_sym}{_C.RST}"
                    f" {_C.SEP_GRY} "
                    f"{txt_col}{t_label:>{_T_W}}{_C.RST}"
                    f"  {t_col}{t_sym}{_C.RST} ")

        parts.append(cell)

    line = f"\r{_C.SEP_RED} {ps} {_C.SEP_RED}" + _C.SEP_RED.join(parts) + f"{_C.SEP_RED}\033[K\n"
    sys.stdout.write(line)
    sys.stdout.flush()