"""Benchmark runner with live terminal progress display."""

import sys
import time
import inspect
import itertools
from functools import partial

from scipy.stats import fisher_exact, mannwhitneyu

from .io import RunRecord, save_records


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_call(func, **kwargs):
    """Call *func* forwarding only the kwargs it accepts (respects partial)."""
    fixed = func.keywords.keys() if isinstance(func, partial) else []
    sig = inspect.signature(func)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and k not in fixed}
    return func(**filtered)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_benchmark(runs, solvers, json_file, timeout, generator, ranges: dict):
    """Run all solvers over a parameter grid and persist results."""
    keys = list(ranges.keys())
    combos = list(itertools.product(*ranges.values()))
    solver_names = list(solvers.keys())

    records: list[RunRecord] = []
    total = len(combos) * runs
    done = 0

    # Display config
    COL_W, SR_W, T_W = 10, 6, 10
    varying = [k for k, v in ranges.items() if len(v) > 1]
    display_axes = varying or ["n", "density"]
    fixes = [f"{k}={v[0]}" for k, v in ranges.items() if k not in varying]

    print("\nCONFIGURATION")
    print(f"Axes variables   : {', '.join(varying) if varying else 'Aucun (Point unique)'}")
    print(f"Paramètres fixes : {', '.join(fixes)}")
    print(f"Runs par point   : {runs} | Total points : {len(combos)}")

    # Header
    h_params = " | ".join(f"{ax:<{COL_W - 1}}" for ax in display_axes)
    pw = len(h_params)
    h1 = f"| {'':<{pw}} | " + " | ".join(f"{n.upper():^{SR_W + T_W + 3}}" for n in solver_names) + " |"
    h2 = f"| {h_params} | " + " | ".join(f"{'SR':<{SR_W}} | {'Time':<{T_W}}" for _ in solver_names) + " |"
    sep = f"|{'-' * (pw + 2)}|" + "|".join(f"{'-' * (SR_W + 2)}|{'-' * (T_W + 2)}" for _ in solver_names) + "|"
    print(f"\n{h1}\n{h2}\n{sep}")

    t_start = time.monotonic()

    for params in combos:
        cfg = dict(zip(keys, params))
        succ = {n: 0 for n in solver_names}
        times = {n: [] for n in solver_names}

        for i in range(runs):
            instance = _safe_call(generator, **cfg)
            tmp = {}

            for name, solver in solvers.items():
                _progress_bar(done, total, t_start, cfg, display_axes, COL_W)
                res = solver(instance)
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

        _freeze_row(cfg, display_axes, succ, times, solver_names, runs, SR_W, T_W, COL_W)

    save_records(records, json_file)
    return records


# ---------------------------------------------------------------------------
# Terminal UI helpers
# ---------------------------------------------------------------------------

BAR_WIDTH = 24
BAR_FILL = "█"
BAR_EMPTY = "░"


def _fmt_params(cfg, axes, w):
    parts = []
    for ax in axes:
        v = cfg[ax]
        parts.append(f"{v:<{w - 1}}" if isinstance(v, int) else f"{v:<{w - 1}.2f}")
    return " | ".join(parts)


def _progress_bar(done, total, t_start, cfg, axes, col_w):
    """Overwrite current line with a progress bar + ETA."""
    pct = done / total if total else 0
    filled = int(BAR_WIDTH * pct)
    bar = BAR_FILL * filled + BAR_EMPTY * (BAR_WIDTH - filled)

    elapsed = time.monotonic() - t_start
    if done > 0:
        eta = elapsed / done * (total - done)
        eta_str = format_time_adaptive(eta)
    else:
        eta_str = "..."

    ps = _fmt_params(cfg, axes, col_w)
    line = (
        f"\r\033[94m  {bar} {pct * 100:5.1f}%\033[0m"
        f"  \033[90m{done}/{total}  ETA {eta_str}  [{ps}]\033[0m"
    )
    sys.stdout.write(f"{line}\033[K")
    sys.stdout.flush()


def _freeze_row(cfg, axes, succ, times, names, n_runs, sr_w, t_w, col_w):
    ps = _fmt_params(cfg, axes, col_w)
    ref = names[0]
    t_ref = times[ref]
    avg_ref = sum(t_ref) / len(t_ref) if t_ref else 0.0

    RST, GRY, GRN, RED, YEL = "\033[0m", "\033[90m", "\033[92m", "\033[91m", "\033[93m"

    parts = []
    for name in names:
        t_cur = times[name]
        has_data = len(t_cur) > 0
        avg = sum(t_cur) / len(t_cur) if t_cur else 0.0
        rate = succ[name] / n_runs

        sr_sym, sr_col = " ", GRY
        if name != ref:
            _, p = fisher_exact([[succ[ref], n_runs - succ[ref]],
                                 [succ[name], n_runs - succ[name]]])
            if p < 0.05:  # type: ignore
                sr_sym = "+" if succ[name] > succ[ref] else "-"
                sr_col = GRN if succ[name] > succ[ref] else RED

        t_sym, t_col = " ", GRY
        if name != ref and t_ref and t_cur:
            try:
                if t_ref != t_cur:
                    res = mannwhitneyu(t_ref, t_cur)
                    if res.pvalue < 0.05:
                        t_sym = "*"
                        t_col = GRN if avg < avg_ref else RED
            except Exception:
                pass

        txt_col = GRN if rate == 1.0 else (YEL if rate > 0 else GRY)
        t_label = format_time_adaptive(avg) if has_data else "-"
        cell = (f" {txt_col}{rate * 100:>3.0f}%{RST}"
                f"{'':>{sr_w - 5}}{sr_col}{sr_sym}{RST}"
                f" | {txt_col}{t_label:<{t_w}}{RST}"
                f"{t_col}{t_sym}{RST} ")
        parts.append(cell)

    sys.stdout.write(f"\r| {ps} |" + "|".join(parts) + "|\033[K\n")
    sys.stdout.flush()