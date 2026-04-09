import sys
from SubsetSumInstance import SubsetSumInstance
from results import RunRecord, save_records


def run_benchmark(
    n: int,
    densities: list[float],
    n_runs: int,
    solvers: dict[str, callable],
    json_file: str,
) -> list[RunRecord]:

    records = []
    total = len(densities) * n_runs
    done = 0

    for d in densities:
        for i in range(n_runs):
            done += 1
            instance = SubsetSumInstance.create_crypto_density_feasible(n, d)

            solver_results = {}
            for name, solver in solvers.items():
                _print_progress(done, total, d, i, n_runs, name)
                solver_results[name] = solver(instance)

            record = RunRecord(density=d, run_idx=i, n=n, results=solver_results)
            records.append(record)

            # Ligne de résumé après chaque run
            _print_run_summary(record, done, total)

    save_records(records, json_file)
    sys.stdout.write("\n")
    print(f"Done — {total} runs saved → {json_file}")
    return records


def _print_progress(
    done: int, total: int, d: float, i: int, n_runs: int, solver: str
) -> None:
    pct = done / total * 100
    bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    sys.stdout.write(
        f"\r[{bar}] {pct:5.1f}%  d={d:.2f}  run {i+1}/{n_runs}  solver={solver:<12}"
    )
    sys.stdout.flush()


import shutil


def _print_run_summary(record: RunRecord, done: int, total: int) -> None:
    width = shutil.get_terminal_size().columns
    sys.stdout.write("\r" + " " * width + "\r")

    parts = []
    for name, res in record.results.items():
        solved = "✓" if res.solution is not None else "✗"
        parts.append(f"{name}:{solved} {res.elapsed:.2f}s")

    summary = "  |  ".join(parts)
    sys.stdout.write(f"  run {done:>3}/{total}  d={record.density:.2f}  {summary}\n")
    sys.stdout.flush()
