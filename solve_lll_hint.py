from constants import TIMEOUT
import time
from ortools.sat.python import cp_model
from util import (
    knapsack_matrix,
    filter_binary_candidates,
    extract_coefficient_submatrix,
)
from fpylll import LLL


def project_to_binary(v):
    return [max(0, min(1, x)) for x in v]


def solve_lll_hint(weights, T, sol):
    start_global_time = time.perf_counter()
    n = len(weights)

    total_branches = 0
    total_conflicts = 0

    # --- LLL reduction ---
    B = knapsack_matrix(weights, T, 2 ** n)
    LLL.reduction(B)
    sub = extract_coefficient_submatrix(B)

    # --- Métrique par défaut (vecteur le plus court projeté) ---
    if len(sub) > 0:
        shortest_bin = project_to_binary(sub[0])
        best_res = abs(sum(weights[i] * shortest_bin[i] for i in range(n)) - T)
        best_ham = sum(1 for i in range(n) if shortest_bin[i] != sol[i])
    else:
        best_res = float("inf")
        best_ham = n

    # --- Construction de la file de hints ---
    # Priorité 1 : candidats binaires directs (résidu=0 potentiel)
    direct_candidates = filter_binary_candidates(sub)

    # Priorité 2 : projections de tous les vecteurs
    seen = set()
    projected_candidates = []
    for v in sub:
        p = project_to_binary(v)
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            projected_candidates.append(p)

    # Affiner best_res/best_ham sur les candidats directs
    for c in direct_candidates:
        res = abs(sum(weights[i] * c[i] for i in range(n)) - T)
        ham = sum(1 for i in range(n) if c[i] != sol[i])
        if res < best_res:
            best_res, best_ham = res, ham

    # Trier directs par résidu croissant
    direct_candidates.sort(
        key=lambda c: abs(sum(weights[i] * c[i] for i in range(n)) - T)
    )

    # Trier projections par résidu projeté croissant
    projected_candidates.sort(
        key=lambda p: abs(sum(weights[i] * p[i] for i in range(n)) - T)
    )

    # File complète : directs en premier, projections ensuite
    # On dédoublonne les projections déjà présentes dans directs
    direct_keys = {tuple(c) for c in direct_candidates}
    projected_only = [p for p in projected_candidates if tuple(p) not in direct_keys]

    all_hints = direct_candidates + projected_only

    # --- Calcul du micro_timeout adaptatif ---
    # Budget : on réserve au moins 50% du timeout pour le fallback
    max_hint_budget = TIMEOUT * 0.5
    micro_timeout = min(2.0, max_hint_budget / max(len(all_hints), 1))

    # --- Tentatives avec hints ---
    for idx, hint in enumerate(all_hints):
        time_spent = time.perf_counter() - start_global_time
        if time_spent >= TIMEOUT * 0.5:
            break  # on arrête les hints, on passe au fallback

        model = cp_model.CpModel()
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = micro_timeout
        solver.parameters.num_search_workers = 1

        x = [model.new_bool_var(f"x{i}") for i in range(n)]
        model.add(sum(weights[i] * x[i] for i in range(n)) == T)

        for i in range(n):
            model.add_hint(x[i], hint[i])

        status = solver.solve(model)
        total_branches += solver.num_branches
        total_conflicts += solver.num_conflicts

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            solution = [solver.value(x[i]) for i in range(n)]
            final_time = time.perf_counter() - start_global_time
            label = (
                f"Direct_Hint_{idx + 1}"
                if idx < len(direct_candidates)
                else f"Projected_Hint_{idx - len(direct_candidates) + 1}"
            )
            return (
                final_time,
                total_branches,
                total_conflicts,
                status,
                solution,
                label,
                best_res,
                best_ham,
            )

    # --- Fallback CP-SAT pur ---
    time_spent_so_far = time.perf_counter() - start_global_time
    remaining_time = TIMEOUT - time_spent_so_far

    if remaining_time <= 0:
        return (
            time_spent_so_far,
            total_branches,
            total_conflicts,
            cp_model.UNKNOWN,
            None,
            "Timeout_LLL",
            best_res,
            best_ham,
        )

    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = remaining_time

    x = [model.new_bool_var(f"x{i}") for i in range(n)]
    model.add(sum(weights[i] * x[i] for i in range(n)) == T)

    status = solver.solve(model)
    total_branches += solver.num_branches
    total_conflicts += solver.num_conflicts

    solution = None
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        solution = [solver.value(x[i]) for i in range(n)]

    final_time = time.perf_counter() - start_global_time
    label = (
        "Standard_Fallback"
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]
        else "Timeout_Fallback"
    )

    return (
        final_time,
        total_branches,
        total_conflicts,
        status,
        solution,
        label,
        best_res,
        best_ham,
    )
