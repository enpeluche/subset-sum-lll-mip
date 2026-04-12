"""Consistent visual style per solver name."""

SOLVER_STYLES = {
    # lattice
    "cpsat":    {"color": "#e74c3c", "marker": "s", "label": "CP-SAT pur"},
    "lll":      {"color": "#3498db", "marker": "o", "label": "LLL + CP-SAT"},
    "bkz":      {"color": "#2ecc71", "marker": "^", "label": "BKZ(30) + CP-SAT"},
    "adaptive": {"color": "#f39c12", "marker": "D", "label": "Adaptive (LLL→BKZ)"},
    "mitm":     {"color": "#9b59b6", "marker": "v", "label": "MITM classique"},
    "ultimate": {"color": "#1abc9c", "marker": "*", "label": "Ultimate"},


    # CP-SAT variants
    "CP-SAT Vanilla":   {"color": "#e74c3c", "marker": "s", "label": "CP-SAT Vanilla"},
    "Greedy Bound":     {"color": "#3498db", "marker": "o", "label": "Greedy Bound"},
    "Smart Window":     {"color": "#2ecc71", "marker": "^", "label": "Smart Window"},
    "Smart+Tightening": {"color": "#f39c12", "marker": "D", "label": "Smart+Tightening"},
    "Greedy Extreme":   {"color": "#9b59b6", "marker": "v", "label": "Greedy Extreme"},
    "Full Greedy":      {"color": "#1abc9c", "marker": "*", "label": "Full Greedy"},

    # Tabu variants
    "Tabu Classic":     {"color": "#e74c3c", "marker": "s", "label": "Tabu Classic"},
    "Tabu Gray":        {"color": "#3498db", "marker": "o", "label": "Tabu Gray"},
    "Tabu Beckett":     {"color": "#2ecc71", "marker": "^", "label": "Tabu Beckett"},
    "Classic+Random":   {"color": "#95a5a6", "marker": "x", "label": "Classic (random)"},
    "Classic+Sign":     {"color": "#f39c12", "marker": "D", "label": "Classic (sign)"},
    
    # Exact
    "MITM":             {"color": "#8e44ad", "marker": "v", "label": "MITM Gray"},
}

_DEFAULT = {"color": "gray", "marker": "x"}


def get_style(name: str) -> dict:
    """Return color/marker/label for *name*, with a sensible fallback."""
    base = SOLVER_STYLES.get(name, {**_DEFAULT, "label": name})
    return base