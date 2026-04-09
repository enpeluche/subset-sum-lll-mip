# Palette cohérente par nom de solveur — extensible
SOLVER_STYLES = {
    "cpsat": {"color": "#e74c3c", "marker": "s", "label": "CP-SAT pur"},
    "lll": {"color": "#3498db", "marker": "o", "label": "LLL + CP-SAT"},
    "bkz": {"color": "#2ecc71", "marker": "^", "label": "BKZ(30) + CP-SAT"},
    "adaptive": {"color": "#f39c12", "marker": "D", "label": "Adaptive (LLL→BKZ)"},
    "mitm": {"color": "#9b59b6", "marker": "v", "label": "MITM classique"},
    "ultimate": {
        "color": "#1abc9c",
        "marker": "*",
        "label": "Ultimate (LLL→BKZ→MITM→CP-SAT)",
    },
}
DEFAULT_STYLE = {"color": "gray", "marker": "x", "label": "Solveur"}


def get_style(name: str) -> dict:
    return SOLVER_STYLES.get(name, {**DEFAULT_STYLE, "label": name})
