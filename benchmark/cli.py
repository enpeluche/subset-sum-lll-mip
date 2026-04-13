"""CLI argument parsing and range utilities."""

import argparse
import numpy as np


def parse_range(s: str, type_cast=float) -> list:
    """
    Parse '0.1,0.5,1.0' or '0.1:1.0:0.1' into a typed list.

    Formats:
        "20,40,60"       → [20, 40, 60]
        "0.1:1.0:0.1"    → [0.1, 0.2, ..., 1.0]
        "10:50"           → [10, 11, ..., 50]  (step=1 for int, 0.1 for float)
    """
    if not s:
        return []

    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1 if type_cast is int else 0.1
        else:
            start, stop, step = parts
        num = int(round((stop - start) / step)) + 1
        return [type_cast(x) for x in np.linspace(start, stop, num)]

    return [type_cast(x.strip()) for x in s.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lattice Benchmark Framework")

    # Axes
    p.add_argument("--n",         type=str, default="30",   help="Dimensions  (20,40,60 ou 20:60:10)")
    p.add_argument("--densities", type=str, default="1.0",  help="Densités    (0.1,0.5,1.0 ou 0.1:1.0:0.1)")
    p.add_argument("--deltas",    type=str, default="0.99", help="Deltas LLL  (0.75,0.99)")
    p.add_argument("--blocks",    type=str, default="20",   help="BKZ blocks  (10,20,30)")

    # Control
    p.add_argument("--suite",   type=str,   choices=["lattice_arch",
                                                     "lattice_delta",
                                                     "cpsat_formulation",
                                                     "lattice_block",
                                                     "tabu_comp",
                                                     "exact_comp",
                                                     "lattice_hybrid_comp",
                                                     "cpsat_comp",
                                                     "lattice_scaling",
                                                     "gray",
                                                     "gray_landscape"])
    p.add_argument("--runs",    type=int,   default=10)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--gen",     type=str,   choices=["uniform", "super_inc", "crypto", "crypto_big"], default="crypto")

    # Output
    p.add_argument("--out", type=str, default="default", help="Sous-dossier dans results/")

    return p


def parse_all_ranges(args) -> dict:
    """Parse every axis from CLI args into a single dict."""
    return {
        "n":          parse_range(args.n, int),
        "density":    parse_range(args.densities, float),
        "delta":      parse_range(args.deltas, float),
        "block_size": parse_range(args.blocks, int),
    }