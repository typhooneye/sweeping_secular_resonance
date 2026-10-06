#!/usr/bin/env python
"""Check that a split power-law disk reproduces the no-gap disk force.

The disk is split into three radial sections:

1. inner disk: R_disk,in < r' < R_in
2. annulus: R_in < r' < R_out
3. outer disk: R_out < r' < infinity

For a body located inside the annulus, R_in < r < R_out, the sum of the
three radial force contributions should match the full no-gap power-law disk
series.  The sign convention is positive outward.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from ssr_single_giant import DiskParams, GiantParams, G_PER_CM2_TO_MSUN_PER_AU2


def ward_coefficients(n_terms: int) -> np.ndarray:
    coeffs = np.zeros(n_terms)
    coeffs[0] = 1.0
    for n in range(1, n_terms):
        coeffs[n] = coeffs[n - 1] * (2 * n - 1) ** 2 / (4 * n**2)
    return coeffs


def sigma_code(r: np.ndarray | float, sigma0_cgs: float, k: float) -> np.ndarray | float:
    return sigma0_cgs * G_PER_CM2_TO_MSUN_PER_AU2 * np.asarray(r) ** (-k)


def force_terms(
    r: np.ndarray,
    r_in: float,
    r_out: float,
    sigma0_cgs: float,
    k: float,
    n_terms: int,
    g_const: float = 1.0,
) -> dict[str, np.ndarray]:
    """Return radial force contributions for r inside R_in < r < R_out."""
    r = np.asarray(r, dtype=float)
    sigma = sigma_code(r, sigma0_cgs, k)
    coeffs = ward_coefficients(n_terms)

    inner_sum = np.zeros_like(r)
    annulus_sum = np.zeros_like(r)
    outer_sum = np.zeros_like(r)
    full_sum = np.zeros_like(r)

    for n, coeff in enumerate(coeffs):
        outer_denom = 2 * n - 1 + k
        inner_denom = 2 * n + 2 - k
        if np.isclose(outer_denom, 0.0) or np.isclose(inner_denom, 0.0):
            raise ValueError(f"singular term for k={k:g}, n={n}")

        outer_factor = 2 * n / outer_denom
        inner_factor = (2 * n + 1) / inner_denom

        outer_edge = outer_factor * (r / r_out) ** (2 * n - 1 + k)
        inner_edge = inner_factor * (r_in / r) ** (2 * n + 2 - k)

        inner_sum -= coeff * inner_edge
        outer_sum += coeff * outer_edge
        annulus_sum += coeff * (outer_factor * (1.0 - (r / r_out) ** (2 * n - 1 + k)))
        annulus_sum -= coeff * (inner_factor * (1.0 - (r_in / r) ** (2 * n + 2 - k)))
        full_sum += coeff * (outer_factor - inner_factor)

    prefactor = 2.0 * math.pi * g_const * sigma
    inner = prefactor * inner_sum
    annulus = prefactor * annulus_sum
    outer = prefactor * outer_sum
    full = prefactor * full_sum
    split_sum = inner + annulus + outer

    return {
        "inner": inner,
        "annulus": annulus,
        "outer": outer,
        "split_sum": split_sum,
        "full_series": full,
        "z_from_series": -0.5 * full_sum,
        "full_zk": -4.0 * math.pi * g_const * sigma * (-0.5 * full_sum),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-giant", type=float, default=5.2)
    parser.add_argument("--e-giant", type=float, default=0.05)
    parser.add_argument("--m-giant-mjup", type=float, default=1.0)
    parser.add_argument("--mstar", type=float, default=1.0)
    parser.add_argument("--gap-width", type=float, default=1.0, help="Hill-radius multiplier n.")
    parser.add_argument("--sigma0-cgs", type=float, default=1700.0)
    parser.add_argument("--k", type=float, default=1.5)
    parser.add_argument("--n-terms", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--output", default="figure/truncated_disk_terms.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_giant = GiantParams(args.a_giant, args.e_giant, args.m_giant_mjup, args.mstar)
    hill_factor = (base_giant.mass_code / (3.0 * base_giant.mstar)) ** (1.0 / 3.0)
    r_in = base_giant.a * (1.0 - base_giant.e) * (1.0 - args.gap_width * hill_factor)
    r_out = base_giant.a * (1.0 + base_giant.e) * (1.0 + args.gap_width * hill_factor)

    r = np.linspace(r_in * 1.001, r_out * 0.999, args.n_samples)
    terms = force_terms(r, r_in, r_out, args.sigma0_cgs, args.k, args.n_terms)
    residual = terms["split_sum"] - terms["full_series"]
    rel = np.abs(residual) / np.maximum(np.abs(terms["full_series"]), np.finfo(float).eps)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r_au", "inner", "annulus", "outer", "split_sum", "full_series", "residual"])
        for idx, rr in enumerate(r):
            writer.writerow([
                rr,
                terms["inner"][idx],
                terms["annulus"][idx],
                terms["outer"][idx],
                terms["split_sum"][idx],
                terms["full_series"][idx],
                residual[idx],
            ])

    print(f"R_in={r_in:.8g} AU")
    print(f"R_out={r_out:.8g} AU")
    print(f"wrote {output}")
    print(f"max |split_sum - full_series| = {np.max(np.abs(residual)):.6e}")
    print(f"max relative residual = {np.max(rel):.6e}")
    print(f"Z_k implied by full series at first sample = {terms['z_from_series'][0]:.8g}")
    print(f"Z_k in DiskParams default = {DiskParams().z_k:.8g}")


if __name__ == "__main__":
    main()
