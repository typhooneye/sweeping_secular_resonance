#!/usr/bin/env python
"""Analytical sweeping secular resonance estimate for one exterior giant.

The calculation follows the precession-frequency matching used by
Nagasawa et al. (2005) and Zheng et al. (2017):

    g_i,planet + g_i,disk(t) = g_j,disk(t)

with the disk terms depleted as exp(-t/t_dep).  The returned value
u_R = exp(-t_R/t_dep) is physically meaningful only for 0 < u_R <= 1.
"""

from __future__ import annotations

import argparse
import csv
import math
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import quad


G_PER_CM2_TO_MSUN_PER_AU2 = 1.125e-7
MJUP_TO_MSUN = 9.545e-4
MEARTH_TO_MSUN = 3.003489614915e-6
C_CODE = 1.0e4


@lru_cache(maxsize=64)
def z_k_exact(disk_k: float, n_terms: int = 100000) -> float:
    """Exact local-disk coefficient z_k for Sigma ~ r^-k.

    F_full = -2 pi G Sigma(r) z_k/(2-k) with
    z_k = 1 + (2-k)(k-1) sum_l (4l+1) A_l/((2l-1+k)(2l+2-k)).
    z_1 = 1 exactly (Mestel); z_1.5 = 1.0942 (legacy hard-coded 1.094).
    """
    total = 0.0
    a_l = 1.0
    for ell in range(1, n_terms + 1):
        a_l *= (2.0 * ell - 1.0) ** 2 / (4.0 * ell * ell)
        total += (4.0 * ell + 1.0) * a_l / ((2.0 * ell - 1.0 + disk_k) * (2.0 * ell + 2.0 - disk_k))
    return 1.0 + (2.0 - disk_k) * (disk_k - 1.0) * total


@dataclass(frozen=True)
class DiskParams:
    sigma0_cgs: float = 1700.0
    k: float = 1.5
    # None -> exact z_k for this k (recommended); a number overrides it.
    z_k: float | None = None
    t_dep: float = 1.0e6
    disk_inner_cutoff: float = 0.0
    n_terms: int = 30
    k_gap_dep: float = 0.0
    scale_height_1au: float = 0.025
    scale_height_index: float = 1.25
    f_e: float = 1.0
    f_a: float = 1.0
    include_gr: bool = True
    include_gap_effect: bool = True

    @property
    def sigma0_code(self) -> float:
        return self.sigma0_cgs * G_PER_CM2_TO_MSUN_PER_AU2

    @property
    def z_k_value(self) -> float:
        return self.z_k if self.z_k is not None else z_k_exact(self.k)


@dataclass(frozen=True)
class GiantParams:
    a: float = 5.2
    e: float = 0.05
    mass_mjup: float = 1.0
    mstar: float = 1.0
    gap_r_in: float | None = None
    gap_r_out: float | None = None

    @property
    def mass_code(self) -> float:
        return self.mass_mjup * MJUP_TO_MSUN


def mean_motion(a: np.ndarray | float, mstar: float) -> np.ndarray | float:
    return np.sqrt(mstar / np.asarray(a) ** 3)


def sigma_init(a: np.ndarray | float, disk: DiskParams) -> np.ndarray | float:
    return disk.sigma0_code * np.asarray(a) ** (-disk.k)


def scale_height(a: np.ndarray | float, disk: DiskParams) -> np.ndarray | float:
    return disk.scale_height_1au * np.asarray(a) ** disk.scale_height_index


def type1_eccentricity_damping_time(
    a: np.ndarray | float,
    embryo_mass_earth: float,
    mstar: float,
    disk: DiskParams,
    depletion_factor: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Artymowicz (1993) / Ward (1993) Type-I eccentricity damping time.

    The returned timescale is in years.  The depletion factor is
    f_dep(t)=exp(-t/T_dep); at SR, pass u_R.
    """
    a = np.asarray(a, dtype=float)
    f_dep = np.asarray(depletion_factor, dtype=float)
    embryo_mass_code = embryo_mass_earth * MEARTH_TO_MSUN
    sigma = sigma_init(a, disk) * f_dep
    h_over_a = scale_height(a, disk) / a
    omega_k = 2.0 * math.pi * np.sqrt(mstar / a**3)
    timescale = (
        (mstar / embryo_mass_code)
        * (mstar / (sigma * a**2))
        * h_over_a**4
        / omega_k
        / disk.f_e
    )
    return np.where((sigma > 0.0) & (f_dep > 0.0), timescale, np.nan)


def type1_migration_time(
    a: np.ndarray | float,
    embryo_mass_earth: float,
    mstar: float,
    disk: DiskParams,
    depletion_factor: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Tanaka et al. (2002) Type-I semimajor-axis migration time."""
    a = np.asarray(a, dtype=float)
    f_dep = np.asarray(depletion_factor, dtype=float)
    embryo_mass_code = embryo_mass_earth * MEARTH_TO_MSUN
    sigma = sigma_init(a, disk) * f_dep
    h_over_a = scale_height(a, disk) / a
    omega_k = 2.0 * math.pi * np.sqrt(mstar / a**3)
    timescale = (
        1.0 / (2.7 + 1.1 * disk.k)
        * (mstar / embryo_mass_code)
        * (mstar / (sigma * a**2))
        * h_over_a**2
        / omega_k
        / disk.f_a
    )
    return np.where((sigma > 0.0) & (f_dep > 0.0), timescale, np.nan)


def type1_migration_speed(
    a: np.ndarray | float,
    embryo_mass_earth: float,
    mstar: float,
    disk: DiskParams,
    depletion_factor: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Inward explicit Type-I migration speed da/dt in AU/yr."""
    a = np.asarray(a, dtype=float)
    t_mig = type1_migration_time(a, embryo_mass_earth, mstar, disk, depletion_factor)
    return -a / t_mig


def circularization_migration_speed(
    a: np.ndarray | float,
    eccentricity: float,
    embryo_mass_earth: float,
    mstar: float,
    disk: DiskParams,
    depletion_factor: np.ndarray | float = 1.0,
) -> np.ndarray:
    r"""Orbit-averaged migration from radial eccentricity damping.

    For a_e = -2 v_r e_r/t_e, angular momentum is conserved and

        <da/dt>/a = -4 [1-sqrt(1-e^2)]/t_e
                  = -2 e^2/t_e + O(e^4).
    """
    a = np.asarray(a, dtype=float)
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("eccentricity must satisfy 0 <= e < 1")
    t_e = type1_eccentricity_damping_time(a, embryo_mass_earth, mstar, disk, depletion_factor)
    if eccentricity == 0.0:
        return np.zeros_like(a)

    return -4.0 * a * (1.0 - math.sqrt(1.0 - eccentricity**2)) / t_e


@lru_cache(maxsize=200_000)
def _laplace_coefficient_scalar(s: float, order: int, alpha: float) -> float:
    """Numerical Laplace coefficient integral matching ref/ref_code.txt."""
    if alpha == 0.0:
        return 0.0

    def integrand(phi: float) -> float:
        denominator = (1.0 - alpha) ** 2 + 2.0 * alpha * (1.0 - math.cos(phi))
        return math.cos(order * phi) / denominator**s

    value, _ = quad(
        integrand,
        0.0,
        math.pi,
        points=[1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2],
        limit=300,
    )
    return 2.0 * value / math.pi


def laplace_coefficient(s: float, order: int, alpha: np.ndarray) -> np.ndarray:
    """Numerical Laplace coefficient integral used by the reference code."""
    alpha_arr = np.asarray(alpha, dtype=float)
    result = np.empty_like(alpha_arr, dtype=float)
    for idx, alpha_value in np.ndenumerate(alpha_arr):
        clipped_alpha = min(max(float(alpha_value), 0.0), 1.0 - 1.0e-12)
        cached_alpha = min(round(clipped_alpha, 12), 1.0 - 1.0e-10)
        result[idx] = _laplace_coefficient_scalar(float(s), int(order), cached_alpha)
    return result


def gap_edges(giant: GiantParams) -> tuple[float, float]:
    if giant.gap_r_in is not None and giant.gap_r_out is not None:
        return giant.gap_r_in, giant.gap_r_out

    hill = (giant.mass_code / (3.0 * giant.mstar)) ** (1.0 / 3.0)
    if giant.e > hill:
        r_in = giant.a * (1.0 - giant.e) * (1.0 - hill)
        r_out = giant.a * (1.0 + giant.e) * (1.0 + hill)
    else:
        r_in = giant.a * (1.0 - hill)
        r_out = giant.a * (1.0 + hill)
    return r_in, r_out


def a_l_coefficients(n_terms: int) -> np.ndarray:
    coeffs = np.zeros(n_terms)
    for idx in range(n_terms):
        ell = idx + 1
        coeffs[idx] = (math.factorial(2 * ell) / 2 ** (2 * ell) / math.factorial(ell) ** 2) ** 2
    return coeffs


def planet_precession_from_giant(a_inner: np.ndarray, giant: GiantParams) -> np.ndarray:
    a_inner = np.asarray(a_inner)
    alpha = np.minimum(a_inner, giant.a) / np.maximum(a_inner, giant.a)
    n_inner = mean_motion(a_inner, giant.mstar)
    a_outer = np.maximum(a_inner, giant.a)
    b = laplace_coefficient(1.5, 1, alpha)
    return giant.mass_code * alpha * b / (4.0 * n_inner * a_inner**2 * a_outer)


def gr_precession(a: np.ndarray | float, mstar: float) -> np.ndarray:
    """General-relativistic apsidal precession in the reference code units."""
    a_arr = np.asarray(a, dtype=float)
    n = mean_motion(a_arr, mstar)
    return 3.0 * n * mstar / (a_arr * C_CODE**2)


def disk_precession_no_gap(a: np.ndarray | float, mstar: float, disk: DiskParams) -> np.ndarray:
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = mean_motion(a, mstar)
    full_disk = -disk.z_k_value * math.pi * sigma_init(a, disk) / (n * a)
    if disk.disk_inner_cutoff <= 0.0:
        return full_disk

    cut_disk = full_disk - disk_precession_inner_disk(a, mstar, disk)
    inside_cut = a <= disk.disk_inner_cutoff
    if np.any(inside_cut):
        cut_disk = np.asarray(cut_disk, dtype=float)
        cut_disk[inside_cut] = disk_precession_exterior_disk(
            a[inside_cut],
            disk.disk_inner_cutoff,
            mstar,
            disk,
        )
    return cut_disk


def disk_precession_exterior_disk(
    a: np.ndarray | float,
    r_edge: float,
    mstar: float,
    disk: DiskParams,
) -> np.ndarray:
    """Precession from a power-law disk exterior to r_edge for a < r_edge."""
    a_arr = np.atleast_1d(np.asarray(a, dtype=float))
    coeffs = a_l_coefficients(disk.n_terms)
    result = np.zeros_like(a_arr)
    for idx, aa in enumerate(a_arr):
        if aa <= 0.0:
            result[idx] = np.nan
            continue

        n = mean_motion(aa, mstar)
        prefactor = math.pi * sigma_init(aa, disk) / (n * aa)
        edge_sum = 0.0
        # 2l(2l+1): each force term g ~ r^{2l-1} precesses at
        # (2g/r+g')/(2 Omega) = (2l+1) g/(2 Omega r); force coefficient 2l A_l
        # verified against direct integration (2026-08-23).
        for ell, coeff in enumerate(coeffs, start=1):
            edge_sum += (
                2 * ell
                * (2 * ell + 1)
                * coeff
                * (aa / r_edge) ** (2 * ell - 1 + disk.k)
                / (2 * ell - 1 + disk.k)
            )
        result[idx] = prefactor * edge_sum

    return result


def disk_precession_inner_disk(
    a: np.ndarray | float,
    mstar: float,
    disk: DiskParams,
) -> np.ndarray:
    """Precession contribution from the removed disk between 0 and disk_inner_cutoff."""
    a_arr = np.atleast_1d(np.asarray(a, dtype=float))
    result = np.zeros_like(a_arr)
    if disk.disk_inner_cutoff <= 0.0:
        return result

    coeffs = a_l_coefficients(disk.n_terms)
    for idx, aa in enumerate(a_arr):
        if aa <= disk.disk_inner_cutoff:
            continue

        n = mean_motion(aa, mstar)
        prefactor = math.pi * sigma_init(aa, disk) / (n * aa)
        edge_sum = 0.0
        # 2l(2l+1) as in the exterior series; the l=0 (monopole) force term
        # is Keplerian and contributes zero precession.
        for ell, coeff in enumerate(coeffs, start=1):
            edge_sum += (
                2 * ell
                * (2 * ell + 1)
                * coeff
                * (disk.disk_inner_cutoff / aa) ** (2 * ell + 2 - disk.k)
                / (2 * ell + 2 - disk.k)
            )
        result[idx] = prefactor * edge_sum

    return result


def disk_precession_gap(a: np.ndarray | float, giant: GiantParams, disk: DiskParams) -> np.ndarray:
    """Disk-induced apsidal precession with a single giant-opened gap.

    Inner embryos use the Multi_HR5183 R_in correction, bodies inside the gap
    use the two truncated disk edges, and exterior bodies use the local disk.
    """
    if not disk.include_gap_effect:
        return disk_precession_no_gap(a, giant.mstar, disk)

    a = np.atleast_1d(np.asarray(a, dtype=float))
    r_in, r_out = gap_edges(giant)
    coeffs = a_l_coefficients(disk.n_terms)
    result = np.zeros_like(a)

    def ext_sum(aa: float, edge: float) -> float:
        # prograde precession sum from disk exterior to `edge` (aa < edge)
        return sum(
            2 * ell * (2 * ell + 1) * coeff
            * (aa / edge) ** (2 * ell - 1 + disk.k) / (2 * ell - 1 + disk.k)
            for ell, coeff in enumerate(coeffs, start=1)
        )

    def int_sum(aa: float, edge: float) -> float:
        # prograde precession sum from disk interior to `edge` (aa > edge)
        return sum(
            2 * ell * (2 * ell + 1) * coeff
            * (edge / aa) ** (2 * ell + 2 - disk.k) / (2 * ell + 2 - disk.k)
            for ell, coeff in enumerate(coeffs, start=1)
        )

    def multi_hr_inner_sum(aa: float) -> float:
        return sum(
            ell * (2 * ell + 1) * coeff
            * (aa / r_in) ** (2 * ell - 1 + disk.k)
            / (2 * ell - 1 + disk.k)
            for ell, coeff in list(enumerate(coeffs, start=1))[:9]
        )

    for idx, aa in enumerate(a):
        n = mean_motion(aa, giant.mstar)
        prefactor = math.pi * sigma_init(aa, disk) / (n * aa)

        if aa < r_in:
            result[idx] = -2.0 * prefactor * (
                (2.0 - disk.k) * disk.z_k_value
                + multi_hr_inner_sum(aa)
            )
        elif aa <= r_in or aa >= r_out:
            result[idx] = disk_precession_no_gap(
                np.array([aa]), giant.mstar, disk
            )[0]
        else:
            result[idx] = prefactor * (int_sum(aa, r_in) + ext_sum(aa, r_out))

    inside_gap = (a > r_in) & (a < r_out)
    if np.any(inside_gap):
        result[inside_gap] -= disk_precession_inner_disk(
            a[inside_gap], giant.mstar, disk
        )
    return result


def disk_precession_dep_gap(
    a: np.ndarray | float,
    giant: GiantParams,
    disk: DiskParams,
    k_gap_dep: float | None = None,
) -> np.ndarray:
    """Disk precession for a gap whose density is k_gap_dep * Sigma(r).

    For the axisymmetric disk potential,

        Phi_dep_gap = Phi_no_gap - (1 - k_gap_dep) Phi_annulus
                    = Phi_empty_gap + k_gap_dep Phi_annulus,

    where Phi_annulus is the contribution from R_in < r' < R_out.  Because
    apsidal precession is linear in the disk potential, the same decomposition
    applies to g_disk:

        g_dep_gap = g_empty_gap + k_gap_dep (g_no_gap - g_empty_gap).
    """
    if k_gap_dep is None:
        k_gap_dep = disk.k_gap_dep
    if not 0.0 <= k_gap_dep <= 1.0:
        raise ValueError("k_gap_dep must satisfy 0 <= k_gap_dep <= 1")
    if not disk.include_gap_effect:
        return disk_precession_no_gap(a, giant.mstar, disk)

    g_no_gap = disk_precession_no_gap(a, giant.mstar, disk)
    g_empty_gap = disk_precession_gap(a, giant, disk)
    return g_empty_gap + k_gap_dep * (g_no_gap - g_empty_gap)


def resonance_u(
    a_embryo: np.ndarray,
    giant: GiantParams,
    disk: DiskParams,
    gap: bool = False,
    gap_depletion: float | None = None,
) -> np.ndarray:
    """Return u_R = exp(-t_R/t_dep) for each embryo semimajor axis."""
    a_embryo = np.asarray(a_embryo, dtype=float)
    g_planet = planet_precession_from_giant(a_embryo, giant)
    if disk.include_gr:
        g_planet = g_planet + gr_precession(a_embryo, giant.mstar)

    if gap and gap_depletion is not None:
        g_giant_disk = disk_precession_dep_gap(np.array([giant.a]), giant, disk, gap_depletion)[0]
        g_embryo_disk = disk_precession_dep_gap(a_embryo, giant, disk, gap_depletion)
    elif gap:
        g_giant_disk = disk_precession_gap(np.array([giant.a]), giant, disk)[0]
        g_embryo_disk = disk_precession_gap(a_embryo, giant, disk)
    else:
        g_giant_disk = disk_precession_no_gap(giant.a, giant.mstar, disk)
        g_embryo_disk = disk_precession_no_gap(a_embryo, giant.mstar, disk)

    denominator = g_giant_disk - g_embryo_disk
    u_r = np.divide(g_planet, denominator, out=np.full_like(g_planet, np.nan), where=denominator != 0.0)
    return np.where((u_r > 0.0) & (u_r <= 1.0), u_r, np.nan)


def resonance_time(
    a_embryo: np.ndarray,
    giant: GiantParams,
    disk: DiskParams,
    gap: bool = False,
    gap_depletion: float | None = None,
) -> np.ndarray:
    u_r = resonance_u(a_embryo, giant, disk, gap=gap, gap_depletion=gap_depletion)
    return -disk.t_dep * np.log(u_r)


def write_csv(
    path: str,
    a_embryo: np.ndarray,
    u_no_gap: np.ndarray,
    u_gap: np.ndarray,
    u_dep_gap: np.ndarray,
    disk: DiskParams,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "a_au",
            "u_no_gap",
            "t_no_gap_yr",
            "u_gap",
            "t_gap_yr",
            "u_dep_gap",
            "t_dep_gap_yr",
        ])
        for aa, ung, ug, udg in zip(a_embryo, u_no_gap, u_gap, u_dep_gap):
            tng = -disk.t_dep * math.log(ung) if np.isfinite(ung) else np.nan
            tg = -disk.t_dep * math.log(ug) if np.isfinite(ug) else np.nan
            tdg = -disk.t_dep * math.log(udg) if np.isfinite(udg) else np.nan
            writer.writerow([aa, ung, tng, ug, tg, udg, tdg])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-giant", type=float, default=5.2)
    parser.add_argument("--e-giant", type=float, default=0.05)
    parser.add_argument("--m-giant-mjup", type=float, default=1.0)
    parser.add_argument("--mstar", type=float, default=1.0)
    parser.add_argument("--gap-r-in", type=float, default=None)
    parser.add_argument("--gap-r-out", type=float, default=None)
    parser.add_argument("--sigma0-cgs", type=float, default=1700.0)
    parser.add_argument("--k", type=float, default=1.5)
    parser.add_argument("--z-k", type=float, default=None,
                        help="Override the local-disk coefficient; default: exact z_k(k).")
    parser.add_argument("--t-dep", type=float, default=1.0e6)
    parser.add_argument("--disk-inner-cutoff", type=float, default=0.01)
    parser.add_argument("--k-gap-dep", type=float, default=0.0)
    parser.add_argument("--f-e", type=float, default=1.0)
    parser.add_argument("--no-gr", action="store_true", help="Omit GR precession from the SR condition.")
    parser.add_argument(
        "--no-gap-effect",
        action="store_true",
        help="Use the no-gap disk precession even for gap/depleted-gap branches.",
    )
    parser.add_argument("--a-min", type=float, default=0.0)
    parser.add_argument("--a-max", type=float, default=10.0)
    parser.add_argument("--da", type=float, default=0.001)
    parser.add_argument("--output", default="figure/single_giant_resonance.csv")
    parser.add_argument(
        "--preset",
        choices=["zheng2017"],
        default=None,
        help="Use the Zheng et al. 2017 default disk/giant/gap setup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preset == "zheng2017":
        args.a_giant = 5.2
        args.e_giant = 0.05
        args.m_giant_mjup = 1.0
        args.mstar = 1.0
        args.gap_r_in = 4.5
        args.gap_r_out = 11.0
        args.sigma0_cgs = 1700.0
        args.k = 1.5
        args.z_k = 1.094
        args.t_dep = 1.0e6
        args.disk_inner_cutoff = 0.01
        args.k_gap_dep = 0.0
        args.f_e = 1.0
        args.a_min = 2.0
        args.a_max = 3.5

    giant = GiantParams(
        args.a_giant,
        args.e_giant,
        args.m_giant_mjup,
        args.mstar,
        args.gap_r_in,
        args.gap_r_out,
    )
    disk = DiskParams(
        sigma0_cgs=args.sigma0_cgs,
        k=args.k,
        z_k=args.z_k,
        t_dep=args.t_dep,
        disk_inner_cutoff=args.disk_inner_cutoff,
        k_gap_dep=args.k_gap_dep,
        f_e=args.f_e,
        include_gr=not args.no_gr,
        include_gap_effect=not args.no_gap_effect,
    )
    r_in, _ = gap_edges(giant)
    a_max = args.a_max if args.a_max is not None else r_in
    a_start = max(args.a_min, args.da)
    a_embryo = np.arange(a_start, a_max, args.da)

    u_no_gap = resonance_u(a_embryo, giant, disk, gap=False)
    u_gap = resonance_u(a_embryo, giant, disk, gap=True)
    u_dep_gap = resonance_u(a_embryo, giant, disk, gap=True, gap_depletion=disk.k_gap_dep)
    write_csv(args.output, a_embryo, u_no_gap, u_gap, u_dep_gap, disk)
    print(f"wrote {args.output}")
    print(f"gap inner edge: {r_in:.6g} AU")
    print(f"gap depletion factor: {disk.k_gap_dep:.6g}")


if __name__ == "__main__":
    main()
