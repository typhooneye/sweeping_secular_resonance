#!/usr/bin/env python
"""Parameter-search plot for resonance surfing (single-giant NLT05-type disk).

Surfing number for the radial eccentricity-damping force, with an assumed
post-crossing eccentricity e_kick:

    S = |adot_damp| * |dt_SR/da|
      = [4 a (1-sqrt(1-e_kick^2)) / t_e] * [tau_n |d ln u_R/da|],

for a_damp=-2 v_r r_hat/t_e.  At small eccentricity,
<adot>/a=-2e^2/t_e.

with t_e evaluated at the depleted surface density Sigma(a)*u_R when the
resonance arrives. S is linear in tau_n, so the surfing boundary is a curve
tau_S=1(a); surfing is possible ABOVE it only where the resonance also sweeps
inward (dt_SR/da < 0).

The plot shows, for one embryo mass and several giant masses m_J:
  * solid lines: tau_S=1(a), the S = 1 surfing boundary;
  * dashed lines: S = 1 for the alternative f_e value;
  * dotted lines: attainability tau_att(a), above which the stationary-phase
    crossing amplitude e_kick_sp = |A_J| e_J sqrt(2 pi/|gdot_rel|) reaches
    the assumed e_kick (linear-theory upper bound; gdot_rel = g_pl/tau at
    crossing, so this depends only on the giant, not the disk/gap config);
  * left panel: gapless disk; right panel: empty gap (edges from gap_edges).

Best cases sit above BOTH lines of a given color.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ssr_single_giant import (
    DiskParams,
    GiantParams,
    disk_precession_gap,
    disk_precession_no_gap,
    circularization_migration_speed,
    laplace_coefficient,
    mean_motion,
    resonance_u,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-min", type=float, default=0.4)
    parser.add_argument("--a-max", type=float, default=3.0)
    parser.add_argument("--n-a", type=int, default=300)
    parser.add_argument("--tau-min", type=float, default=3.0e4)
    parser.add_argument("--tau-max", type=float, default=3.0e7)
    parser.add_argument("--m-emb", type=float, default=1.0, help="embryo mass [Mearth]")
    parser.add_argument("--mj-list", default="1,3,10", help="giant masses [Mjup]")
    parser.add_argument("--e-kick", type=float, default=0.1)
    parser.add_argument("--a-jup", type=float, default=5.203, help="giant semimajor axis [AU]")
    parser.add_argument("--e-jup", type=float, default=0.048)
    parser.add_argument("--h0", type=float, default=0.025, help="scale height at 1 AU [AU]")
    parser.add_argument("--f-e-alt", type=float, default=0.5,
                        help="alternative damping factor for the dashed S=1 boundaries")
    parser.add_argument("--output", default=None,
                        help="output PNG path; default: figure/surfing_search_map_aj<a_jup>_ej<e_jup>.png")
    return parser.parse_args()


def boundary_curves(a, giant, disk, m_emb, e_kick, gap):
    """Return (tau_S1, tau_att) on the a grid for one giant mass."""
    if gap:
        u_r = resonance_u(a, giant, disk, gap=True, gap_depletion=0.0)
        g_giant = disk_precession_gap(np.array([giant.a]), giant, disk)[0]
        g_emb = disk_precession_gap(a, giant, disk)
    else:
        u_r = resonance_u(a, giant, disk, gap=False)
        g_giant = disk_precession_no_gap(giant.a, giant.mstar, disk)
        g_emb = disk_precession_no_gap(a, giant.mstar, disk)
    with np.errstate(invalid="ignore", divide="ignore"):
        dlnu_da = np.gradient(np.log(u_r), a)
    # t_R=-tau_n ln(u_R), so dln(u_R)/da>0 is an inward-moving resonance.
    inward_sweep = dlnu_da > 0.0

    adot = np.abs(circularization_migration_speed(
        a,
        e_kick,
        m_emb,
        giant.mstar,
        disk,
        depletion_factor=u_r,
    ))
    with np.errstate(invalid="ignore", divide="ignore"):
        tau_s1 = 1.0 / (adot * np.abs(dlnu_da))
    tau_s1 = np.where(inward_sweep, tau_s1, np.nan)

    # attainability: gdot_rel = u_R |gJ - ge| / tau = g_pl/tau at crossing
    n = mean_motion(a, giant.mstar)
    alpha = a / giant.a
    b2 = laplace_coefficient(1.5, 2, alpha)
    coupling = (n / 4.0) * (giant.mass_code / giant.mstar) * alpha**2 * b2
    gdot_over_tau = 2.0 * np.pi * u_r * np.abs(g_giant - g_emb)
    with np.errstate(invalid="ignore", divide="ignore"):
        kick_coeff = 2.0 * np.pi * coupling * giant.e * np.sqrt(2.0 * np.pi / gdot_over_tau)
        tau_att = (e_kick / kick_coeff) ** 2
    tau_att = np.where(inward_sweep, tau_att, np.nan)
    return tau_s1, tau_att


def main() -> None:
    args = parse_args()
    mj_list = [float(v) for v in args.mj_list.split(",")]
    disk = DiskParams(sigma0_cgs=1700.0, k=1.5, z_k=1.094, t_dep=1.0e6,
                      disk_inner_cutoff=0.0, scale_height_1au=args.h0,
                      scale_height_index=1.25, f_e=1.0, include_gr=False)
    disk_alt = replace(disk, f_e=args.f_e_alt)
    a = np.linspace(args.a_min, args.a_max, args.n_a)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True,
                             constrained_layout=True)
    colors = plt.cm.plasma(np.linspace(0.0, 0.75, len(mj_list)))

    print(f"tau_S=1 [yr] (solid; dashed for alternate f_e) / tau_att [yr] (dotted), "
          f"m_emb={args.m_emb:g} Me, h0={args.h0:g}, e_kick={args.e_kick:g}")
    for ax, (label, gap) in zip(axes, (("no gap", False), ("empty gap", True))):
        for color, m_j in zip(colors, mj_list):
            giant = GiantParams(a=args.a_jup, e=args.e_jup, mass_mjup=m_j, mstar=1.0)
            tau_s1, tau_att = boundary_curves(a, giant, disk, args.m_emb,
                                              args.e_kick, gap)
            # weaker damping: t_e -> t_e/f_e_alt, so tau_S1 scales by 1/f_e_alt
            tau_s1_alt, _ = boundary_curves(a, giant, disk_alt, args.m_emb,
                                            args.e_kick, gap)
            ax.plot(a, tau_s1, color=color, linewidth=2.2,
                    label=rf"$m_J={m_j:g}\,M_{{\rm Jup}}$")
            ax.plot(a, tau_s1_alt, color=color, linewidth=1.8, linestyle="--")
            ax.plot(a, tau_att, color=color, linewidth=1.3, linestyle=":")
            for atest in (1.0, 2.0):
                i = np.argmin(np.abs(a - atest))
                print(f"  [{label}] mJ={m_j:4g}  a={a[i]:4.2f}: "
                      f"tau_S1={tau_s1[i]:9.3e}  tau_S1(f_e={args.f_e_alt:g})="
                      f"{tau_s1_alt[i]:9.3e}  tau_att={tau_att[i]:9.3e}")

        ax.set_yscale("log")
        ax.set_xlim(args.a_min, args.a_max)
        ax.set_ylim(args.tau_min, args.tau_max)
        ax.set_xlabel(r"$a_{\rm emb}$ [AU]")
        ax.set_title(label)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel(r"$\tau_n$ [yr]")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle(
        rf"Circular-velocity damping surfing boundaries"
        "\n"
        rf"$m={args.m_emb:g}\,M_\oplus$, $h_0={args.h0:g}$, "
        rf"$e_{{\rm kick}}={args.e_kick:g}$, $a_J={args.a_jup:g}$ AU, $e_J={args.e_jup:g}$ "
        rf"(solid: $S=1$, $f_e=1$; dashed: $S=1$, $f_e={args.f_e_alt:g}$; "
        r"dotted: $e_{\rm kick}$ attainable above)",
        fontsize=10.5,
    )

    output = Path(args.output) if args.output else Path(
        f"figure/surfing_search_map_aj{args.a_jup:g}_ej{args.e_jup:g}.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250)
    print(output)


if __name__ == "__main__":
    main()
