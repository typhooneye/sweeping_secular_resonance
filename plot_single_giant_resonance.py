#!/usr/bin/env python
"""Plot t_R/t_dep versus embryo semimajor axis for one exterior giant."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from ssr_single_giant import (
    DiskParams,
    GiantParams,
    circularization_migration_speed,
    gap_edges,
    resonance_u,
    type1_eccentricity_damping_time,
)


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
    parser.add_argument("--z-k", type=float, default=1.094)
    parser.add_argument("--t-dep", type=float, default=1.0e6)
    parser.add_argument("--disk-inner-cutoff", type=float, default=0)
    parser.add_argument("--k-gap-dep", type=float, default=0.0)
    parser.add_argument("--no-gr", action="store_true", help="Omit GR precession from the SR condition.")
    parser.add_argument(
        "--no-gap-effect",
        action="store_true",
        help="Use the no-gap disk precession even for gap/depleted-gap branches.",
    )
    parser.add_argument("--scale-height-1au", type=float, default=0.025)
    parser.add_argument("--scale-height-index", type=float, default=1.25)
    parser.add_argument("--f-e", type=float, default=1.0)
    parser.add_argument("--embryo-mass-earth", type=float, default=1.0)
    parser.add_argument("--surfing-eccentricity", type=float, default=0.3)
    parser.add_argument("--surfing-k-gap-dep", type=float, default=0.5)
    parser.add_argument("--surfing-a-giant-values", default="1,5,10,15,20,30")
    parser.add_argument("--surfing-t-dep-values", default="1e5,1e6")
    parser.add_argument("--a-min", type=float, default=0.1)
    parser.add_argument("--a-max", type=float, default=10.0)
    parser.add_argument("--da", type=float, default=0.001)
    parser.add_argument("--output", default="figure/single_giant_resonance.png")
    parser.add_argument("--timescale-output", default="figure/single_giant_damping_window.png")
    parser.add_argument("--surfing-output", default="figure/single_giant_surfing_speed.png")
    parser.add_argument("--k-sweep-output", default="figure/single_giant_k_sweep.png")
    parser.add_argument("--k-values", default="1,1.5,2,2.5,3,3.5")
    parser.add_argument(
        "--gap-widths",
        default="1,2,3",
        help=(
            "Comma-separated Hill-radius gap width multipliers n, where "
            "R_in=a_J(1-e_J)(1-n RH/a_J) and R_out=a_J(1+e_J)(1+n RH/a_J)."
        ),
    )
    parser.add_argument(
        "--preset",
        choices=["zheng2017"],
        default=None,
        help="Use the Zheng et al. 2017 default disk/giant/gap setup.",
    )
    return parser.parse_args()


def parse_gap_widths(value: str) -> list[float]:
    widths: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        widths.append(float(item))
    return widths


def parse_float_list(value: str) -> list[float]:
    values: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def format_time_yr(value: float) -> str:
    if value >= 1.0e6:
        return rf"{value / 1.0e6:g} Myr"
    if value >= 1.0e3:
        return rf"{value / 1.0e3:g} kyr"
    return rf"{value:g} yr"


def model_switch_label(disk: DiskParams) -> str:
    gr_label = "GR on" if disk.include_gr else "GR off"
    gap_label = "gap effect on" if disk.include_gap_effect else "gap effect off"
    return f"{gr_label}; {gap_label}"


def hill_gap_edges(giant: GiantParams, width_multiplier: float) -> tuple[float, float]:
    hill_factor = (giant.mass_code / (3.0 * giant.mstar)) ** (1.0 / 3.0)
    width = width_multiplier * hill_factor
    if giant.e > hill_factor:
        r_in = giant.a * (1.0 - giant.e) * (1.0 - width)
        r_out = giant.a * (1.0 + giant.e) * (1.0 + width)
    else:
        r_in = giant.a * (1.0 - width)
        r_out = giant.a * (1.0 + width)
    return r_in, r_out


def u_to_t_over_tdep(u_r: np.ndarray) -> np.ndarray:
    return -np.log(u_r)


def plot_k_sweep_case(ax, giant: GiantParams, disk: DiskParams, args: argparse.Namespace) -> None:
    a_max = args.a_max if args.a_max is not None else giant.a
    a_start = max(args.a_min, args.da)
    a_embryo = np.arange(a_start, a_max, args.da)
    gap_r_in, gap_r_out = hill_gap_edges(giant, 1.0)
    gap_giant = GiantParams(
        giant.a,
        giant.e,
        giant.mass_mjup,
        giant.mstar,
        gap_r_in,
        gap_r_out,
    )

    k_values = parse_float_list(args.k_values)
    for k_value in k_values:
        k_disk = replace(disk, k=k_value)

        u_gap = resonance_u(a_embryo, gap_giant, k_disk, gap=True)
        ax.plot(a_embryo, u_to_t_over_tdep(u_gap), linestyle="-", linewidth=1.8, label=rf"$k={k_value:g}$")

    ax.axvline(gap_r_in, color="0.5", linestyle="--", linewidth=1.2)
    ax.axvline(gap_r_out, color="0.5", linestyle="--", linewidth=1.2)
    ax.set_xlim(0.0, giant.a)
    ax.set_xlabel("a [AU]")
    ax.set_title(rf"$M_p={giant.mass_mjup:g}M_J$; $a={giant.a:g}$ AU; $e={giant.e:g}$")
    ax.grid(True, which="both", alpha=0.3)

    model_handles = [
        Line2D([0], [0], color="0.25", linestyle="-", linewidth=1.8, label=r"empty gap, $1R_H$"),
        Line2D([0], [0], color="0.5", linestyle="--", linewidth=1.2, label="gap edge"),
    ]
    return model_handles


def damping_time_over_tdep_at_sr(
    a_embryo: np.ndarray,
    u_r: np.ndarray,
    giant: GiantParams,
    disk: DiskParams,
    args: argparse.Namespace,
) -> np.ndarray:
    t_damp = type1_eccentricity_damping_time(
        a_embryo,
        args.embryo_mass_earth,
        giant.mstar,
        disk,
        depletion_factor=u_r,
    )
    return t_damp / disk.t_dep


def plot_damping_window(
    ratio_ax,
    a_embryo: np.ndarray,
    t_sr_over_tdep: np.ndarray,
    u_r: np.ndarray,
    giant: GiantParams,
    disk: DiskParams,
    args: argparse.Namespace,
    color: str,
    linestyle,
    show_arrival: bool = True,
) -> None:
    t_damp_over_tdep = damping_time_over_tdep_at_sr(a_embryo, u_r, giant, disk, args)
    t_damp_done = t_sr_over_tdep + t_damp_over_tdep
    if show_arrival:
        ratio_ax.plot(a_embryo, t_sr_over_tdep, color=color, linestyle=linestyle, linewidth=2.0)
    ratio_ax.plot(a_embryo, t_damp_done, color=color, linestyle=linestyle, linewidth=1.4, alpha=0.35)


def plot_surfing_metric(
    surf_ax,
    a_embryo: np.ndarray,
    t_sr_over_tdep: np.ndarray,
    u_r: np.ndarray,
    giant: GiantParams,
    disk: DiskParams,
    args: argparse.Namespace,
    color,
) -> None:
    t_sr = t_sr_over_tdep * disk.t_dep
    r_in, _ = gap_edges(giant)
    finite = (a_embryo < r_in) & np.isfinite(a_embryo) & np.isfinite(t_sr) & np.isfinite(u_r)
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))

    for start, stop in zip(edges[0::2], edges[1::2]):
        if stop - start < 3:
            continue

        a_segment = a_embryo[start:stop]
        t_segment = t_sr[start:stop]
        u_segment = u_r[start:stop]
        dt_da = np.gradient(t_segment, a_segment)

        circ_speed = circularization_migration_speed(
            a_segment,
            args.surfing_eccentricity,
            args.embryo_mass_earth,
            giant.mstar,
            disk,
            depletion_factor=u_segment,
        )
        inv_circ_speed = np.divide(
            1.0,
            np.abs(circ_speed),
            out=np.full_like(circ_speed, np.nan),
            where=circ_speed != 0.0,
        )
        surf_ax.plot(a_segment, np.abs(dt_da), color=color, linestyle="-", linewidth=1.9)
        surf_ax.plot(a_segment, inv_circ_speed, color=color, linestyle="--", linewidth=1.7)


def plot_surfing_case(
    surf_axes,
    giant: GiantParams,
    disk: DiskParams,
    args: argparse.Namespace,
) -> tuple[list[Line2D], list[Line2D]]:
    surfing_a_giant_values = parse_float_list(args.surfing_a_giant_values)
    surfing_gap_width = 1.0
    surfing_t_dep_rows = parse_float_list(args.surfing_t_dep_values)
    max_surf_r_in = max(
        hill_gap_edges(
            GiantParams(a_giant_value, giant.e, giant.mass_mjup, giant.mstar),
            surfing_gap_width,
        )[0]
        for a_giant_value in surfing_a_giant_values
    )
    a_start = max(args.a_min, args.da)
    a_embryo = np.arange(a_start, max(args.a_max, max_surf_r_in), args.da)
    surfing_cmap = plt.get_cmap("viridis_r")
    surfing_norm = plt.Normalize(min(surfing_a_giant_values), max(surfing_a_giant_values))

    for surf_ax, surf_t_dep in zip(surf_axes, surfing_t_dep_rows):
        surf_xmax = 0.0
        for a_giant_value in surfing_a_giant_values:
            color = surfing_cmap(surfing_norm(a_giant_value))
            surf_base_giant = GiantParams(
                a_giant_value,
                giant.e,
                giant.mass_mjup,
                giant.mstar,
            )
            surf_r_in, surf_r_out = hill_gap_edges(surf_base_giant, surfing_gap_width)
            surf_xmax = max(surf_xmax, surf_r_in)
            surf_giant = GiantParams(
                surf_base_giant.a,
                surf_base_giant.e,
                surf_base_giant.mass_mjup,
                surf_base_giant.mstar,
                surf_r_in,
                surf_r_out,
            )
            surfing_disk = replace(disk, k=1.5, t_dep=surf_t_dep, k_gap_dep=args.surfing_k_gap_dep)
            u_surf_gap = resonance_u(a_embryo, surf_giant, surfing_disk, gap=True, gap_depletion=surfing_disk.k_gap_dep)
            t_surf_gap = u_to_t_over_tdep(u_surf_gap)
            plot_surfing_metric(
                surf_ax,
                a_embryo,
                t_surf_gap,
                u_surf_gap,
                surf_giant,
                surfing_disk,
                args,
                color,
            )
        surf_ax.set_xlim(0.0, surf_xmax)
        surf_ax.set_yscale("log")
        surf_ax.set_xlabel("a [AU]")
        surf_ax.set_title(
            rf"$M_p={giant.mass_mjup:g}M_J$; $e={giant.e:g}$; gap=$1R_H$; "
            rf"$k=1.5$; $T_{{dep}}={format_time_yr(surf_t_dep)}$"
        )
        surf_ax.grid(True, which="both", alpha=0.3)

    migration_handles = [
        Line2D([0], [0], color="0.25", linestyle="-", linewidth=1.9, label=r"$|dt_{\rm SR}/da|$"),
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle="--",
            linewidth=1.7,
            label=rf"$|1/\dot a_{{\rm circ}}|$, $e={args.surfing_eccentricity:g}$",
        ),
    ]
    surfing_a_giant_handles = [
        Line2D(
            [0],
            [0],
            color=surfing_cmap(surfing_norm(a_giant_value)),
            linewidth=2.0,
            label=rf"$a_J={a_giant_value:g}$ AU",
        )
        for a_giant_value in surfing_a_giant_values
    ]
    return migration_handles, surfing_a_giant_handles


def plot_case(ax, ratio_ax, giant: GiantParams, disk: DiskParams, args: argparse.Namespace) -> None:
    r_in, r_out = gap_edges(giant)
    base_a_max = args.a_max if args.a_max is not None else r_out
    a_max = base_a_max
    x_max = r_in
    a_start = max(args.a_min, args.da)
    a_embryo = np.arange(a_start, a_max, args.da)

    u_no_gap = resonance_u(a_embryo, giant, disk, gap=False)
    t_no_gap = u_to_t_over_tdep(u_no_gap)
    ax.plot(a_embryo, t_no_gap, color="C0", linewidth=2.0)
    plot_damping_window(ratio_ax, a_embryo, t_no_gap, u_no_gap, giant, disk, args, "C0", "-")

    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    gap_widths = parse_gap_widths(args.gap_widths)
    for width, linestyle in zip(gap_widths, linestyles):
        width_r_in, width_r_out = hill_gap_edges(giant, width)
        gap_giant = GiantParams(
            giant.a,
            giant.e,
            giant.mass_mjup,
            giant.mstar,
            width_r_in,
            width_r_out,
        )
        u_gap = resonance_u(a_embryo, gap_giant, disk, gap=True)
        t_gap = u_to_t_over_tdep(u_gap)
        ax.plot(a_embryo, t_gap, color="C1", linestyle=linestyle)
        plot_damping_window(ratio_ax, a_embryo, t_gap, u_gap, gap_giant, disk, args, "C1", linestyle)

        u_dep_gap = resonance_u(a_embryo, gap_giant, disk, gap=True, gap_depletion=disk.k_gap_dep)
        t_dep_gap = u_to_t_over_tdep(u_dep_gap)
        ax.plot(a_embryo, t_dep_gap, color="C2", linestyle=linestyle)
        plot_damping_window(
            ratio_ax,
            a_embryo,
            t_dep_gap,
            u_dep_gap,
            gap_giant,
            disk,
            args,
            "C2",
            linestyle,
    )

    if args.a_min < r_in < x_max:
        ax.axvline(r_in, color="0.4", linestyle=":", linewidth=1.0)
    if args.a_min < r_out < x_max:
        ax.axvline(r_out, color="0.6", linestyle=":", linewidth=1.0)

    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("a [AU]")
    ax.set_title(rf"$M_p={giant.mass_mjup:g}M_J$; $a={giant.a:g}$ AU; $e={giant.e:g}$")
    ax.grid(True, which="both", alpha=0.3)
    ratio_ax.set_xlim(0.0, x_max)
    ratio_ax.set_xlabel("a [AU]")
    ratio_ax.set_title(rf"$M_p={giant.mass_mjup:g}M_J$; $a={giant.a:g}$ AU; $e={giant.e:g}$")
    ratio_ax.grid(True, which="both", alpha=0.3)
    scenario_handles = [
        Line2D([0], [0], color="C0", linewidth=2.0, label="no gap"),
        Line2D([0], [0], color="C1", linewidth=2.0, label="empty gap"),
        Line2D([0], [0], color="C2", linewidth=2.0, label=rf"depleted gap, $k_{{gap}}={disk.k_gap_dep:g}$"),
    ]
    width_handles = [
        Line2D([0], [0], color="0.25", linewidth=2.0, linestyle=linestyle, label=rf"${width:g}R_H$")
        for width, linestyle in zip(gap_widths, linestyles)
    ]
    timing_handles = [
        Line2D([0], [0], color="0.25", linewidth=2.0, label=r"$t_{\rm SR}$"),
        Line2D([0], [0], color="0.25", linewidth=1.4, alpha=0.35, label=r"$t_{\rm SR}+T_{\rm damp,t}$"),
    ]
    print(f"e={giant.e:g}: 1 Hill-radius gap edges: R_in={r_in:.6g} AU, R_out={r_out:.6g} AU")
    print(f"e={giant.e:g}: 1 Hill-radius gap width: {r_out - r_in:.6g} AU")
    for label, arr in [("no gap", t_no_gap)]:
        finite = np.isfinite(arr)
        if finite.any():
            print(f"e={giant.e:g} {label}: finite={finite.sum()} min={np.nanmin(arr):.6g} max={np.nanmax(arr):.6g}")
        else:
            print(f"e={giant.e:g} {label}: no finite resonance locations")

    return scenario_handles, width_handles, timing_handles


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
        args.scale_height_1au = 0.025
        args.scale_height_index = 1.25
        args.f_e = 1.0
        args.surfing_eccentricity = 0.3
        args.surfing_k_gap_dep = 0.5
        args.surfing_a_giant_values = "1,5,10,15,20,30"
        args.a_min = 0.0
        args.a_max = 10

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
        scale_height_1au=args.scale_height_1au,
        scale_height_index=args.scale_height_index,
        f_e=args.f_e,
        include_gr=not args.no_gr,
        include_gap_effect=not args.no_gap_effect,
    )

    high_e_giant = GiantParams(
        giant.a,
        0.5,
        giant.mass_mjup,
        giant.mstar,
        giant.gap_r_in,
        giant.gap_r_out,
    )
    very_high_e_giant = GiantParams(
        giant.a,
        0.8,
        giant.mass_mjup,
        giant.mstar,
        giant.gap_r_in,
        giant.gap_r_out,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True)
    ratio_fig, ratio_axes = plt.subplots(1, 2, figsize=(11, 5.8), sharey=True)
    surf_fig, surf_axes = plt.subplots(2, 3, figsize=(14, 8.8), sharey=True)
    k_fig, k_axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    scenario_handles, width_handles, timing_handles = plot_case(
        axes[0], ratio_axes[0], giant, disk, args
    )
    plot_case(axes[1], ratio_axes[1], high_e_giant, disk, args)
    migration_handles, surfing_a_giant_handles = plot_surfing_case(surf_axes[:, 0], giant, disk, args)
    plot_surfing_case(surf_axes[:, 1], high_e_giant, disk, args)
    plot_surfing_case(surf_axes[:, 2], very_high_e_giant, disk, args)
    k_model_handles = plot_k_sweep_case(k_axes[0], giant, disk, args)
    plot_k_sweep_case(k_axes[1], high_e_giant, disk, args)
    axes[0].set_ylabel(r"$t_R/t_{\rm dep}$")
    ratio_axes[0].set_ylabel(r"$t/T_{\rm dep}$")
    surf_axes[0, 0].set_ylabel(r"time per AU [yr AU$^{-1}$]")
    surf_axes[1, 0].set_ylabel(r"time per AU [yr AU$^{-1}$]")
    k_axes[0].set_ylabel(r"$t_R/T_{\rm dep}$")
    fig.legend(
        handles=scenario_handles,
        title="Disk model",
        loc="lower center",
        bbox_to_anchor=(0.36, 0.02),
        ncol=3,
        frameon=False,
    )
    fig.legend(
        handles=width_handles,
        title="Gap width",
        loc="lower center",
        bbox_to_anchor=(0.78, 0.02),
        ncol=len(width_handles),
        frameon=False,
    )
    fig.suptitle(
        rf"$\Sigma(r,t)={disk.sigma0_cgs:g}\,g\,cm^{{-2}}"
        rf"\,\exp(-t/{disk.t_dep:g}\,yr)\,(r/1AU)^{{-{disk.k:g}}}$, "
        rf"$\Sigma=0$ for $r<{disk.disk_inner_cutoff:g}$ AU; "
        rf"{model_switch_label(disk)}",
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.92))

    ratio_fig.legend(
        handles=scenario_handles,
        title="Disk model",
        loc="lower center",
        bbox_to_anchor=(0.2, 0.02),
        ncol=1,
        frameon=False,
    )
    ratio_fig.legend(
        handles=width_handles,
        title="Gap width",
        loc="lower center",
        bbox_to_anchor=(0.58, 0.02),
        ncol=len(width_handles),
        frameon=False,
    )
    ratio_fig.legend(
        handles=timing_handles,
        title="Timing",
        loc="lower center",
        bbox_to_anchor=(0.9, 0.02),
        ncol=1,
        frameon=False,
    )
    ratio_fig.suptitle(
        rf"SR arrival and one Type-I damping time: $M_i={args.embryo_mass_earth:g}M_\oplus$, "
        rf"$H={disk.scale_height_1au:g}(r/1AU)^{{{disk.scale_height_index:g}}}$ AU, "
        rf"$f_e={disk.f_e:g}$; "
        rf"{model_switch_label(disk)}",
        y=0.98,
    )
    ratio_fig.tight_layout(rect=(0, 0.23, 1, 0.92))

    surf_fig.legend(
        handles=surfing_a_giant_handles,
        title=r"$a_J$",
        loc="lower center",
        bbox_to_anchor=(0.34, 0.01),
        ncol=3,
        frameon=False,
    )
    surf_fig.legend(
        handles=migration_handles,
        title="Time-per-AU",
        loc="lower center",
        bbox_to_anchor=(0.76, 0.01),
        ncol=1,
        frameon=False,
    )
    surf_fig.subplots_adjust(left=0.06, right=0.98, bottom=0.24, top=0.88, wspace=0.08, hspace=0.38)
    surf_fig.suptitle(
        rf"Depleted gap only: $k_{{gap}}={args.surfing_k_gap_dep:g}$, "
        rf"$1R_H$ gap, $M_i={args.embryo_mass_earth:g}M_\oplus$, "
        rf"$k=1.5$, $T_{{dep}}={', '.join(format_time_yr(value) for value in parse_float_list(args.surfing_t_dep_values))}$, "
        rf"$e_{{circ}}={args.surfing_eccentricity:g}$, $f_e={disk.f_e:g}$, "
        rf"$a_J={args.surfing_a_giant_values}$ AU; {model_switch_label(disk)}",
        y=0.98,
    )

    k_fig.legend(
        handles=k_model_handles,
        title="Disk model",
        loc="lower center",
        bbox_to_anchor=(0.23, 0.01),
        ncol=1,
        frameon=False,
    )
    k_handles, k_labels = k_axes[0].get_legend_handles_labels()
    k_fig.legend(
        handles=k_handles,
        labels=k_labels,
        title="Surface density",
        loc="lower center",
        bbox_to_anchor=(0.68, 0.01),
        ncol=len(k_handles),
        frameon=False,
    )
    k_fig.suptitle(
        rf"$\Sigma(r,t)={disk.sigma0_cgs:g}\,g\,cm^{{-2}}"
        rf"\,\exp(-t/{disk.t_dep:g}\,yr)\,(r/1AU)^{{-k}}$; "
        rf"$k={args.k_values}$; {model_switch_label(disk)}",
        y=0.98,
    )
    k_fig.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.86, wspace=0.08)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    timescale_output = Path(args.timescale_output)
    timescale_output.parent.mkdir(parents=True, exist_ok=True)
    ratio_fig.savefig(timescale_output, dpi=300)
    surfing_output = Path(args.surfing_output)
    surfing_output.parent.mkdir(parents=True, exist_ok=True)
    surf_fig.savefig(surfing_output, dpi=300)
    k_sweep_output = Path(args.k_sweep_output)
    k_sweep_output.parent.mkdir(parents=True, exist_ok=True)
    k_fig.savefig(k_sweep_output, dpi=300)
    print(f"saved {output}")
    print(f"saved {timescale_output}")
    print(f"saved {surfing_output}")
    print(f"saved {k_sweep_output}")


if __name__ == "__main__":
    main()
