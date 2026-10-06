#!/usr/bin/env python
"""Measure the SR frequency-matching condition from a SimulationArchive."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rebound
from scipy.signal import savgol_filter

import ssr_single_giant as analytical


ANALYTICAL_TO_RAD_PER_YR = 2.0 * math.pi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--giant-index", type=int, default=1)
    parser.add_argument("--rocky-index", type=int, default=2)
    parser.add_argument("--sigma0-cgs", type=float, default=1700.0)
    parser.add_argument("--disk-k", type=float, default=1.5)
    parser.add_argument("--z-k", type=float, default=1.094)
    parser.add_argument("--disk-inner-cutoff", type=float, default=0.01)
    parser.add_argument("--k-gap-dep", type=float, default=0.1)
    parser.add_argument("--t-dep", type=float, default=1.0e5)
    parser.add_argument("--include-gr", type=int, choices=[0, 1], default=1)
    parser.add_argument("--n-terms", type=int, default=30)
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=101,
        help="Odd Savitzky-Golay window in archive samples for measured rates.",
    )
    return parser.parse_args()


def read_archive(args: argparse.Namespace) -> dict[str, np.ndarray]:
    archive = rebound.Simulationarchive(str(args.archive))
    time = []
    rocky_varpi = []
    giant_varpi = []
    rocky_a = []
    giant_a = []
    giant_e = []
    rocky_e = []

    for sim in archive:
        star = sim.particles[0]
        rocky = sim.particles[args.rocky_index]
        giant = sim.particles[args.giant_index]
        rocky_orbit = rocky.orbit(primary=star)
        giant_orbit = giant.orbit(primary=star)
        time.append(sim.t)
        rocky_varpi.append(rocky_orbit.pomega)
        giant_varpi.append(giant_orbit.pomega)
        rocky_a.append(rocky_orbit.a)
        giant_a.append(giant_orbit.a)
        giant_e.append(giant_orbit.e)
        rocky_e.append(rocky_orbit.e)

    return {
        "time": np.asarray(time),
        "rocky_varpi": np.unwrap(np.asarray(rocky_varpi)),
        "giant_varpi": np.unwrap(np.asarray(giant_varpi)),
        "rocky_a": np.asarray(rocky_a),
        "giant_a": np.asarray(giant_a),
        "giant_e": np.asarray(giant_e),
        "rocky_e": np.asarray(rocky_e),
    }


def measured_rate(angle: np.ndarray, time: np.ndarray, window: int) -> np.ndarray:
    if len(time) < 5:
        raise ValueError("The archive contains too few snapshots.")
    window = min(window, len(time) if len(time) % 2 == 1 else len(time) - 1)
    if window < 5:
        window = 5 if len(time) >= 5 else len(time) | 1
    if window % 2 == 0:
        window -= 1
    smoothed = savgol_filter(angle, window_length=window, polyorder=2, mode="interp")
    return np.gradient(smoothed, time)


def analytical_rates(data: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    disk = analytical.DiskParams(
        sigma0_cgs=args.sigma0_cgs,
        k=args.disk_k,
        z_k=args.z_k,
        t_dep=args.t_dep,
        disk_inner_cutoff=args.disk_inner_cutoff,
        n_terms=args.n_terms,
        k_gap_dep=args.k_gap_dep,
        include_gr=bool(args.include_gr),
        include_gap_effect=True,
    )
    rocky_rate = np.empty_like(data["time"], dtype=float)
    giant_rate = np.empty_like(data["time"], dtype=float)
    for idx, time in enumerate(data["time"]):
        giant = analytical.GiantParams(
            a=float(data["giant_a"][idx]),
            e=float(data["giant_e"][idx]),
            mass_mjup=1.0,
            mstar=1.0,
        )
        rocky_a = np.array([data["rocky_a"][idx]])
        g_rocky = analytical.planet_precession_from_giant(rocky_a, giant)[0]
        g_giant_disk = analytical.disk_precession_dep_gap(
            np.array([giant.a]), giant, disk, args.k_gap_dep
        )[0]
        g_rocky_disk = analytical.disk_precession_dep_gap(
            rocky_a, giant, disk, args.k_gap_dep
        )[0]
        if args.include_gr:
            g_rocky += analytical.gr_precession(rocky_a, 1.0)[0]
        depletion = math.exp(-float(time) / args.t_dep)
        rocky_rate[idx] = ANALYTICAL_TO_RAD_PER_YR * (g_rocky + depletion * g_rocky_disk)
        giant_rate[idx] = ANALYTICAL_TO_RAD_PER_YR * depletion * g_giant_disk
    return rocky_rate, giant_rate


def main() -> None:
    args = parse_args()
    data = read_archive(args)
    measured_rocky = measured_rate(data["rocky_varpi"], data["time"], args.smooth_window)
    measured_giant = measured_rate(data["giant_varpi"], data["time"], args.smooth_window)
    measured_delta = measured_rocky - measured_giant
    analytic_rocky, analytic_giant = analytical_rates(data, args)
    analytic_delta = analytic_rocky - analytic_giant

    output = args.output
    if output is None:
        output = args.archive.with_name(args.archive.stem + "_frequency_matching.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_yr",
                "rocky_a_au",
                "giant_a_au",
                "rocky_e",
                "measured_g_rocky_rad_per_yr",
                "measured_g_giant_rad_per_yr",
                "measured_delta_g_rad_per_yr",
                "analytical_g_rocky_rad_per_yr",
                "analytical_g_giant_rad_per_yr",
                "analytical_delta_g_rad_per_yr",
            ]
        )
        for row in zip(
            data["time"],
            data["rocky_a"],
            data["giant_a"],
            data["rocky_e"],
            measured_rocky,
            measured_giant,
            measured_delta,
            analytic_rocky,
            analytic_giant,
            analytic_delta,
        ):
            writer.writerow(row)

    time_myr = data["time"] / 1.0e6
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    axes[0].plot(time_myr, measured_rocky, color="tab:blue", label="REBOUND rocky", lw=1.2)
    axes[0].plot(time_myr, measured_giant, color="tab:orange", label="REBOUND giant", lw=1.2)
    axes[0].plot(time_myr, analytic_rocky, color="tab:blue", ls="--", label="Analytical rocky", lw=1.0)
    axes[0].plot(time_myr, analytic_giant, color="tab:orange", ls="--", label="Analytical giant", lw=1.0)
    axes[0].set_ylabel(r"$g=d\varpi/dt$ [rad yr$^{-1}$]")
    axes[0].set_title("Precession-rate comparison")
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_myr, measured_delta, color="tab:purple", label="REBOUND $\\Delta g$", lw=1.2)
    axes[1].plot(time_myr, analytic_delta, color="black", ls="--", label="Analytical $\\Delta g$", lw=1.2)
    axes[1].axhline(0.0, color="gray", lw=0.9)
    axes[1].set_xlabel("Time [Myr]")
    axes[1].set_ylabel(r"$\Delta g=g_{\rm rocky}-g_J$ [rad yr$^{-1}$]")
    axes[1].set_title("SR condition: $\\Delta g=0$")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle(args.archive.name)
    fig.savefig(output, dpi=220)
    plt.close(fig)

    crossing = np.where(np.diff(np.signbit(analytic_delta)))[0]
    print(f"snapshots: {len(data['time'])}")
    print(f"measured delta-g range: {measured_delta.min():.6e} to {measured_delta.max():.6e} rad/yr")
    print(f"analytical delta-g range: {analytic_delta.min():.6e} to {analytic_delta.max():.6e} rad/yr")
    print(f"analytical zero crossings: {len(crossing)}")
    for idx in crossing:
        print(f"  t ~ {data['time'][idx] / 1e6:.6f} Myr")
    print(f"figure: {output.resolve()}")
    print(f"data: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
