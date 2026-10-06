#!/usr/bin/env python
"""Measure isolated giant-planet and disk apsidal precession in REBOUND."""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rebound

import ssr_single_giant as analytical


REPO_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent)).resolve()
BASELINE = SourceFileLoader(
    "rebound_ssr_precession_diagnostic",
    str(REPO_DIR / "rebound+SSR+type1_baseline.py"),
).load_module()

# ssr_single_giant.py uses G=1, whose time unit is yr/(2*pi) for AU and Msun.
ANALYTICAL_TO_RAD_PER_YR = 2.0 * math.pi


@dataclass
class PrecessionResult:
    name: str
    time: np.ndarray
    varpi: np.ndarray
    eccentricity: np.ndarray
    measured_rate: float
    analytical_rate: float

    @property
    def relative_error(self) -> float:
        return (self.measured_rate - self.analytical_rate) / self.analytical_rate


def configure_simulation(dt: float) -> rebound.Simulation:
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.integrator = "whfast"
    sim.dt = dt
    sim.add(m=1.0)
    return sim


def sample_precession(
    sim: rebound.Simulation,
    particle_index: int,
    duration: float,
    sample_interval: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.arange(0.0, duration + 0.5 * sample_interval, sample_interval)
    varpi = np.empty_like(times)
    eccentricity = np.empty_like(times)

    for index, time in enumerate(times):
        sim.integrate(time, exact_finish_time=1)
        orbit = sim.particles[particle_index].orbit(primary=sim.particles[0])
        varpi[index] = orbit.pomega
        eccentricity[index] = orbit.e

    return times, np.unwrap(varpi), eccentricity


def fit_rate(time: np.ndarray, varpi: np.ndarray, trim_fraction: float) -> float:
    first = int(trim_fraction * len(time))
    return float(np.polyfit(time[first:], varpi[first:], 1)[0])


def run_giant_test(args: argparse.Namespace) -> PrecessionResult:
    sim = configure_simulation(args.dt)
    sim.add(
        m=args.giant_mass_mjup * analytical.MJUP_TO_MSUN,
        a=args.giant_a,
        e=0.0,
        primary=sim.particles[0],
    )
    sim.add(
        m=args.rocky_mass_earth * analytical.MEARTH_TO_MSUN,
        a=args.rocky_a,
        e=args.rocky_e,
        primary=sim.particles[0],
    )
    sim.move_to_com()

    time, varpi, eccentricity = sample_precession(
        sim,
        particle_index=2,
        duration=args.giant_duration,
        sample_interval=args.giant_sample_interval,
    )
    measured = fit_rate(time, varpi, args.trim_fraction)
    giant = analytical.GiantParams(
        a=args.giant_a,
        e=0.0,
        mass_mjup=args.giant_mass_mjup,
        mstar=1.0,
    )
    predicted = float(
        analytical.planet_precession_from_giant(np.array([args.rocky_a]), giant)[0]
        * ANALYTICAL_TO_RAD_PER_YR
    )
    return PrecessionResult("Giant only", time, varpi, eccentricity, measured, predicted)


def run_disk_test(args: argparse.Namespace) -> PrecessionResult:
    sim = configure_simulation(args.dt)
    sim.add(
        m=args.rocky_mass_earth * analytical.MEARTH_TO_MSUN,
        a=args.rocky_a,
        e=args.rocky_e,
        primary=sim.particles[0],
    )
    sim.move_to_com()

    disk = analytical.DiskParams(
        sigma0_cgs=args.sigma0_cgs,
        k=args.disk_k,
        z_k=args.z_k,
        t_dep=math.inf,
        disk_inner_cutoff=args.disk_inner_cutoff,
        n_terms=args.n_terms,
        include_gr=False,
        include_gap_effect=False,
    )
    coeffs = BASELINE.a_l_coefficients(args.n_terms)

    def static_disk_force(reb_sim_pointer) -> None:
        reb_sim = reb_sim_pointer.contents
        star = reb_sim.particles[0]
        rocky = reb_sim.particles[1]
        radius = math.sqrt(
            (rocky.x - star.x) ** 2
            + (rocky.y - star.y) ** 2
            + (rocky.z - star.z) ** 2
        )
        accel = BASELINE.disk_acceleration_no_gap(
            radius,
            reb_sim.G,
            disk.sigma0_code,
            disk.k,
            disk.z_k,
            0.0,
            math.inf,
            disk.disk_inner_cutoff,
            coeffs,
        )
        BASELINE.apply_radial_acceleration(rocky, star, accel)

    sim.additional_forces = static_disk_force
    time, varpi, eccentricity = sample_precession(
        sim,
        particle_index=1,
        duration=args.disk_duration,
        sample_interval=args.disk_sample_interval,
    )
    measured = fit_rate(time, varpi, args.trim_fraction)
    predicted = float(
        analytical.disk_precession_no_gap(np.array([args.rocky_a]), 1.0, disk)[0]
        * ANALYTICAL_TO_RAD_PER_YR
    )
    return PrecessionResult("Static disk only", time, varpi, eccentricity, measured, predicted)


def write_summary(results: list[PrecessionResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "test",
                "measured_rad_per_yr",
                "analytical_rad_per_yr",
                "relative_error",
                "measured_arcsec_per_yr",
                "analytical_arcsec_per_yr",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.measured_rate,
                    result.analytical_rate,
                    result.relative_error,
                    math.degrees(result.measured_rate) * 3600.0,
                    math.degrees(result.analytical_rate) * 3600.0,
                ]
            )


def plot_results(results: list[PrecessionResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(12, 4.5), constrained_layout=True)
    for axis, result in zip(np.atleast_1d(axes), results):
        elapsed_varpi = result.varpi - result.varpi[0]
        analytical_line = result.analytical_rate * result.time
        axis.plot(result.time, elapsed_varpi, color="tab:blue", lw=1.5, label="REBOUND")
        axis.plot(result.time, analytical_line, color="black", ls="--", lw=1.4, label="Analytical")
        axis.set_title(
            f"{result.name}\n"
            f"measured/predicted = {result.measured_rate / result.analytical_rate:.4f}"
        )
        axis.set_xlabel("Time [yr]")
        axis.set_ylabel(r"$\varpi(t)-\varpi(0)$ [rad]")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocky-a", type=float, default=1.0)
    parser.add_argument("--rocky-e", type=float, default=0.01)
    parser.add_argument("--rocky-mass-earth", type=float, default=1.0)
    parser.add_argument("--giant-a", type=float, default=5.0)
    parser.add_argument("--giant-mass-mjup", type=float, default=1.0)
    parser.add_argument("--sigma0-cgs", type=float, default=1700.0)
    parser.add_argument("--disk-k", type=float, default=1.5)
    parser.add_argument("--z-k", type=float, default=1.094)
    parser.add_argument("--disk-inner-cutoff", type=float, default=0.01)
    parser.add_argument("--n-terms", type=int, default=30)
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Integrator timestep [yr]; default is about 1/200 of the 1 AU orbital period.",
    )
    parser.add_argument("--giant-duration", type=float, default=1.0e5)
    parser.add_argument("--disk-duration", type=float, default=5.0e3)
    parser.add_argument("--giant-sample-interval", type=float, default=20.37)
    parser.add_argument("--disk-sample-interval", type=float, default=1.73)
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figure/precession_diagnostic.png"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("output/precession_diagnostic.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [run_giant_test(args), run_disk_test(args)]
    write_summary(results, args.summary)
    plot_results(results, args.figure)

    for result in results:
        print(
            f"{result.name}: measured={result.measured_rate:.8e} rad/yr, "
            f"analytical={result.analytical_rate:.8e} rad/yr, "
            f"relative_error={result.relative_error:+.3%}"
        )
    print(f"figure={args.figure.resolve()}")
    print(f"summary={args.summary.resolve()}")


if __name__ == "__main__":
    main()
