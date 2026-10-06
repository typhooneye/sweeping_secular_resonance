#!/usr/bin/env python
"""Measure the giant planet's apsidal precession and compare with analytics."""

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
    "rebound_ssr_giant_precession_diagnostic",
    str(REPO_DIR / "rebound+SSR+type1_baseline.py"),
).load_module()

# The analytical script uses G=1; REBOUND uses G=4*pi^2 for yr, AU, Msun.
ANALYTICAL_TO_RAD_PER_YR = 2.0 * math.pi


@dataclass
class Result:
    name: str
    time: np.ndarray
    omega: np.ndarray
    measured_rate: float
    analytical_rate: float

    @property
    def relative_error(self) -> float:
        if self.analytical_rate == 0.0:
            return math.nan
        return (self.measured_rate - self.analytical_rate) / self.analytical_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--giant-a", type=float, default=5.0)
    parser.add_argument("--giant-e", type=float, default=0.05)
    parser.add_argument("--giant-mass-mjup", type=float, default=1.0)
    parser.add_argument("--sigma0-cgs", type=float, default=1700.0)
    parser.add_argument("--disk-k", type=float, default=1.5)
    parser.add_argument("--z-k", type=float, default=1.094)
    parser.add_argument("--disk-inner-cutoff", type=float, default=0.01)
    parser.add_argument("--k-gap-dep", type=float, default=0.1)
    parser.add_argument("--n-terms", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.01, help="Integrator timestep [yr].")
    parser.add_argument(
        "--duration",
        type=float,
        default=1.5e4,
        help="Integration duration [yr]; default covers multiple giant precession cycles.",
    )
    parser.add_argument("--sample-interval", type=float, default=20.37)
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("output/giant_precession_diagnostic"))
    return parser.parse_args()


def make_disk(sigma0_cgs: float, disk_k: float, z_k: float, cutoff: float, n_terms: int):
    return analytical.DiskParams(
        sigma0_cgs=sigma0_cgs,
        k=disk_k,
        z_k=z_k,
        t_dep=math.inf,
        disk_inner_cutoff=cutoff,
        n_terms=n_terms,
        include_gr=False,
        include_gap_effect=True,
    )


def run_case(args: argparse.Namespace, case: str) -> Result:
    disk = make_disk(args.sigma0_cgs, args.disk_k, args.z_k, args.disk_inner_cutoff, args.n_terms)
    giant = analytical.GiantParams(
        a=args.giant_a,
        e=args.giant_e,
        mass_mjup=args.giant_mass_mjup,
        mstar=1.0,
    )
    coeffs = BASELINE.a_l_coefficients(args.n_terms)

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.integrator = "whfast"
    sim.dt = args.dt
    sim.add(m=1.0)
    sim.add(
        m=giant.mass_code,
        a=giant.a,
        e=giant.e,
        primary=sim.particles[0],
    )
    sim.move_to_com()
    fixed_gap_edges = analytical.gap_edges(giant)

    def additional_forces(reb_sim_pointer) -> None:
        if case == "no force":
            return

        reb_sim = reb_sim_pointer.contents
        star = reb_sim.particles[0]
        planet = reb_sim.particles[1]
        x = planet.x - star.x
        y = planet.y - star.y
        z = planet.z - star.z
        radius = math.sqrt(x * x + y * y + z * z)

        if case == "no gap":
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
        elif case == "depleted gap":
            orbit = planet.orbit(primary=star)
            r_in, r_out = fixed_gap_edges
            accel = BASELINE.disk_acceleration_gap_by_location(
                radius,
                orbit.a,
                r_in,
                r_out,
                reb_sim.G,
                disk.sigma0_code,
                disk.k,
                disk.z_k,
                0.0,
                math.inf,
                disk.disk_inner_cutoff,
                args.k_gap_dep,
                coeffs,
            )
        else:
            raise ValueError(f"Unknown case: {case}")
        BASELINE.apply_radial_acceleration(planet, star, accel)

    if case != "no force":
        sim.additional_forces = additional_forces

    times = np.arange(0.0, args.duration + 0.5 * args.sample_interval, args.sample_interval)
    omega = np.empty_like(times)
    for index, time in enumerate(times):
        sim.integrate(time, exact_finish_time=1)
        omega[index] = sim.particles[1].orbit(primary=sim.particles[0]).pomega
    omega = np.unwrap(omega)

    start = int(args.trim_fraction * len(times))
    measured = float(np.polyfit(times[start:], omega[start:], 1)[0])
    if case == "no force":
        predicted = 0.0
    elif case == "no gap":
        predicted = float(
            analytical.disk_precession_no_gap(np.array([args.giant_a]), 1.0, disk)[0]
            * ANALYTICAL_TO_RAD_PER_YR
        )
    else:
        predicted = float(
            analytical.disk_precession_dep_gap(
                np.array([args.giant_a]), giant, disk, args.k_gap_dep
            )[0]
            * ANALYTICAL_TO_RAD_PER_YR
        )
    return Result(case, times, omega, measured, predicted)


def main() -> None:
    args = parse_args()
    cases = ["no force", "no gap", "depleted gap"]
    results = [run_case(args, case) for case in cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.output_dir / "giant_precession_rates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "measured_rad_per_yr", "analytical_rad_per_yr", "relative_error"])
        for result in results:
            writer.writerow(
                [result.name, result.measured_rate, result.analytical_rate, result.relative_error]
            )
            np.savetxt(
                args.output_dir / f"{result.name.replace(' ', '_')}_omega.csv",
                np.column_stack((result.time, result.omega, result.omega - result.omega[0])),
                delimiter=",",
                header="time_yr,omega_unwrapped_rad,delta_omega_rad",
                comments="",
            )
            print(
                f"{result.name}: measured={result.measured_rate:.8e} rad/yr, "
                f"analytical={result.analytical_rate:.8e} rad/yr, "
                f"relative_error={result.relative_error:+.3%}"
            )

    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    for result in results:
        delta_omega = result.omega - result.omega[0]
        axis.plot(result.time, delta_omega, lw=1.3, label=result.name)
        if result.name != "no force":
            axis.plot(
                result.time,
                result.analytical_rate * result.time,
                "--",
                lw=1.0,
                label=f"{result.name} analytical",
            )
    axis.set_xlabel("Time [yr]")
    axis.set_ylabel(r"$\omega(t)-\omega(0)$ [rad]")
    axis.set_title(
        f"Giant apsidal precession: a={args.giant_a:g} AU, e={args.giant_e:g}, "
        f"dt={args.dt:g} yr"
    )
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(args.output_dir / "giant_precession_omega.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
