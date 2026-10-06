#!/usr/bin/env python

#SBATCH --job-name=diagnose-
#SBATCH --output=logs/diagnose-_%j.out
#SBATCH --error=logs/diagnose-_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

"""Decompose the rocky planet's apsidal precession into four REBOUND tests.

Cases:
    1. rocky + giant
    2. rocky + no-gap disk
    3. rocky + giant + depleted giant-opened gap
    4. rocky + GR

Each case is run independently in parallel, saved as a SimulationArchive,
and compared with the corresponding analytical omega(t) prediction.
"""



from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import os
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
import os
import sys


# sbatch copies the submitted script into a spool directory and executes that
# copy, so __file__'s directory (and a plain `import ssr_single_giant`) would
# otherwise resolve to the spool dir, not the repo. SLURM_SUBMIT_DIR is set by
# Slurm to the real submission directory; fall back to __file__ when run
# directly. Same pattern as submit_rebound_sr_quick_test.py.
REPO_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent)).resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import matplotlib.pyplot as plt
import numpy as np
import rebound

import ssr_single_giant as analytical


REPO_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent)).resolve()
BASELINE = SourceFileLoader(
    "rebound_ssr_component_diagnostic",
    str(REPO_DIR / "rebound+SSR+type1_baseline.py"),
).load_module()

# The analytical script uses G=1; REBOUND uses G=4*pi^2 in (yr, AU, Msun).
ANALYTICAL_TO_RAD_PER_YR = 2.0 * math.pi
REBOUND_C_CODE = 63241.077


@dataclass(frozen=True)
class Case:
    name: str
    filename: str


CASES = (
    Case("rocky + giant", "rocky_giant"),
    Case("rocky + no-gap disk", "rocky_no_gap_disk"),
    Case("rocky + giant + depleted gap", "rocky_giant_depleted_gap"),
    Case("rocky + GR", "rocky_gr"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocky-a", type=float, default=1.0)
    parser.add_argument("--rocky-e", type=float, default=0.01)
    parser.add_argument("--rocky-mass-earth", type=float, default=1.0)
    parser.add_argument("--giant-a", type=float, default=5.0)
    parser.add_argument("--giant-e", type=float, default=0.05)
    parser.add_argument("--giant-mass-mjup", type=float, default=1.0)
    parser.add_argument("--sigma0-cgs", type=float, default=1700.0)
    parser.add_argument("--disk-k", type=float, default=1.5)
    parser.add_argument("--z-k", type=float, default=1.094)
    parser.add_argument("--disk-inner-cutoff", type=float, default=0.01)
    parser.add_argument("--k-gap-dep", type=float, default=0.1)
    parser.add_argument("--t-dep", type=float, default=1.0e3)
    parser.add_argument("--duration", type=float, default=2.0e4)
    parser.add_argument("--dt-fraction", type=float, default=0.01)
    parser.add_argument("--dt-output", type=float, default=10.0)
    parser.add_argument("--n-terms", type=int, default=30)
    parser.add_argument("--ncpus", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rocky_precession_components"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figure/rocky_precession_components.png"),
    )
    return parser.parse_args()


def disk_parameters(args: argparse.Namespace) -> analytical.DiskParams:
    return analytical.DiskParams(
        sigma0_cgs=args.sigma0_cgs,
        k=args.disk_k,
        z_k=args.z_k,
        t_dep=args.t_dep,
        disk_inner_cutoff=args.disk_inner_cutoff,
        n_terms=args.n_terms,
        k_gap_dep=args.k_gap_dep,
        include_gr=False,
        include_gap_effect=True,
    )


def build_analytical_prediction(case: Case, args: argparse.Namespace) -> tuple[float, float]:
    disk = disk_parameters(args)
    giant = analytical.GiantParams(
        a=args.giant_a,
        e=args.giant_e,
        mass_mjup=args.giant_mass_mjup,
        mstar=1.0,
    )
    a = np.array([args.rocky_a])

    giant_rate = 0.0
    disk_rate = 0.0
    gr_rate = 0.0
    if case.filename in {"rocky_giant", "rocky_giant_depleted_gap"}:
        giant_rate = float(analytical.planet_precession_from_giant(a, giant)[0])
    if case.filename == "rocky_no_gap_disk":
        disk_rate = float(analytical.disk_precession_no_gap(a, 1.0, disk)[0])
    elif case.filename == "rocky_giant_depleted_gap":
        disk_rate = float(
            analytical.disk_precession_dep_gap(a, giant, disk, args.k_gap_dep)[0]
        )
    if case.filename == "rocky_gr":
        gr_rate = float(analytical.gr_precession(a, 1.0)[0])

    initial_rate = (giant_rate + gr_rate) * ANALYTICAL_TO_RAD_PER_YR
    initial_disk_rate = disk_rate * ANALYTICAL_TO_RAD_PER_YR
    return initial_rate, initial_disk_rate


def run_case(case: Case, args: argparse.Namespace) -> dict[str, object]:
    disk = disk_parameters(args)
    giant_params = analytical.GiantParams(
        a=args.giant_a,
        e=args.giant_e,
        mass_mjup=args.giant_mass_mjup,
        mstar=1.0,
    )
    coeffs = BASELINE.a_l_coefficients(args.n_terms)
    has_giant = case.filename in {"rocky_giant", "rocky_giant_depleted_gap"}
    has_disk = case.filename in {"rocky_no_gap_disk", "rocky_giant_depleted_gap"}
    has_gr = case.filename == "rocky_gr"

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.integrator = "mercurius"
    sim.add(m=1.0)
    if has_giant:
        sim.add(
            m=giant_params.mass_code,
            a=args.giant_a,
            e=args.giant_e,
            primary=sim.particles[0],
        )
    rocky_index = len(sim.particles)
    sim.add(
        m=args.rocky_mass_earth * analytical.MEARTH_TO_MSUN,
        a=args.rocky_a,
        e=args.rocky_e,
        primary=sim.particles[0],
    )
    sim.move_to_com()
    sim.dt = sim.particles[rocky_index].P * args.dt_fraction
    sim.force_is_velocity_dependent = int(has_gr)
    fixed_gap_edges = analytical.gap_edges(giant_params) if has_giant else None

    def additional_forces(reb_sim_pointer) -> None:
        reb_sim = reb_sim_pointer.contents
        star = reb_sim.particles[0]
        rocky = reb_sim.particles[rocky_index]

        if has_disk:
            disk_particles = [rocky]
            if has_giant:
                disk_particles.append(reb_sim.particles[1])
            for particle in disk_particles:
                x = particle.x - star.x
                y = particle.y - star.y
                z = particle.z - star.z
                radius = math.sqrt(x * x + y * y + z * z)
                if fixed_gap_edges is None:
                    accel = BASELINE.disk_acceleration_no_gap(
                        radius,
                        reb_sim.G,
                        disk.sigma0_code,
                        disk.k,
                        disk.z_k,
                        reb_sim.t,
                        disk.t_dep,
                        disk.disk_inner_cutoff,
                        coeffs,
                    )
                else:
                    r_in, r_out = fixed_gap_edges
                    orbit = particle.orbit(primary=star)
                    accel = BASELINE.disk_acceleration_gap_by_location(
                        radius,
                        orbit.a,
                        r_in,
                        r_out,
                        reb_sim.G,
                        disk.sigma0_code,
                        disk.k,
                        disk.z_k,
                        reb_sim.t,
                        disk.t_dep,
                        disk.disk_inner_cutoff,
                        args.k_gap_dep,
                        coeffs,
                    )
                BASELINE.apply_radial_acceleration(particle, star, accel)

        if has_gr:
            BASELINE.apply_gr_acceleration(rocky, star, reb_sim.G, REBOUND_C_CODE)

    if has_disk or has_gr:
        sim.additional_forces = additional_forces

    archive_path = args.output_dir / f"{case.filename}.bin"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sim.save_to_file(str(archive_path), interval=args.dt_output, delete_file=True)

    times = np.arange(0.0, args.duration + 0.5 * args.dt_output, args.dt_output)
    omega_raw = np.empty_like(times)
    eccentricity = np.empty_like(times)
    semimajor_axis = np.empty_like(times)
    for index, time in enumerate(times):
        sim.integrate(time, exact_finish_time=1)
        orbit = sim.particles[rocky_index].orbit(primary=sim.particles[0])
        omega_raw[index] = orbit.pomega
        eccentricity[index] = orbit.e
        semimajor_axis[index] = orbit.a
    # REBOUND reports pomega modulo 2*pi.  Unwrap before fitting or
    # comparing with the accumulated analytical precession.
    omega = np.unwrap(omega_raw)

    giant_rate, disk_rate = build_analytical_prediction(case, args)
    analytical_omega = (
        omega[0]
        + giant_rate * times
        + disk_rate * args.t_dep * (1.0 - np.exp(-times / args.t_dep))
    )
    gr_rate = 0.0
    if case.filename == "rocky_gr":
        gr_rate = float(analytical.gr_precession(np.array([args.rocky_a]), 1.0)[0]) * ANALYTICAL_TO_RAD_PER_YR
        analytical_omega = omega[0] + gr_rate * times

    first = max(1, int(0.05 * len(times)))
    measured_rate = float(np.polyfit(times[first:], omega[first:], 1)[0])
    fit_rate = float(np.polyfit(times[first:], analytical_omega[first:], 1)[0])

    csv_path = args.output_dir / f"{case.filename}_omega.csv"
    np.savetxt(
        csv_path,
        np.column_stack((times, omega_raw, omega, analytical_omega, eccentricity, semimajor_axis)),
        delimiter=",",
        header="time_yr,omega_rebound_raw_rad,omega_rebound_unwrapped_rad,omega_analytical_rad,eccentricity,semimajor_axis_au",
        comments="",
    )
    return {
        "case": case,
        "archive": archive_path,
        "csv": csv_path,
        "time": times,
        "omega": omega,
        "analytical_omega": analytical_omega,
        "measured_rate": measured_rate,
        "analytical_rate": fit_rate,
    }


def plot_results(results: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False, constrained_layout=True)
    for axis, result in zip(axes.flat, results):
        case = result["case"]
        time = np.asarray(result["time"])
        omega = np.asarray(result["omega"])
        analytical_omega = np.asarray(result["analytical_omega"])
        axis.plot(time, omega - omega[0], label="REBOUND", lw=1.2)
        axis.plot(time, analytical_omega - analytical_omega[0], "k--", label="Analytical", lw=1.2)
        ratio = float(result["measured_rate"]) / float(result["analytical_rate"]) if result["analytical_rate"] else math.nan
        axis.set_title(f"{case.name}\nmeasured / analytical = {ratio:.4f}")
        axis.set_xlabel("Time [yr]")
        axis.set_ylabel(r"$\omega(t)-\omega(0)$ [rad]")
        axis.grid(alpha=0.25)
        axis.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ncpus = args.ncpus or min(multiprocessing.cpu_count(), len(CASES))
    ncpus = min(ncpus, len(CASES))
    print(f"running {len(CASES)} component tests with {ncpus} workers")
    print(f"output_dir={args.output_dir.resolve()}")

    with multiprocessing.Pool(ncpus) as pool:
        results = pool.starmap(run_case, [(case, args) for case in CASES])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "precession_component_rates.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case", "measured_rad_per_yr", "analytical_rad_per_yr", "relative_error"])
        for result in results:
            analytical_rate = float(result["analytical_rate"])
            measured_rate = float(result["measured_rate"])
            relative_error = (
                (measured_rate - analytical_rate) / analytical_rate
                if analytical_rate != 0.0
                else math.nan
            )
            writer.writerow([result["case"].name, measured_rate, analytical_rate, relative_error])
            print(
                f"{result['case'].name}: measured={measured_rate:.8e}, "
                f"analytical={analytical_rate:.8e}, relative_error={relative_error:+.3%}"
            )

    plot_results(results, args.figure)
    print(f"figure={args.figure.resolve()}")


if __name__ == "__main__":
    main()
