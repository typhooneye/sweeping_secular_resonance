#!/usr/bin/env python

#SBATCH --job-name=precession-force-tests
#SBATCH --output=logs/precession-force-tests_%j.out
#SBATCH --error=logs/precession-force-tests_%j.err
#SBATCH --account=pi-abbot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11

"""Run paired REBOUND tests of every conservative SSR precession term.

Submit the default suite with

    sbatch precession_force_experiments.py

Every experiment writes a REBOUND SimulationArchive (``.bin``).  After all
runs finish, the same script reads those archives, measures apsidal rates from
the eccentricity-vector phase, and compares control-subtracted rates with
``ssr_single_giant.py``.

The disk is static and Type-I forces are absent by construction.  This keeps
the measured slope attributable to conservative gravity, disk, or GR terms.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path


REPO_DIR = Path(
    os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent)
).resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rebound

import ssr_single_giant as analytical


BASELINE = SourceFileLoader(
    "rebound_ssr_precession_force_tests",
    str(REPO_DIR / "rebound+SSR+type1_baseline.py"),
).load_module()

ANALYTICAL_TO_RAD_PER_YR = 2.0 * math.pi
SIGMA_CGS_TO_CODE = analytical.G_PER_CM2_TO_MSUN_PER_AU2


@dataclass(frozen=True)
class Experiment:
    name: str
    rocky: bool
    giant: bool
    giant_e: float
    disk_mode: str = "none"  # none, no_gap, empty_gap, depleted_gap
    include_gr: bool = False
    duration_group: str = "disk"  # disk, gravity, gr


EXPERIMENTS = (
    Experiment("rocky_control", True, False, 0.0, duration_group="gr"),
    Experiment("rocky_giant", True, True, 0.0, duration_group="gravity"),
    Experiment("rocky_disk_no_gap", True, False, 0.0, "no_gap"),
    Experiment("giant_control", False, True, 0.05),
    Experiment("giant_disk_empty", False, True, 0.05, "empty_gap"),
    Experiment("giant_disk_depleted", False, True, 0.05, "depleted_gap"),
    Experiment("pair_control", True, True, 0.05),
    Experiment("pair_disk_empty", True, True, 0.05, "empty_gap"),
    Experiment("pair_disk_depleted", True, True, 0.05, "depleted_gap"),
    Experiment("rocky_gr", True, False, 0.0, include_gr=True, duration_group="gr"),
    Experiment(
        "pair_full_depleted_gr",
        True,
        True,
        0.05,
        "depleted_gap",
        include_gr=True,
    ),
)


def parse_slurm_cpus(value: str | None) -> int:
    """Parse values such as '6', '6(x2)', or '6,5,2'."""
    if not value:
        return multiprocessing.cpu_count()
    total = 0
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "(x" in item:
            cpus, repeat = item.split("(x", 1)
            total += int(cpus) * int(repeat.rstrip(")"))
        else:
            total += int(item)
    return total or multiprocessing.cpu_count()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "run", "analyze"), default="all")
    parser.add_argument("--case", choices=[item.name for item in EXPERIMENTS])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/precession_force_tests"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figure/precession_force_tests"),
    )
    parser.add_argument("--ncpus", type=int)

    parser.add_argument("--mstar", type=float, default=1.0)
    parser.add_argument("--rocky-a", type=float, default=1.0)
    parser.add_argument("--rocky-e", type=float, default=0.02)
    parser.add_argument("--giant-a", type=float, default=5.0)
    parser.add_argument("--giant-e", type=float, default=0.05)
    parser.add_argument("--giant-mass-mjup", type=float, default=1.0)

    parser.add_argument("--sigma0-cgs", type=float, default=1700.0)
    parser.add_argument("--disk-k", type=float, default=1.5)
    parser.add_argument("--z-k", type=float, default=1.094)
    parser.add_argument("--disk-inner-cutoff", type=float, default=0.01)
    parser.add_argument("--k-gap-dep", type=float, default=0.1)
    parser.add_argument("--gap-hill-width", type=float, default=1.0)
    parser.add_argument("--n-terms", type=int, default=30)
    parser.add_argument("--c-code", type=float, default=63241.077)

    parser.add_argument("--dt-fraction", type=float, default=0.01)
    parser.add_argument("--softening", type=float, default=0.0)
    parser.add_argument("--gravity-duration", type=float, default=5.0e4)
    parser.add_argument("--disk-duration", type=float, default=2.0e4)
    parser.add_argument("--gr-duration", type=float, default=2.0e5)
    parser.add_argument("--archive-samples", type=int, default=4000)
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    parser.add_argument("--fit-blocks", type=int, default=5)
    return parser.parse_args()


def selected_experiments(args: argparse.Namespace) -> list[Experiment]:
    if args.case:
        return [item for item in EXPERIMENTS if item.name == args.case]
    return list(EXPERIMENTS)


def duration_for(experiment: Experiment, args: argparse.Namespace) -> float:
    return float(getattr(args, f"{experiment.duration_group}_duration"))


def gap_edges(a: float, e: float, giant_mass: float, mstar: float, width: float) -> tuple[float, float]:
    hill = (giant_mass / (3.0 * mstar)) ** (1.0 / 3.0)
    scaled_hill = width * hill
    if e > hill:
        return (
            a * (1.0 - e) * (1.0 - scaled_hill),
            a * (1.0 + e) * (1.0 + scaled_hill),
        )
    return a * (1.0 - scaled_hill), a * (1.0 + scaled_hill)


def run_experiment(experiment: Experiment, options: dict[str, object]) -> str:
    args = argparse.Namespace(**options)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{experiment.name}.bin"
    metadata_path = output_dir / f"{experiment.name}.json"

    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.integrator = "mercurius"
    sim.softening = args.softening
    sim.collision = "none"
    sim.add(m=args.mstar, hash="star")

    giant_mass = args.giant_mass_mjup * analytical.MJUP_TO_MSUN
    if experiment.giant:
        giant_e = args.giant_e if experiment.giant_e != 0.0 else 0.0
        sim.add(
            m=giant_mass,
            a=args.giant_a,
            e=giant_e,
            omega=0.0,
            f=0.0,
            primary=sim.particles[0],
            hash="giant",
        )
    if experiment.rocky:
        sim.add(
            m=0.0,
            a=args.rocky_a,
            e=args.rocky_e,
            omega=0.0,
            f=math.pi,
            primary=sim.particles[0],
            hash="rocky",
        )

    sim.N_active = sim.N
    sim.move_to_com()
    orbital_periods = [particle.P for particle in sim.particles[1:]]
    sim.dt = min(orbital_periods) * args.dt_fraction

    coeffs = BASELINE.a_l_coefficients(args.n_terms)
    dens0 = args.sigma0_cgs * SIGMA_CGS_TO_CODE
    fixed_edges = None
    if experiment.disk_mode in {"empty_gap", "depleted_gap"}:
        fixed_edges = gap_edges(
            args.giant_a,
            args.giant_e if experiment.giant_e != 0.0 else 0.0,
            giant_mass,
            args.mstar,
            args.gap_hill_width,
        )

    def additional_forces(reb_sim_pointer) -> None:
        reb_sim = reb_sim_pointer.contents
        star = reb_sim.particles["star"]

        for particle in reb_sim.particles[1:]:
            x = particle.x - star.x
            y = particle.y - star.y
            z = particle.z - star.z
            radius = math.sqrt(x * x + y * y + z * z)

            if experiment.disk_mode == "no_gap":
                acceleration = BASELINE.disk_acceleration_no_gap(
                    radius,
                    reb_sim.G,
                    dens0,
                    args.disk_k,
                    args.z_k,
                    0.0,
                    math.inf,
                    args.disk_inner_cutoff,
                    coeffs,
                )
                BASELINE.apply_radial_acceleration(particle, star, acceleration)
            elif experiment.disk_mode in {"empty_gap", "depleted_gap"}:
                r_in, r_out = fixed_edges
                gap_fraction = 0.0 if experiment.disk_mode == "empty_gap" else args.k_gap_dep
                orbit = particle.orbit(primary=star)
                acceleration = BASELINE.disk_acceleration_gap_by_location(
                    radius,
                    orbit.a,
                    r_in,
                    r_out,
                    reb_sim.G,
                    dens0,
                    args.disk_k,
                    args.z_k,
                    0.0,
                    math.inf,
                    args.disk_inner_cutoff,
                    gap_fraction,
                    coeffs,
                )
                BASELINE.apply_radial_acceleration(particle, star, acceleration)

        if experiment.include_gr and experiment.rocky:
            BASELINE.apply_gr_acceleration(
                reb_sim.particles["rocky"],
                star,
                reb_sim.G,
                args.c_code,
            )

    if experiment.disk_mode != "none" or experiment.include_gr:
        sim.additional_forces = additional_forces
        sim.force_is_velocity_dependent = int(experiment.include_gr)

    duration = duration_for(experiment, args)
    archive_interval = duration / args.archive_samples
    sim.save_to_file(str(archive_path), interval=archive_interval, delete_file=True)

    metadata = {
        "experiment": asdict(experiment),
        "archive": str(archive_path),
        "duration_yr": duration,
        "archive_interval_yr": archive_interval,
        "particle_indices": {
            "giant": 1 if experiment.giant else None,
            "rocky": (2 if experiment.giant else 1) if experiment.rocky else None,
        },
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    sim.integrate(duration, exact_finish_time=1)
    return str(archive_path)


@dataclass
class OrbitSeries:
    time: np.ndarray
    a: np.ndarray
    e: np.ndarray
    varpi: np.ndarray


def read_orbit_series(path: Path, particle_index: int) -> OrbitSeries:
    archive = rebound.Simulationarchive(str(path))
    time: list[float] = []
    semimajor_axis: list[float] = []
    eccentricity: list[float] = []
    varpi: list[float] = []
    for snapshot in archive:
        orbit = snapshot.particles[particle_index].orbit(primary=snapshot.particles[0])
        time.append(snapshot.t)
        semimajor_axis.append(orbit.a)
        eccentricity.append(orbit.e)
        varpi.append(orbit.pomega)
    return OrbitSeries(
        np.asarray(time),
        np.asarray(semimajor_axis),
        np.asarray(eccentricity),
        np.unwrap(np.asarray(varpi)),
    )


def fit_rate(series: OrbitSeries, tmax: float, trim_fraction: float, blocks: int) -> tuple[float, float]:
    selected = series.time <= tmax * (1.0 + 1.0e-12)
    time = series.time[selected]
    phase = series.varpi[selected]
    first = int(trim_fraction * len(time))
    time = time[first:]
    phase = phase[first:]
    if len(time) < max(20, 4 * blocks):
        raise ValueError("Too few archive snapshots for a reliable phase fit.")
    rate = float(np.polyfit(time, phase, 1)[0])
    block_rates = []
    for indices in np.array_split(np.arange(len(time)), blocks):
        if len(indices) >= 4:
            block_rates.append(float(np.polyfit(time[indices], phase[indices], 1)[0]))
    uncertainty = float(np.std(block_rates, ddof=1)) if len(block_rates) > 1 else math.nan
    return rate, uncertainty


def analytical_setup(args: argparse.Namespace) -> tuple[analytical.DiskParams, analytical.GiantParams]:
    disk = analytical.DiskParams(
        sigma0_cgs=args.sigma0_cgs,
        k=args.disk_k,
        z_k=args.z_k,
        t_dep=math.inf,
        disk_inner_cutoff=args.disk_inner_cutoff,
        n_terms=args.n_terms,
        k_gap_dep=args.k_gap_dep,
        include_gr=True,
        include_gap_effect=True,
    )
    giant_mass = args.giant_mass_mjup * analytical.MJUP_TO_MSUN
    r_in, r_out = gap_edges(
        args.giant_a,
        args.giant_e,
        giant_mass,
        args.mstar,
        args.gap_hill_width,
    )
    giant = analytical.GiantParams(
        a=args.giant_a,
        e=args.giant_e,
        mass_mjup=args.giant_mass_mjup,
        mstar=args.mstar,
        gap_r_in=r_in,
        gap_r_out=r_out,
    )
    return disk, giant


def analyze_suite(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    specs = {item.name: item for item in EXPERIMENTS}
    series: dict[tuple[str, str], OrbitSeries] = {}
    metadata: dict[str, dict[str, object]] = {}

    for name, spec in specs.items():
        archive_path = output_dir / f"{name}.bin"
        metadata_path = output_dir / f"{name}.json"
        if not archive_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Missing archive or metadata for {name}")
        metadata[name] = json.loads(metadata_path.read_text())
        indices = metadata[name]["particle_indices"]
        for body in ("rocky", "giant"):
            index = indices[body]
            if index is not None:
                body_series = read_orbit_series(archive_path, int(index))
                series[(name, body)] = body_series
                np.savetxt(
                    output_dir / f"{name}_{body}_orbit.csv",
                    np.column_stack(
                        (
                            body_series.time,
                            body_series.a,
                            body_series.e,
                            body_series.varpi,
                            body_series.e * np.cos(body_series.varpi),
                            body_series.e * np.sin(body_series.varpi),
                        )
                    ),
                    delimiter=",",
                    header="time_yr,a_au,e,varpi_unwrapped_rad,e_cos_varpi,e_sin_varpi",
                    comments="",
                )

    rates: dict[tuple[str, str, float], tuple[float, float]] = {}

    def rate(name: str, body: str, duration: float) -> tuple[float, float]:
        key = (name, body, duration)
        if key not in rates:
            rates[key] = fit_rate(
                series[(name, body)], duration, args.trim_fraction, args.fit_blocks
            )
        return rates[key]

    def difference(
        force_case: str,
        control_case: str,
        body: str,
        duration: float,
    ) -> tuple[float, float]:
        force_rate, force_unc = rate(force_case, body, duration)
        control_rate, control_unc = rate(control_case, body, duration)
        return force_rate - control_rate, math.hypot(force_unc, control_unc)

    disk, giant = analytical_setup(args)
    rocky_a = np.array([args.rocky_a])
    giant_a = np.array([args.giant_a])
    empty_disk = analytical.DiskParams(**{**asdict(disk), "k_gap_dep": 0.0})

    gij = float(analytical.planet_precession_from_giant(rocky_a, giant)[0]) * ANALYTICAL_TO_RAD_PER_YR
    gi_no_gap = float(analytical.disk_precession_no_gap(rocky_a, args.mstar, disk)[0]) * ANALYTICAL_TO_RAD_PER_YR
    gi_empty = float(analytical.disk_precession_dep_gap(rocky_a, giant, empty_disk, 0.0)[0]) * ANALYTICAL_TO_RAD_PER_YR
    gj_empty = float(analytical.disk_precession_dep_gap(giant_a, giant, empty_disk, 0.0)[0]) * ANALYTICAL_TO_RAD_PER_YR
    gi_depleted = float(analytical.disk_precession_dep_gap(rocky_a, giant, disk, args.k_gap_dep)[0]) * ANALYTICAL_TO_RAD_PER_YR
    gj_depleted = float(analytical.disk_precession_dep_gap(giant_a, giant, disk, args.k_gap_dep)[0]) * ANALYTICAL_TO_RAD_PER_YR
    gi_gr = float(analytical.gr_precession(rocky_a, args.mstar)[0]) * ANALYTICAL_TO_RAD_PER_YR
    gj_gr = float(analytical.gr_precession(giant_a, args.mstar)[0]) * ANALYTICAL_TO_RAD_PER_YR

    gravity_duration = min(args.gravity_duration, args.gr_duration)
    disk_duration = args.disk_duration
    gr_duration = args.gr_duration

    rows: list[dict[str, object]] = []

    def add_row(name: str, measured: tuple[float, float], predicted: float, note: str) -> None:
        value, uncertainty = measured
        residual = value - predicted
        relative = residual / predicted if predicted != 0.0 else math.nan
        rows.append(
            {
                "term": name,
                "measured_rad_per_yr": value,
                "block_scatter_rad_per_yr": uncertainty,
                "analytical_rad_per_yr": predicted,
                "residual_rad_per_yr": residual,
                "relative_error": relative,
                "note": note,
            }
        )

    add_row(
        "g_i,J",
        difference("rocky_giant", "rocky_control", "rocky", gravity_duration),
        gij,
        "circular giant; control-subtracted",
    )
    add_row(
        "g_i,disk no gap",
        difference("rocky_disk_no_gap", "rocky_control", "rocky", disk_duration),
        gi_no_gap,
        "static smooth disk; control-subtracted",
    )
    add_row(
        "g_J,disk empty gap",
        difference("giant_disk_empty", "giant_control", "giant", disk_duration),
        gj_empty,
        "giant-only paired test",
    )
    add_row(
        "g_J,disk depleted gap",
        difference("giant_disk_depleted", "giant_control", "giant", disk_duration),
        gj_depleted,
        "giant-only paired test",
    )
    add_row(
        "g_i,disk empty gap",
        difference("pair_disk_empty", "pair_control", "rocky", disk_duration),
        gi_empty,
        "same rocky+giant system, disk off/on",
    )
    add_row(
        "g_J,disk empty gap (pair)",
        difference("pair_disk_empty", "pair_control", "giant", disk_duration),
        gj_empty,
        "callback wiring cross-check",
    )
    add_row(
        "g_i,disk depleted gap",
        difference("pair_disk_depleted", "pair_control", "rocky", disk_duration),
        gi_depleted,
        "same rocky+giant system, disk off/on",
    )
    add_row(
        "g_J,disk depleted gap (pair)",
        difference("pair_disk_depleted", "pair_control", "giant", disk_duration),
        gj_depleted,
        "callback wiring cross-check",
    )
    add_row(
        "g_i,GR",
        difference("rocky_gr", "rocky_control", "rocky", gr_duration),
        gi_gr,
        "1PN force; current analytical code uses C_CODE=1e4 and omits 1/(1-e^2)",
    )
    add_row(
        "full additional force on rocky",
        difference("pair_full_depleted_gr", "pair_control", "rocky", disk_duration),
        gi_depleted + gi_gr,
        "depleted-gap disk + GR",
    )
    add_row(
        "full additional force on giant",
        difference("pair_full_depleted_gr", "pair_control", "giant", disk_duration),
        gj_depleted + gj_gr,
        "depleted-gap disk + GR",
    )

    summary_path = output_dir / "precession_rate_summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(3, 4, figsize=(15, 10), constrained_layout=True)
    for axis, experiment in zip(axes.flat, EXPERIMENTS):
        plotted = False
        for body, color in (("rocky", "tab:blue"), ("giant", "tab:orange")):
            key = (experiment.name, body)
            if key not in series:
                continue
            body_series = series[key]
            axis.plot(
                body_series.time / 1.0e3,
                body_series.varpi - body_series.varpi[0],
                color=color,
                lw=1.0,
                label=body,
            )
            plotted = True
        axis.set_title(experiment.name.replace("_", " "), fontsize=10)
        axis.set_xlabel("Time [kyr]")
        axis.set_ylabel(r"$\Delta\varpi$ [rad]")
        axis.grid(alpha=0.25)
        if plotted:
            axis.legend(fontsize=8)
    for axis in axes.flat[len(EXPERIMENTS):]:
        axis.set_visible(False)
    phase_path = figure_dir / "precession_phase_timeseries.png"
    fig.savefig(phase_path, dpi=200)
    plt.close(fig)

    labels = [str(row["term"]) for row in rows]
    measured_values = np.array([float(row["measured_rad_per_yr"]) for row in rows])
    predicted_values = np.array([float(row["analytical_rad_per_yr"]) for row in rows])
    uncertainties = np.array([float(row["block_scatter_rad_per_yr"]) for row in rows])
    positions = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(11, 7), constrained_layout=True)
    axis.errorbar(
        measured_values,
        positions,
        xerr=uncertainties,
        fmt="o",
        color="tab:blue",
        capsize=3,
        label="REBOUND, control-subtracted",
    )
    axis.plot(predicted_values, positions, "x", color="black", ms=8, label="Analytical")
    axis.axvline(0.0, color="gray", lw=0.8)
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlabel(r"Apsidal precession rate [rad yr$^{-1}$]")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    comparison_path = figure_dir / "precession_rate_comparison.png"
    fig.savefig(comparison_path, dpi=220)
    plt.close(fig)

    for row in rows:
        print(
            f"{row['term']}: measured={row['measured_rad_per_yr']:+.6e}, "
            f"analytical={row['analytical_rad_per_yr']:+.6e}, "
            f"relative_error={row['relative_error']:+.3%}"
        )
    print(f"summary={summary_path}")
    print(f"phase_figure={phase_path}")
    print(f"comparison_figure={comparison_path}")


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.figure_dir = args.figure_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in {"all", "run"}:
        experiments = selected_experiments(args)
        ncpus = args.ncpus
        if ncpus is None:
            ncpus = parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
        ncpus = max(1, min(ncpus, len(experiments)))
        options = vars(args).copy()
        options["output_dir"] = str(args.output_dir)
        options["figure_dir"] = str(args.figure_dir)
        print(f"running {len(experiments)} experiments with {ncpus} workers")
        print(f"output_dir={args.output_dir}")
        if ncpus == 1:
            for experiment in experiments:
                archive = run_experiment(experiment, options)
                print(f"finished {experiment.name}: {archive}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=ncpus) as executor:
                futures = {
                    executor.submit(run_experiment, experiment, options): experiment.name
                    for experiment in experiments
                }
                for future in as_completed(futures):
                    print(f"finished {futures[future]}: {future.result()}", flush=True)

    if args.mode in {"all", "analyze"}:
        if args.case:
            raise ValueError("--case can only be used with --mode run")
        analyze_suite(args)


if __name__ == "__main__":
    main()
