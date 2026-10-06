#!/usr/bin/env python

#SBATCH --job-name=diagnose-sr-stepwise
#SBATCH --output=logs/diagnose-sr-stepwise_%j.out
#SBATCH --error=logs/diagnose-sr-stepwise_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1

"""Isolate each force in the SSR model and check it against its analytical rate.

Four independent REBOUND runs, each built from the *actual* production force
code in rebound+SSR+type1_baseline.py (not a reimplementation), so any bug in
that code shows up here too:

  A. rocky_giant  -- rocky embryo + giant planet, gravity only (no disk, no GR)
                      vs. Laplace-Lagrange planet_precession_from_giant
  B. rocky_disk   -- rocky embryo + static disk potential, no giant, no GR
                      vs. disk_precession_no_gap
  C. giant_disk   -- giant planet + its own static gapped disk potential, no
                      rocky embryo, no GR (a close-in, negligible-mass decoy
                      embryo is added only because run_simulation's dt-setting
                      code indexes particles[n_j+1] and needs one to exist --
                      see the --decoy-a note below)
                      vs. disk_precession_dep_gap
  D. rocky_gr     -- rocky embryo + GR precession only, no giant, no disk
                      vs. gr_precession

Each run saves its own REBOUND SimulationArchive .bin via the same
sim.save_to_file(...) call run_simulation() always makes, so the raw
trajectories are on disk for independent inspection.  The four runs are
launched in parallel with multiprocessing.Pool, then each archive is read
back to fit a measured apsidal precession rate d(pomega)/dt and compare it to
the analytical prediction.

Do not run this locally against the full-duration defaults -- the GR test in
particular integrates ~1e8 steps.  Submit it to the cluster:
    sbatch diagnose_sr_stepwise.py
or run interactively with a shorter --gr-duration for a quick smoke test.

Note on softening: run_simulation() hardcodes sim.softening=1e-3 AU.  Do not
lower --rocky-a much below 1 AU for the GR test without also shrinking that --
a softened potential itself induces spurious apsidal precession of order
(softening/a)^2 per orbit, which at a few tenths of an AU can rival or exceed
the genuine (tiny) GR signal and would corrupt this specific test.

Known analytical-side caveat (not fixed here): ssr_single_giant.py's
gr_precession() uses C_CODE = 1.0e4, but the correct speed of light in these
G=1 analytical units is c_true_AU_per_yr / (2*pi) = 63241.077/(2*pi) =
10065.13 (i.e. reboundx.constants.C). That is a ~0.65% error in the GR rate
prediction alone -- small, but it will show up as a nonzero relative_error on
the "rocky_gr" row even if the simulation is perfect. gr_precession() also
omits the 1/(1-e^2) eccentricity factor.
"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import os
import sys
from dataclasses import dataclass, field
from importlib.machinery import SourceFileLoader
from pathlib import Path

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

BASELINE = SourceFileLoader(
    "rebound_ssr_stepwise_baseline",
    str(REPO_DIR / "rebound+SSR+type1_baseline.py"),
).load_module()

# ssr_single_giant.py works in G=1 units (time unit = yr/2*pi for AU, Msun);
# REBOUND's ('yr','AU','Msun') units have G=4*pi^2, so true rad/yr = 2*pi * (G=1 rate).
ANALYTICAL_TO_RAD_PER_YR = 2.0 * math.pi


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Fold an angle (or array of angles) into [0, pi)."""
    return np.mod(angle, math.pi)


@dataclass
class StepConfig:
    name: str
    fname: str
    run_args: tuple
    track_index: int
    duration: float
    sample_interval: float
    analytical_rate_rad_per_yr: float = field(default=0.0)


@dataclass
class StepResult:
    name: str
    time: np.ndarray
    varpi_raw: np.ndarray
    varpi_unwrapped: np.ndarray
    eccentricity: np.ndarray
    measured_rate: float
    analytical_rate: float
    bin_path: Path

    @property
    def relative_error(self) -> float:
        if self.analytical_rate == 0.0:
            return math.nan
        return (self.measured_rate - self.analytical_rate) / self.analytical_rate


def make_disk(args: argparse.Namespace) -> analytical.DiskParams:
    return analytical.DiskParams(
        sigma0_cgs=args.sigma0_cgs,
        k=args.disk_k,
        z_k=args.z_k,
        t_dep=math.inf,
        disk_inner_cutoff=args.disk_inner_cutoff,
        n_terms=args.n_terms,
        k_gap_dep=args.k_gap_dep,
        include_gr=False,
        include_gap_effect=True,
    )


def base_force_params(args: argparse.Namespace, output_dir: str, **overrides) -> dict:
    params = {
        "dens0": args.sigma0_cgs * 1.125e-7,
        "di": args.disk_k,
        "zk": args.z_k,
        "tdep": math.inf,  # static disk: isolate the precession rate, not depletion
        "disk_inner_cutoff": args.disk_inner_cutoff,
        "k_gap_dep": args.k_gap_dep,
        "n_terms": args.n_terms,
        "include_gr": 0,
        "use_gap": 0,
        "output_dir": output_dir,
        "dt_fraction": args.dt_fraction,
    }
    params.update(overrides)
    return params


def build_configs(args: argparse.Namespace) -> list[StepConfig]:
    output_dir = str(Path(args.output_dir).resolve())
    configs: list[StepConfig] = []

    # --- A: rocky + giant, gravity only -----------------------------------
    giant_a = args.giant_a
    force_params_a = base_force_params(
        args,
        output_dir,
        t_int=args.giant_duration,
        dt_int=args.giant_sample_interval,
    )
    configs.append(
        StepConfig(
            name="rocky_giant",
            fname="A_rocky_giant.bin",
            run_args=(
                "A_rocky_giant.bin", 8.0, 1, 1, args.mstar, args.rstar,
                np.array([args.giant_mass_mjup]), np.array([1.0]), np.array([giant_a]), np.array([0.0]),
                args.rocky_mass_earth, 1.0, args.rocky_a, args.rocky_e,
                1, 0, 0, force_params_a,
            ),
            track_index=2,
            duration=args.giant_duration,
            sample_interval=args.giant_sample_interval,
            analytical_rate_rad_per_yr=float(
                analytical.planet_precession_from_giant(
                    np.array([args.rocky_a]),
                    analytical.GiantParams(a=giant_a, e=0.0, mass_mjup=args.giant_mass_mjup, mstar=args.mstar),
                )[0] * ANALYTICAL_TO_RAD_PER_YR
            ),
        )
    )

    # --- B: rocky + static disk, no giant -----------------------------------
    disk_b = make_disk(args)
    force_params_b = base_force_params(
        args,
        output_dir,
        t_int=args.disk_duration,
        dt_int=args.disk_sample_interval,
        use_gap=0,
    )
    configs.append(
        StepConfig(
            name="rocky_disk",
            fname="B_rocky_disk.bin",
            run_args=(
                "B_rocky_disk.bin", 8.0, 0, 1, args.mstar, args.rstar,
                np.array([]), np.array([]), np.array([]), np.array([]),
                args.rocky_mass_earth, 1.0, args.rocky_a, args.rocky_e,
                1, 1, 0, force_params_b,
            ),
            track_index=1,
            duration=args.disk_duration,
            sample_interval=args.disk_sample_interval,
            analytical_rate_rad_per_yr=float(
                analytical.disk_precession_no_gap(np.array([args.rocky_a]), args.mstar, disk_b)[0]
                * ANALYTICAL_TO_RAD_PER_YR
            ),
        )
    )

    # --- C: giant + its own gapped disk, no rocky ---------------------------
    # run_simulation() sets sim.dt from particles[n_j+1].P, i.e. from whatever
    # p2-type body exists -- so the decoy must be close-in (small period) or
    # it would size the (shared, global) timestep off its own long period and
    # badly under-resolve the giant's orbit. Placed close to the star at
    # --decoy-a (default 0.1 AU, ~4.9 AU inside the giant's orbit at a=5 AU --
    # many times the giant's ~0.3 AU Hill radius) it is both dynamically
    # inert on the giant and gives a short period, so it safely over-resolves
    # rather than under-resolves the timestep. Only the giant (index 1) is
    # tracked/analyzed.
    disk_c = make_disk(args)
    giant_c = analytical.GiantParams(a=args.giant_a, e=args.giant_e_gap, mass_mjup=args.giant_mass_mjup, mstar=args.mstar)
    force_params_c = base_force_params(
        args,
        output_dir,
        t_int=args.giant_disk_duration,
        dt_int=args.giant_disk_sample_interval,
        use_gap=1,
    )
    configs.append(
        StepConfig(
            name="giant_disk",
            fname="C_giant_disk.bin",
            run_args=(
                "C_giant_disk.bin", 8.0, 1, 1, args.mstar, args.rstar,
                np.array([args.giant_mass_mjup]), np.array([1.0]), np.array([args.giant_a]), np.array([args.giant_e_gap]),
                1.0, 1.0, args.decoy_a, 0.0,
                1, 1, 0, force_params_c,
            ),
            track_index=1,
            duration=args.giant_disk_duration,
            sample_interval=args.giant_disk_sample_interval,
            analytical_rate_rad_per_yr=float(
                analytical.disk_precession_dep_gap(
                    np.array([args.giant_a]), giant_c, disk_c, args.k_gap_dep
                )[0] * ANALYTICAL_TO_RAD_PER_YR
            ),
        )
    )

    # --- D: rocky + GR only, no giant, no disk ------------------------------
    force_params_d = base_force_params(
        args,
        output_dir,
        t_int=args.gr_duration,
        dt_int=args.gr_sample_interval,
        include_gr=1,
    )
    configs.append(
        StepConfig(
            name="rocky_gr",
            fname="D_rocky_gr.bin",
            run_args=(
                "D_rocky_gr.bin", 8.0, 0, 1, args.mstar, args.rstar,
                np.array([]), np.array([]), np.array([]), np.array([]),
                args.rocky_mass_earth, 1.0, args.rocky_a, args.rocky_e,
                1, 0, 0, force_params_d,
            ),
            track_index=1,
            duration=args.gr_duration,
            sample_interval=args.gr_sample_interval,
            analytical_rate_rad_per_yr=float(
                analytical.gr_precession(np.array([args.rocky_a]), args.mstar)[0]
                * ANALYTICAL_TO_RAD_PER_YR
            ),
        )
    )

    return configs


def read_archive(bin_path: Path, track_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sa = rebound.Simulationarchive(str(bin_path))
    times, varpi, ecc = [], [], []
    for sim_snap in sa:
        orbit = sim_snap.particles[track_index].orbit(primary=sim_snap.particles[0])
        times.append(sim_snap.t)
        varpi.append(orbit.pomega)
        ecc.append(orbit.e)
    return np.array(times), np.array(varpi), np.array(ecc)


def fit_rate(time: np.ndarray, varpi_unwrapped: np.ndarray, trim_fraction: float) -> float:
    first = int(trim_fraction * len(time))
    if len(time) - first < 2:
        first = 0
    return float(np.polyfit(time[first:], varpi_unwrapped[first:], 1)[0])


def analyze(config: StepConfig, output_dir: Path, trim_fraction: float) -> StepResult:
    bin_path = output_dir / config.fname
    time, varpi_raw, ecc = read_archive(bin_path, config.track_index)
    varpi_unwrapped = np.unwrap(varpi_raw)
    measured = fit_rate(time, varpi_unwrapped, trim_fraction)
    return StepResult(
        name=config.name,
        time=time,
        varpi_raw=varpi_raw,
        varpi_unwrapped=varpi_unwrapped,
        eccentricity=ecc,
        measured_rate=measured,
        analytical_rate=config.analytical_rate_rad_per_yr,
        bin_path=bin_path,
    )


def write_summary(results: list[StepResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["test", "measured_rad_per_yr", "analytical_rad_per_yr", "relative_error", "bin_path"]
        )
        for result in results:
            writer.writerow(
                [result.name, result.measured_rate, result.analytical_rate,
                 result.relative_error, result.bin_path]
            )
            np.savetxt(
                path.parent / f"{result.name}_omega.csv",
                np.column_stack((result.time, result.varpi_raw, result.varpi_unwrapped, result.eccentricity)),
                delimiter=",",
                header="time_yr,pomega_raw_rad,pomega_unwrapped_rad,eccentricity",
                comments="",
            )


def plot_results(results: list[StepResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(results), figsize=(4.5 * len(results), 8.0), constrained_layout=True)

    for col, result in enumerate(results):
        top = axes[0, col]
        delta = result.varpi_unwrapped - result.varpi_unwrapped[0]
        analytical_line = result.analytical_rate * result.time
        top.plot(result.time, delta, color="tab:blue", lw=1.5, label="REBOUND")
        top.plot(result.time, analytical_line, color="black", ls="--", lw=1.4, label="Analytical")
        ratio = result.measured_rate / result.analytical_rate if result.analytical_rate != 0.0 else math.nan
        top.set_title(f"{result.name}\nmeasured/analytical = {ratio:.4f}")
        top.set_xlabel("Time [yr]")
        top.set_ylabel(r"$\varpi(t)-\varpi(0)$ [rad]")
        top.grid(alpha=0.25)
        top.legend(fontsize=8)

        bottom = axes[1, col]
        analytical_mod = wrap_to_pi(result.varpi_unwrapped[0] + analytical_line)
        bottom.plot(result.time, wrap_to_pi(result.varpi_raw), ".", ms=2, color="tab:blue", label="REBOUND")
        bottom.plot(result.time, analytical_mod, ".", ms=2, color="black", label="Analytical")
        bottom.set_xlabel("Time [yr]")
        bottom.set_ylabel(r"$\varpi$ mod $\pi$ [rad]")
        bottom.set_ylim(0.0, math.pi)
        bottom.grid(alpha=0.25)
        bottom.legend(fontsize=8)

    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_slurm_cpus(value: str | None) -> int:
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
    return total if total > 0 else multiprocessing.cpu_count()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mstar", type=float, default=1.0)
    parser.add_argument("--rstar", type=float, default=1.0)
    parser.add_argument("--rocky-a", type=float, default=1.0)
    parser.add_argument("--rocky-e", type=float, default=0.01)
    parser.add_argument("--rocky-mass-earth", type=float, default=1.0)

    parser.add_argument("--giant-a", type=float, default=5.0)
    parser.add_argument("--giant-mass-mjup", type=float, default=1.0)
    parser.add_argument("--giant-e-gap", type=float, default=0.05, help="Giant eccentricity used only in test C (gap shape).")

    parser.add_argument("--sigma0-cgs", type=float, default=1700.0)
    parser.add_argument("--disk-k", type=float, default=1.5)
    parser.add_argument("--z-k", type=float, default=1.094)
    parser.add_argument("--disk-inner-cutoff", type=float, default=0.01)
    parser.add_argument("--n-terms", type=int, default=30)
    parser.add_argument("--k-gap-dep", type=float, default=0.1)
    parser.add_argument(
        "--decoy-a", type=float, default=0.1,
        help="Placement of the inert embryo in test C. Must stay close-in (short period) -- "
             "see the comment in build_configs() for why a far-out decoy would break the timestep.",
    )

    parser.add_argument("--dt-fraction", type=float, default=0.01, help="REBOUND timestep as a fraction of the embryo/giant period.")

    parser.add_argument("--giant-duration", type=float, default=1.0e4)
    parser.add_argument("--giant-sample-interval", type=float, default=20.37)
    parser.add_argument("--disk-duration", type=float, default=1.0e4)
    parser.add_argument("--disk-sample-interval", type=float, default=1.73)
    parser.add_argument("--giant-disk-duration", type=float, default=1.0e4)
    parser.add_argument("--giant-disk-sample-interval", type=float, default=20.37)
    parser.add_argument("--gr-duration", type=float, default=1.0e4,
                         help="GR precession at 1 AU is ~1e-7 rad/yr; needs a long baseline to fit cleanly.")
    parser.add_argument("--gr-sample-interval", type=float, default=20.37)

    parser.add_argument("--trim-fraction", type=float, default=0.05, help="Fraction of each series dropped before fitting the rate.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/diagnose_sr_stepwise"))
    parser.add_argument("--figure", type=Path, default=Path("figure/diagnose_sr_stepwise_omega.png"))
    parser.add_argument("--ncpus", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args)

    ncpus = args.ncpus
    if ncpus is None:
        ncpus = parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
    ncpus = min(ncpus, len(configs))

    print(f"running {len(configs)} SSR component tests with {ncpus} workers")
    print(f"output_dir={output_dir}")
    with multiprocessing.Pool(ncpus) as pool:
        for fname in pool.starmap(BASELINE.run_simulation, [c.run_args for c in configs]):
            print(f"finished: {fname}", flush=True)

    results = [analyze(config, output_dir, args.trim_fraction) for config in configs]

    write_summary(results, output_dir / "summary.csv")
    plot_results(results, args.figure)

    for result in results:
        print(
            f"{result.name}: measured={result.measured_rate:.6e} rad/yr, "
            f"analytical={result.analytical_rate:.6e} rad/yr, "
            f"relative_error={result.relative_error:+.3%}"
        )
    print(f"summary={output_dir / 'summary.csv'}")
    print(f"figure={args.figure.resolve()}")


if __name__ == "__main__":
    main()
