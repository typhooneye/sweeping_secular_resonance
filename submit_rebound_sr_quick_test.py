#!/usr/bin/env python

#SBATCH --job-name=ssr-quick-test
#SBATCH --output=logs/ssr-quick-test_%j.out
#SBATCH --error=logs/ssr-quick-test_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1

"""Submit a short, high-signal SSR test grid in parallel.

The analytical crossing for an embryo at 1 AU occurs at a few kyr for this
setup, so the 20 kyr integration should contain the resonance passage while
remaining suitable for a quick timestep/force sanity check.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import argparse
from argparse import Namespace
from pathlib import Path

REPO_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent)).resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from submit_rebound_aj_ej_tdep import BASELINE, build_runs, parse_slurm_cpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-giants", default="5", help="Comma-separated giant semimajor axes [AU].")
    parser.add_argument("--e-giants", default="0.05", help="Comma-separated giant eccentricities.")
    parser.add_argument("--t-dep", type=float, default=1.0e3, help="Disk depletion time [yr].")
    parser.add_argument("--f-e", type=float, default=1.0, help="Eccentricity damping multiplier.")
    parser.add_argument("--t-int", type=float, default=2.0e4, help="Integration time [yr].")
    parser.add_argument("--dt-int", type=float, default=10.0, help="Archive output interval [yr].")
    parser.add_argument("--dt-fraction", type=float, default=0.01, help="Timestep as a fraction of embryo period.")
    parser.add_argument("--k-gap-dep", type=float, default=0.0)
    parser.add_argument("--output-dir", default="output/rebound_ssr_quick_test")
    parser.add_argument("--include-gr", type=int, choices=[0, 1], default=1)
    parser.add_argument("--ncpus", type=int, default=None)
    return parser.parse_args()


def build_args(cli_args: argparse.Namespace) -> Namespace:
    return Namespace(
        a_giants=cli_args.a_giants,
        e_giants=cli_args.e_giants,
        tdeps=str(cli_args.t_dep),
        output_dir=cli_args.output_dir,
        ncpus=cli_args.ncpus,
        n_embryos=1,
        spacing="single",
        embryo_a0=1.0,
        embryo_mass_earth=1.0,
        embryo_radius_earth=1.0,
        embryo_e=0.0,
        mstar=1.0,
        rstar=1.0,
        m_giant_mjup=1.0,
        r_giant_rjup=1.0,
        open_disk=1,
        open_typeI=1,
        k_hill=8.0,
        k_gap_dep=cli_args.k_gap_dep,
        f_e=cli_args.f_e,
        t_int=cli_args.t_int,
        dt_int=cli_args.dt_int,
        dt_fraction=cli_args.dt_fraction,
        include_gr=cli_args.include_gr,
    )


def main() -> None:
    args = build_args(parse_args())
    runs = build_runs(args)
    ncpus = args.ncpus
    if ncpus is None:
        ncpus = parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
    ncpus = min(ncpus, len(runs))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"running {len(runs)} quick SSR tests with {ncpus} workers")
    print(f"output_dir={Path(args.output_dir).resolve()}")

    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(BASELINE.run_simulation, runs):
            print(result, flush=True)


if __name__ == "__main__":
    main()
