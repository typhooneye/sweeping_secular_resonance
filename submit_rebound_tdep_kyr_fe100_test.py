#!/usr/bin/env python

#SBATCH --job-name=ssr-kyr-fe100
#SBATCH --output=logs/ssr-kyr-fe100_%j.out
#SBATCH --error=logs/ssr-kyr-fe100_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1

"""Submit kyr-scale REBOUND tests with 100x stronger eccentricity damping."""

from __future__ import annotations

import multiprocessing
import os
import sys
from argparse import Namespace
from pathlib import Path

REPO_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent)).resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from submit_rebound_aj_ej_tdep import BASELINE, build_runs, parse_slurm_cpus


def build_args() -> Namespace:
    return Namespace(
        a_giants="5,10,20,30",
        e_giants="0.05,0.5,0.8",
        tdeps="1e3,1e4",
        output_dir="output/rebound_aj_ej_tdep_kyr_fe100",
        ncpus=None,
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
        k_gap_dep=0.0,
        f_e=100.0,
        t_int=1.0e5,
        dt_int=10.0,
        include_gr=1,
    )


def main() -> None:
    args = build_args()
    runs = build_runs(args)
    ncpus = min(parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE")), len(runs))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"running {len(runs)} kyr-scale f_e=100 simulations with {ncpus} workers")
    print(f"output_dir={Path(args.output_dir).resolve()}")

    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(BASELINE.run_simulation, runs):
            print(result, flush=True)


if __name__ == "__main__":
    main()
