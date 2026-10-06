#!/usr/bin/env python

#SBATCH --job-name=ssr-tdep1k-test
#SBATCH --output=logs/ssr-tdep1k-test_%j.out
#SBATCH --error=logs/ssr-tdep1k-test_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1

"""Submit the 12-run Tdep=1 kyr REBOUND test grid.

Grid:
    a_J = 5, 10, 20, 30 AU
    e_J = 0.05, 0.5, 0.8
    T_dep = 1 kyr

Everything else follows submit_rebound_aj_ej_tdep.py defaults:
one Earth-mass embryo starts at 1 AU, gap+GR+Type-I damping enabled,
and the integration/output cadence defaults are unchanged.
"""

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
        tdeps="1e3",
        output_dir="output/rebound_aj_ej_tdep_1kyr",
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
        f_e=1000.0,
        t_int=1.0e5,
        dt_int=10.0,
        include_gr=1,
    )


def main() -> None:
    args = build_args()
    runs = build_runs(args)
    ncpus = parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
    ncpus = min(ncpus, len(runs))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"running {len(runs)} Tdep=1 kyr simulations with {ncpus} workers")
    print(f"output_dir={Path(args.output_dir).resolve()}")

    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(BASELINE.run_simulation, runs):
            print(result, flush=True)


if __name__ == "__main__":
    main()
