#!/usr/bin/env python

#SBATCH --job-name=ssr-fast
#SBATCH --output=logs/ssr-fast_%j.out
#SBATCH --error=logs/ssr-fast_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00

"""Submit fast, controlled SR runs with normal giant-planet parameters.

Both runs use a_J=5 AU, e_J=0.2, T_dep=1 kyr, and an Earth-mass embryo at
1 AU.  The analytical SR crossing is near 4.8 kyr.  The two runs differ only
in eccentricity damping: f_e=1 is the analytical default and f_e=0 is the
undamped control.
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


BASE_OUTPUT = REPO_DIR / "output" / "rebound_sr_fast"


CASES = (
    ("fe1", 1.0),
    ("fe0", 0.0),
)


def build_fast_runs() -> list[tuple]:
    runs = []
    for label, f_e in CASES:
        args = Namespace(
            a_giants="5",
            e_giants="0.2",
            tdeps="1e3",
            output_dir=str(BASE_OUTPUT / label),
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
            f_e=f_e,
            t_int=1.5e4,
            dt_int=10.0,
            dt_fraction=0.1,
            include_gr=1,
        )
        runs.extend(build_runs(args))
    return runs


def main() -> None:
    runs = build_fast_runs()
    ncpus = min(parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE")), len(runs))
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"running {len(runs)} fast SR runs with {ncpus} workers")
    print("parameters: M_J=1, a_J=5 AU, e_J=0.2, T_dep=1 kyr, a_emb=1 AU")
    print("predicted SR crossing: approximately 4.8 kyr")
    for run in runs:
        print(f"  {run[17]['output_dir']}/{run[0]}")

    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(BASELINE.run_simulation, runs):
            print(f"finished: {result}", flush=True)


if __name__ == "__main__":
    main()
