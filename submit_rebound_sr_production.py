#!/usr/bin/env python

#SBATCH --job-name=ssr-production
#SBATCH --output=logs/ssr-production_%j.out
#SBATCH --error=logs/ssr-production_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=35:00:00

"""Official SR production grid (revised 2026-08-21).

Grid: a_J = 5, 10, 15, 20 AU x e_J = 0.05, 0.2, 0.5 x tdep = 1e5, 1e6 yr
      -> 24 runs.
Fixed physics: m_J=1 Mjup, embryo 1 Mearth at 1 AU, sigma0=1700 g/cm^2,
k=1.5, k_gap_dep=0.0, gap + GR on, f_e=F_E (below), dt = P_emb/100.
t_int = 12*tdep: the latest empty-gap crossing in the grid is
t_R = 8.93*tdep (a_J=20), leaving >= 3*tdep afterward everywhere.

Empty-gap analytical crossing ranges at 1 AU, from ssr_single_giant.py:
a_J = 5, 10, 15, 20 AU cross at 4.81-5.58, 6.81-7.32, 7.93-8.30,
and 8.63-8.93 tdep, respectively.

Cost: tdep=1e5 runs are ~1.2e8 steps (~17 h); tdep=1e6 runs are ~1.2e9
steps (~7 days, i.e. ~5 wall-time windows).

RESUBMIT TO CONTINUE: every run sets "resume": 1, so when the 35 h wall
limit kills the job, run `sbatch submit_rebound_sr_production.py` again --
each run reloads the last snapshot of its archive and continues; finished
runs return immediately.  Repeat until the log shows all 24 finished.
(Requires the resume branch added to run_simulation in
rebound+SSR+type1_baseline.py on 2026-08-20.  Do NOT change grid
parameters between resubmissions without deleting the affected archives.)
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

BASE_OUTPUT = REPO_DIR / "output" / "rebound_sr_production_empty_gap"

A_GIANTS = "5,10,15,20"
E_GIANTS = "0.05,0.2,0.5"
TDEPS = (1.0e5, 1.0e6)
T_INT_OVER_TDEP = 12.0
# Eccentricity damping: f_e=0 so the sweep kick is undamped and visible.
F_E = 0.0


def build_production_runs() -> list[tuple]:
    runs = []
    for tdep in TDEPS:
        # dt_int scaled so every run archives ~6000 snapshots.
        t_int = T_INT_OVER_TDEP * tdep
        dt_int = t_int / 6000.0
        args = Namespace(
            a_giants=A_GIANTS,
            e_giants=E_GIANTS,
            tdeps=str(tdep),
            output_dir=str(BASE_OUTPUT / f"tdep{tdep:.0e}"),
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
            f_e=F_E,
            t_int=t_int,
            dt_int=dt_int,
            dt_fraction=0.1,
            include_gr=1,
        )
        case_runs = build_runs(args)
        for run in case_runs:
            run[17]["resume"] = 1
        runs.extend(case_runs)
    return runs


def main() -> None:
    runs = build_production_runs()
    ncpus = min(parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE")), len(runs))
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"running {len(runs)} production SR runs with {ncpus} workers (f_e={F_E}, resume enabled)")
    for run in runs:
        fp = run[17]
        print(f"  {os.path.basename(fp['output_dir'])}/{run[0]}  t_int={fp['t_int']:.1e}")

    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(BASELINE.run_simulation, runs):
            print(f"finished: {result}", flush=True)


if __name__ == "__main__":
    main()
