#!/usr/bin/env python

#SBATCH --job-name=ssr-demo
#SBATCH --output=logs/ssr-demo_%j.out
#SBATCH --error=logs/ssr-demo_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=18
#SBATCH --cpus-per-task=1
#SBATCH --time=35:00:00

"""SR demo grid with varied type-I damping strength (revised 2026-08-21).

Grid: a_J = 5, 10 AU x e_J = 0.1, 0.3, 0.5 x f_e = 1, 10, 100
      at tdep = 1e5 yr -> 18 runs.
Fixed physics: m_J=1 Mjup, embryo 1 Mearth at 1 AU, sigma0=1700 g/cm^2,
k=1.5, k_gap_dep=0.0, gap + GR on, dt = P_emb/100.
t_int = 5e6 yr = 50*tdep: empty-gap crossings occur at 4.81-5.09*tdep
(a_J=5) and 6.81-7.00*tdep (a_J=10).

f_e multiplies the type-I eccentricity-damping rate (t_e = t_damp/f_e), so
LARGER f_e means STRONGER damping.

Cost: each run is ~1e8 steps (~17 h at ~2e3 steps/s), all 18 in parallel.

RESUBMIT TO CONTINUE: every run sets "resume": 1, so if the wall limit
kills the job, `sbatch submit_rebound_sr_demo.py` again -- each run
reloads the last snapshot of its archive and continues; finished runs
return immediately.  Do NOT change grid parameters between resubmissions
without deleting the affected archives.
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

BASE_OUTPUT = REPO_DIR / "output" / "rebound_sr_demo_empty_gap"

A_GIANTS = (5.0, 10.0)
E_GIANTS = (0.1, 0.3, 0.5)
F_ES = (1.0, 10.0, 100.0)
TDEP = 1.0e5
T_INT = 5.0e6
DT_INT = 1000.0


def build_demo_runs() -> list[tuple]:
    runs = []
    for a_j in A_GIANTS:
        for e_j in E_GIANTS:
            for f_e in F_ES:
                # The archive filename does not encode f_e, so each case
                # writes to its own subdirectory.
                label = f"aj{a_j:g}_ej{e_j:g}_fe{f_e:g}"
                args = Namespace(
                    a_giants=str(a_j),
                    e_giants=str(e_j),
                    tdeps=str(TDEP),
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
                    t_int=T_INT,
                    dt_int=DT_INT,
                    dt_fraction=0.1,
                    include_gr=1,
                )
                case_runs = build_runs(args)
                for run in case_runs:
                    run[17]["resume"] = 1
                runs.extend(case_runs)
    return runs


def main() -> None:
    runs = build_demo_runs()
    ncpus = min(parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE")), len(runs))
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"running {len(runs)} SR demo runs with {ncpus} workers (resume enabled)")
    for run in runs:
        print(f"  {os.path.basename(run[17]['output_dir'])}/{run[0]}")

    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(BASELINE.run_simulation, runs):
            print(f"finished: {result}", flush=True)


if __name__ == "__main__":
    main()
