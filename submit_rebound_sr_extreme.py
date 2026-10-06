#!/usr/bin/env python

#SBATCH --job-name=ssr-extreme
#SBATCH --output=logs/ssr-extreme_%j.out
#SBATCH --error=logs/ssr-extreme_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

"""Two extreme short-timescale SR demonstration runs (see 2026-08-20 diagnosis).

The trick: push u_R (disk depletion fraction at crossing) close to 1 so the
resonance sweeps past early, while keeping u_R < 1 so a sweep exists at all
(lower sigma0 / higher m_J eventually give u_R > 1 -> no crossing).

  1. "robust":  m_J=10 Mjup, a_J=5, e_J=0.5, embryo 1 AU, sigma0=1700,
     tdep=1e4, f_e=0, t_int=5e4 yr.
     Predicted: u_R=0.089, crossing t_R=2.4e4 yr, kick Delta_e~0.6.
     Clean geometry (gap edge 2.1 AU, giant perihelion 2.5 AU).
     ~5e6 steps -> under an hour.

  2. "ultrafast": m_J=10 Mjup, a_J=5, e_J=0.2, embryo 1.2 AU, sigma0=500,
     tdep=1e4, f_e=0, t_int=2e4 yr.
     Predicted: u_R=0.48, crossing t_R=7.4e3 yr, kick De~0.34.
     Revision 2026-08-20: the original e_J=0.5 / embryo 1.5 AU version went
     unstable -- the giant's perihelion (2.5 AU) sat only ~1.4 Hill radii
     from the embryo. At e_J=0.2 / 1.2 AU the perihelion-apocenter clearance
     stays >= 3.2 r_H even after the kick, and u_R=0.48 leaves a factor-2
     margin from the u_R > 1 no-crossing boundary (the old 0.80 left ~20%).
     ~1.5e6 steps -> ~15 min.

Both use f_e=0 (no eccentricity damping) so the kick is undamped.
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

BASE_OUTPUT = REPO_DIR / "output" / "rebound_sr_extreme"

# (label, e_giant, m_giant_mjup, embryo_a0, sigma0_cgs, t_int [yr], dt_int [yr])
CASES = [
    ("robust_mj10_aemb1_sig1700", 0.5, 10.0, 1.0, 1700.0, 5.0e4, 25.0),
    ("ultrafast_ej02_mj10_aemb1.2_sig500", 0.2, 10.0, 1.2, 500.0, 2.0e4, 10.0),
]

A_GIANT = 5.0
TDEP = 1.0e4
F_E = 0.0


def build_extreme_runs() -> list[tuple]:
    runs = []
    for label, e_j, m_j, a_emb, sigma0_cgs, t_int, dt_int in CASES:
        args = Namespace(
            a_giants=str(A_GIANT),
            e_giants=str(e_j),
            tdeps=str(TDEP),
            output_dir=str(BASE_OUTPUT / label),
            ncpus=None,
            n_embryos=1,
            spacing="single",
            embryo_a0=a_emb,
            embryo_mass_earth=1.0,
            embryo_radius_earth=1.0,
            embryo_e=0.0,
            mstar=1.0,
            rstar=1.0,
            m_giant_mjup=m_j,
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
        # build_runs does not expose sigma0, so patch dens0 into force_params
        # (run_simulation's define_variables reads it in Msun/AU^2; the
        # g/cm^2 -> Msun/AU^2 conversion factor is 1.125e-7).
        for run in case_runs:
            run[17]["dens0"] = sigma0_cgs * 1.125e-7
        runs.extend(case_runs)
    return runs


def main() -> None:
    runs = build_extreme_runs()
    ncpus = min(parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE")), len(runs))
    BASE_OUTPUT.mkdir(parents=True, exist_ok=True)
    print(f"running {len(runs)} extreme SR runs with {ncpus} workers")
    for run in runs:
        print(f"  {run[17]['output_dir']}/{run[0]} (dens0={run[17]['dens0']:.3e} Msun/AU^2)")

    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(BASELINE.run_simulation, runs):
            print(f"finished: {result}", flush=True)


if __name__ == "__main__":
    main()
