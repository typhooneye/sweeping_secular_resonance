#!/usr/bin/env python

#SBATCH --job-name=hr5183-repro
#SBATCH --output=logs/hr5183-repro_%j.out
#SBATCH --error=logs/hr5183-repro_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=1
#SBATCH --time=35:00:00

"""Reproduce the mycode/0702/1pt1M_rc HR 5183 runs on the current baseline.

Source: 4.SSR/mycode/0702/1pt1M_rc/0531_Multi_HR5183_1pa{05,055withoutdisk,
07,13,17,20}.py + input_4planets.  The six originals differ only in the
embryo semimajor axis; the 0.55 AU one had every additional force
commented out (pure N-body).  Each case is run TWICE: with GR off
(matching the original script, which never implemented a GR force even
though input_4planets set open_gr=1) and with GR on -> 12 runs, written
to gr0/ and gr1/ subdirectories.

System (from input_4planets): mstar=1.07 Msun, rstar=1.53 Rsun;
giant 3.23 Mjup, r=1.47 Rjup, a=18 AU, e=0.84 (perihelion 2.88 AU,
gap inner edge ~2.6 AU -- all embryos are interior to the gap);
embryo 1 Mearth, 1 Rearth, e=0.
Disk: sigma0=50 g/cm^2, k(di)=1.01, h0=0.05, hi=1.25, zk=exact(k),
tdep=1e6 yr, empty gap (no depletion), type-I e-damping f_e=1.
t_int=1e7 yr, archive every 1e3 yr, dt = 0.1 * innermost-planet period
(the originals used the same fraction).

Deliberate differences from the originals (not reproduced), per the
2026-08-23 direct-integration audit of the disk force:
  * embryo local disk term: the originals used -4 pi G Sigma z_k, a
    factor ~2.2 too strong; these runs use the verified
    -2 pi G Sigma z_k/(2-k).  (The originals' gap-edge series were
    CORRECT; the baseline's halved edge terms were fixed 2026-08-23.)
  * z_k: exact value for k=1.01 is 1.0038; the originals' 1.094 (the
    k=1.5 value) made the local term ~9% strong.
  * empty gap removes only the annulus [r_in, r_out]; the originals
    truncated the whole disk beyond the gap inner edge.
  * giant initial angles (omega=-0.35, f=1.50) are zeroed by
    run_simulation; with an axisymmetric disk this is a rotation +
    orbital-phase shift with no secular consequence.

AUTO-RESUBMISSION: when running under SLURM, the job submits a successor
with --dependency=afterany:<this job> BEFORE starting the integrations,
so a wall-limit kill is followed automatically by a continuation job
(all runs set "resume": 1 and pick up from their archives).  A job that
finds every archive already at t_int exits immediately WITHOUT chaining,
which ends the chain -- expect one short no-op job at the end.  The
chain is capped at MAX_CHAIN successors; set HR5183_NO_AUTO_RESUBMIT=1
in the environment to disable chaining and resubmit by hand instead.
"""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent)).resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from submit_rebound_aj_ej_tdep import BASELINE, parse_slurm_cpus

SCRIPT_PATH = REPO_DIR / "submit_rebound_hr5183_repro.py"
OUTPUT_DIR = REPO_DIR / "output" / "hr5183_1pt1M_repro"

MSTAR = 1.07          # Msun
RSTAR = 1.53          # Rsun
M_GIANT = 3.23        # Mjup
R_GIANT = 1.47        # Rjup
A_GIANT = 18.0        # AU
E_GIANT = 0.84
T_INT = 1.0e7         # yr
DT_INT = 1.0e3        # yr
DT_FRACTION = 0.1     # dt = 0.1 * innermost (embryo) period
GR_SETS = (0, 1)
MAX_CHAIN = 8         # safety cap on automatic continuation jobs

BASE_FORCE_PARAMS = {
    "dens0": 50.0 * 1.125e-7,  # 50 g/cm^2 in Msun/AU^2
    "di": 1.01,
    "h0": 0.05,
    "hi": 1.25,
    "zk": None,   # exact z_k(di=1.01) = 1.0038 (legacy 1.094 was ~9% strong)
    "tdep": 1.0e6,
    "k_gap_dep": 0.0,          # fully empty gap, as in the originals
    "use_gap": 1,
    "f_e": 1.0,
    "f_a": 1.0,
    "t_int": T_INT,
    "dt_int": DT_INT,
    "dt_fraction": DT_FRACTION,
    "resume": 1,
}

# (embryo a [AU], open_disk, open_typeI, tag)
CASES = [
    (0.5, 1, 1, ""),
    (0.55, 0, 0, "_noforce"),
    (0.7, 1, 1, ""),
    (1.3, 1, 1, ""),
    (1.7, 1, 1, ""),
    (2.0, 1, 1, ""),
]


def build_repro_runs() -> list[tuple]:
    runs = []
    for include_gr in GR_SETS:
        for a_emb, open_disk, open_typeI, tag in CASES:
            fname = f"HR5183_1pt1M_aemb_{a_emb:.2f}{tag}.bin"
            force_params = dict(BASE_FORCE_PARAMS)
            force_params["include_gr"] = include_gr
            # Filenames do not encode GR, so each set gets its own subdir.
            force_params["output_dir"] = str(OUTPUT_DIR / f"gr{include_gr}")
            runs.append((
                fname,
                8,                      # k (Hill spacing; unused for 1 embryo)
                1,                      # n_j
                1,                      # n_p2
                MSTAR,
                RSTAR,
                np.array([M_GIANT]),
                np.array([R_GIANT]),
                np.array([A_GIANT]),
                np.array([E_GIANT]),
                1.0,                    # m_p2 [Mearth]
                1.0,                    # r_p2 [Rearth]
                a_emb,
                0.0,                    # e_p2
                0,                      # Hill_or_MMR (single embryo)
                open_disk,
                open_typeI,
                force_params,
            ))
    return runs


def run_is_finished(run: tuple) -> bool:
    import rebound

    archive_path = Path(run[17]["output_dir"]) / run[0]
    if not archive_path.exists():
        return False
    try:
        sim = rebound.Simulation(str(archive_path))  # last snapshot
    except Exception:
        return False
    # The last archived snapshot can trail sim.t by up to one output
    # interval, so accept anything within DT_INT of t_int.
    return sim.t >= T_INT - DT_INT


def maybe_submit_continuation() -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return  # not under SLURM: nothing to chain
    if os.environ.get("HR5183_NO_AUTO_RESUBMIT") == "1":
        print("auto-resubmission disabled by HR5183_NO_AUTO_RESUBMIT=1")
        return
    chain = int(os.environ.get("HR5183_CHAIN", "0"))
    if chain >= MAX_CHAIN:
        print(f"chain cap reached ({chain} continuations): not resubmitting")
        return
    cmd = [
        "sbatch",
        f"--dependency=afterany:{job_id}",
        f"--export=ALL,HR5183_CHAIN={chain + 1}",
        str(SCRIPT_PATH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"queued continuation job (chain {chain + 1}): {result.stdout.strip()}")
    else:
        print(f"WARNING: continuation sbatch failed: {result.stderr.strip()}")


def main() -> None:
    runs = build_repro_runs()

    unfinished = [run for run in runs if not run_is_finished(run)]
    if not unfinished:
        print(f"all {len(runs)} runs already at t_int={T_INT:.1e} yr -- nothing to do")
        return

    # Queue the continuation BEFORE integrating, so a wall-limit kill
    # cannot prevent it.  The successor no-ops (above) once all finish.
    maybe_submit_continuation()

    ncpus = min(parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE")), len(unfinished))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"running {len(unfinished)}/{len(runs)} unfinished HR5183 runs with {ncpus} workers")
    for run in unfinished:
        print(f"  {os.path.basename(run[17]['output_dir'])}/{run[0]}  "
              f"(open_disk={run[15]}, open_typeI={run[16]}, gr={run[17]['include_gr']})")

    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(BASELINE.run_simulation, unfinished):
            print(f"finished: {result}", flush=True)


if __name__ == "__main__":
    main()
