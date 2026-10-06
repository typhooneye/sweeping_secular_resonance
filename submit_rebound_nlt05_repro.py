#!/usr/bin/env python

#SBATCH --job-name=nlt05-repro
#SBATCH --output=logs/nlt05-repro_%j.out
#SBATCH --error=logs/nlt05-repro_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --time=35:00:00

"""Reproduce single-embryo cases from Nagasawa, Lin & Thommes (2005),
ApJ 635, 578 ("Dynamical Shake-up of Planetary Systems. I.").

Their setup mapped onto run_simulation (revised 2026-08-23: Saturn
removed, tau_n shortened to keep the runs affordable):
  * Star 1 Msun; Jupiter only (1 Mjup, 5.203 AU), e_J = 0.01 or 0.048
    depending on the case (their Figs. 3-4).  WITHOUT Saturn the
    late-time nu_5 has no anchor (Jupiter's residual precession comes
    only from the embryo), so these runs probe the disk-driven sweep
    itself -- which also makes them exactly the single-giant gapless
    configuration of ssr_single_giant.py, so the analytics apply
    directly.  Predicted crossings (gap=False, GR off, any e_J):
      a_emb [AU]:   0.7    1.0    1.5    2.0
      u_R:        3.8e-3 1.0e-2 3.5e-2 9.6e-2
      t_R/tau_n:   5.58   4.58   3.35   2.34
    i.e. the resonance sweeps INWARD, crossing 2 AU first.
  * MMSN disk: Sigma = 1700 (a/1AU)^-1.5 exp(-t/tau_n) g/cm^2, NO gap
    (their baseline; gap depletion is a separate NLT05 Sec. 3.4
    experiment) -> use_gap=0, so the axisymmetric disk potential acts on
    the giants AND the embryo, as in the paper.  The sweeping nu_5
    resonance comes out of the dynamics: Jupiter's precession (disk +
    Saturn) falls as the disk drains, and the resonance sweeps through
    the terrestrial region.
  * Type-I eccentricity damping (their eq. [8]):
    tau_damp = (M*/m)(M*/Sigma a^2)(H/a)^4 / Omega_K with
    H = 0.05 (r/1AU)^(5/4) AU -- identical to the baseline t_damp with
    h0=0.05, hi=1.25, f_e=1.
  * No GR (not in their force model).
  * dt = 0.1 * innermost-planet (embryo) period.

Cases (20 runs), all with t_int = 12*tau_n:
  fig3/  Resonance-passage timing vs. embryo position (their Fig. 3):
         m=0.03 Mearth at a = 0.7, 1.5, 2.0 AU; e_J=0.01;
         tau_n = 0.1, 1 Myr.  (a=1 AU is covered by fig4.)
  fig4/  Kick vs. embryo mass and e_J at 1 AU (their Fig. 4):
         m = 0.03, 0.3, 1.0 Mearth x e_J = 0.01, 0.048;
         tau_n = 0.1, 1 Myr.
  fig7/  Depletion-rate dependence (their Fig. 7): m=0.3 Mearth at
         1 AU, e_J=0.048; tau_n = 0.5, 5 Myr (the 1 Myr point of the
         tau_n sequence is fig4's m=0.3/e_J=0.048/tau_n=1e6 run).

NLT05 used tau_n = 10 Myr and 100 Myr integrations for Figs. 3-4; the
shortened tau_n keeps the sweep well inside the adiabatic regime
(tdep=1e5 gave clean kicks in the demo grid) at ~1/10 the cost, but
note the tau_n=0.1 Myr sweeps give sqrt(tdep)-smaller kicks and are
too fast for the resonance TRAPPING/migration behavior NLT05 emphasize
-- treat tau_n=1 Myr as the primary reproduction and 0.1 Myr as the
fast-sweep comparison.

Approximations vs. the paper: no Saturn; embryo radii scale as m^(1/3)
of Earth's; disk forces use our verified Laplace-series implementation
rather than their exact potential.

Cost: most runs are <= ~1.2e8 steps (<= 17 h).  The two long ones --
fig3 a=0.7/tau_n=1 Myr (~2e8 steps, ~28 h) and fig7 tau_n=5 Myr
(~6e8 steps, ~3.5 days) -- are carried across wall windows by the
automatic chaining below.

AUTO-RESUBMISSION: identical scheme to submit_rebound_hr5183_repro.py.
Under SLURM the job queues a successor with --dependency=afterany
before integrating; every run has "resume": 1; a job that finds all
archives finished exits without chaining.  Cap: MAX_CHAIN successors;
disable with NLT05_NO_AUTO_RESUBMIT=1.
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

SCRIPT_PATH = REPO_DIR / "submit_rebound_nlt05_repro.py"
OUTPUT_DIR = REPO_DIR / "output" / "nlt05_repro"

MSTAR = 1.0
RSTAR = 1.0
M_JUP, R_JUP, A_JUP = 1.0, 1.0, 5.203          # Mjup, Rjup, AU
DT_FRACTION = 0.1                              # dt = 0.1 * embryo period
MAX_CHAIN = 12

BASE_FORCE_PARAMS = {
    "dens0": 1700.0 * 1.125e-7,   # g/cm^2 -> Msun/AU^2
    "di": 1.5,                    # MMSN power law
    "h0": 0.05,                   # H = 0.05 (r/1AU)^(5/4) AU  (NLT05)
    "hi": 1.25,
    "zk": 1.094,
    "use_gap": 0,                 # NLT05 baseline: no gap
    "include_gr": 0,              # not in their force model
    "f_e": 1.0,
    "f_a": 1.0,
    "dt_fraction": DT_FRACTION,
    "resume": 1,
}

T_INT_OVER_TDEP = 12.0

# (group, m_emb [Mearth], a_emb [AU], e_jup, tdep [yr])
CASES = (
    # fig3: passage timing vs. position
    [("fig3", 0.03, a_emb, 0.01, tdep)
     for tdep in (1.0e5, 1.0e6) for a_emb in (0.7, 1.5, 2.0)]
    # fig4: mass and e_J dependence at 1 AU
    + [("fig4", m_emb, 1.0, e_jup, tdep)
       for tdep in (1.0e5, 1.0e6) for e_jup in (0.01, 0.048)
       for m_emb in (0.03, 0.3, 1.0)]
    # fig7: depletion-rate dependence (1e6 point covered by fig4)
    + [("fig7", 0.3, 1.0, 0.048, tdep) for tdep in (5.0e5, 5.0e6)]
)


def build_nlt05_runs() -> list[tuple]:
    runs = []
    for group, m_emb, a_emb, e_jup, tdep in CASES:
        fname = (
            f"NLT05_memb_{m_emb:g}_aemb_{a_emb:.2f}_ejup_{e_jup:g}_"
            f"tdep_{tdep:.0e}.bin"
        )
        t_int = T_INT_OVER_TDEP * tdep
        force_params = dict(BASE_FORCE_PARAMS)
        force_params["tdep"] = tdep
        force_params["t_int"] = t_int
        force_params["dt_int"] = t_int / 5000.0   # ~5000 snapshots per run
        force_params["output_dir"] = str(OUTPUT_DIR / group)
        runs.append((
            fname,
            8,                        # k (Hill spacing; unused, single embryo)
            1,                        # n_j: Jupiter only (no Saturn)
            1,                        # n_p2
            MSTAR,
            RSTAR,
            np.array([M_JUP]),
            np.array([R_JUP]),
            np.array([A_JUP]),
            np.array([e_jup]),
            m_emb,                    # Mearth
            m_emb ** (1.0 / 3.0),     # Rearth, r ~ m^(1/3)
            a_emb,
            0.0,                      # e_p2
            0,                        # Hill_or_MMR
            1,                        # open_disk
            1,                        # open_typeI
            force_params,
        ))
    return runs


def run_is_finished(run: tuple) -> bool:
    import rebound

    fp = run[17]
    archive_path = Path(fp["output_dir"]) / run[0]
    if not archive_path.exists():
        return False
    try:
        sim = rebound.Simulation(str(archive_path))  # last snapshot
    except Exception:
        return False
    return sim.t >= fp["t_int"] - fp["dt_int"]


def maybe_submit_continuation() -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return
    if os.environ.get("NLT05_NO_AUTO_RESUBMIT") == "1":
        print("auto-resubmission disabled by NLT05_NO_AUTO_RESUBMIT=1")
        return
    chain = int(os.environ.get("NLT05_CHAIN", "0"))
    if chain >= MAX_CHAIN:
        print(f"chain cap reached ({chain} continuations): not resubmitting")
        return
    cmd = [
        "sbatch",
        f"--dependency=afterany:{job_id}",
        f"--export=ALL,NLT05_CHAIN={chain + 1}",
        str(SCRIPT_PATH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"queued continuation job (chain {chain + 1}): {result.stdout.strip()}")
    else:
        print(f"WARNING: continuation sbatch failed: {result.stderr.strip()}")


def main() -> None:
    runs = build_nlt05_runs()

    unfinished = [run for run in runs if not run_is_finished(run)]
    if not unfinished:
        print(f"all {len(runs)} NLT05 runs finished -- nothing to do")
        return

    maybe_submit_continuation()

    ncpus = min(parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE")), len(unfinished))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"running {len(unfinished)}/{len(runs)} NLT05 reproduction runs with {ncpus} workers")
    for run in unfinished:
        fp = run[17]
        print(f"  {os.path.basename(fp['output_dir'])}/{run[0]}  t_int={fp['t_int']:.1e}")

    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(BASELINE.run_simulation, unfinished):
            print(f"finished: {result}", flush=True)


if __name__ == "__main__":
    main()
