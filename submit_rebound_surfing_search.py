#!/usr/bin/env python

#SBATCH --job-name=surfing-search
#SBATCH --output=logs/surfing-search_%j.out
#SBATCH --error=logs/surfing-search_%j.err
#SBATCH --account=pi-abbot
#SBATCH --time=35:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

"""Parameter search for resonance-surfing SR crossings.

SHARDING: running this script (either `python submit_rebound_surfing_search.py`
or `sbatch submit_rebound_surfing_search.py`) with no SURF_SHARD in the
environment is a dispatch call -- it submits NSHARDS=6 separate sbatch jobs
(SURF_SHARD=0..5, each with its own --ntasks sized to its share of the runs,
    ~54 each instead of one 324-task job) and exits immediately without running
any simulation itself.  The 324 runs are split by round-robining the 108
(m_j, a_emb, f_e, e_jup, gap) combos across the 6 shards, and every combo
brings its full set of 3 tau_n values with it -- so each shard gets a
balanced mix of short and long tau_n runs rather than one shard inheriting
all the slow tau_n=3e6 cases.  Each shard job auto-resubmits (chains)
independently, exactly as before.

Grid (324 runs), each with four embryos swept by a sweeping secular resonance
from one giant:

    m_J     in {1, 10} Mjup             giant mass
    a_inner in {1.0, 1.5, 2.0} AU       innermost embryo semimajor axis
    f_e     in {1.0, 0.5, 0.0}          tidal e-damping strength
                                         (f_e=0: damping force off -- the
                                         undamped control, see typeI() in
                                         the baseline)
    e_J     in {0.1, 0.3, 0.6}          giant eccentricity
    tau_n   in {3e5, 1e6, 3e6} yr       disk-depletion time
    gap     in {no gap, empty gap}      giant clears a Hill-based gap or not

System: Sun-like star (1 Msun, 1 Rsun), giant at 10 AU (e_J swept above),
four 1 Mearth embryos at e=0, separated by 6 mutual Hill radii and initially
spaced by 90 degrees in true anomaly. Giant radius fixed at 1 Rjup for all masses --
physically reasonable since cold gas-giant radii are nearly mass-independent
over 1-13 Mjup (electron-degeneracy support).

Disk: MMSN (Sigma0=1700 g/cm^2, k=1.5), h0=0.025 AU at 1 AU (hi=1.25),
zk=exact(k), no gap: use_gap=0; empty gap: use_gap=1, k_gap_dep=0.0 (Hill-
based edges).  GR on (include_gr=1), radial eccentricity damping only
(no independent Type-I migration torque).  dt = 0.1 * embryo period.

t_int = 12*tau_n for every run, matching submit_rebound_nlt05_repro.py's
convention -- covers the crossing (roughly 2-6 tau_n depending on a_emb/m_J)
plus several tau_n of post-crossing evolution.  Snapshot every t_int/5000.

AUTO-RESUBMISSION: identical scheme to submit_rebound_nlt05_repro.py, run
per shard.  Each shard job queues its own successor (same SURF_SHARD, via
--export=ALL) with --dependency=afterany before integrating; every run has
"resume": 1; a shard that finds all its archives finished exits without
chaining.  Cap: MAX_CHAIN successors per shard; disable with
SURF_NO_AUTO_RESUBMIT=1.
t_int = 12*tau_n reaches 3.6e7 yr for the tau_n=3e6 cell (30x longer than the
shortest cell), so individual runs take correspondingly longer; size any
--time SBATCH directive accordingly.
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

SCRIPT_PATH = REPO_DIR / "submit_rebound_surfing_search.py"
OUTPUT_DIR = REPO_DIR / "output" / "surfing_search_4emb_m1_kmut6"

MSTAR = 1.0
RSTAR = 1.0
A_JUP = 10.0       # AU
R_JUP = 1.0        # Rjup, ~mass-independent for cold gas giants
N_EMB = 4
M_EMB = 1.0        # Mearth
R_EMB = 1.0        # Rearth
MUTUAL_HILL_SPACING = 6.0
# The baseline's legacy k gives Delta a / mean(a) = 2 k (m/3M*)^(1/3).
LEGACY_HILL_K = MUTUAL_HILL_SPACING * 2.0 ** (1.0 / 3.0) / 2.0
EMBRYO_TRUE_ANOMALIES = tuple(2.0 * np.pi * i / N_EMB for i in range(N_EMB))
DT_FRACTION = 0.1  # dt = 0.1 * embryo period
T_INT_OVER_TDEP = 12.0
MAX_CHAIN = 6
NSHARDS = 6

MJ_LIST = (1.0, 10.0)
A_EMB_LIST = (1.0, 1.5, 2.0)
FE_LIST = (1.0, 0.5, 0.0)
E_JUP_LIST = (0.1, 0.3, 0.6)
TDEP_LIST = (3.0e5, 1.0e6, 3.0e6)
GAP_LIST = (False, True)

BASE_FORCE_PARAMS = {
    "dens0": 1700.0 * 1.125e-7,   # g/cm^2 -> Msun/AU^2
    "di": 1.5,
    "h0": 0.025,
    "hi": 1.25,
    "zk": None,                   # exact z_k(di=1.5)
    "disk_inner_cutoff": 0.0,
    "n_terms": 30,
    "c_code": 63241.077,
    "include_gr": 1,
    "f_a": 1.0,
    "dt_fraction": DT_FRACTION,
    "resume": 1,
}

# 108 (gap, m_j, a_emb, f_e, e_jup) combos; each expands to len(TDEP_LIST)=3 runs.
COMBOS = [
    (m_j, a_emb, f_e, e_jup, gap)
    for gap in GAP_LIST
    for m_j in MJ_LIST
    for a_emb in A_EMB_LIST
    for f_e in FE_LIST
    for e_jup in E_JUP_LIST
]

CASES = [
    (m_j, a_emb, f_e, e_jup, tdep, gap)
    for (m_j, a_emb, f_e, e_jup, gap) in COMBOS
    for tdep in TDEP_LIST
]


def combos_for_shard(shard: int) -> list[tuple]:
    # Round-robin over combos (not individual cases) so every shard gets a
    # full spread of tau_n values -- balances both task count and runtime.
    return [combo for i, combo in enumerate(COMBOS) if i % NSHARDS == shard]


def cases_for_shard(shard: int) -> list[tuple]:
    return [
        (m_j, a_emb, f_e, e_jup, tdep, gap)
        for (m_j, a_emb, f_e, e_jup, gap) in combos_for_shard(shard)
        for tdep in TDEP_LIST
    ]


def build_surfing_runs(cases: list[tuple]) -> list[tuple]:
    runs = []
    for m_j, a_emb, f_e, e_jup, tdep, gap in cases:
        gap_tag = "gap" if gap else "nogap"
        fname = (
            f"SURF_mJ_{m_j:g}_aemb_{a_emb:.2f}_nemb_{N_EMB}_memb_{M_EMB:g}_"
            f"kmut_{MUTUAL_HILL_SPACING:g}_fe_{f_e:g}_ej_{e_jup:g}_tdep_{tdep:.0e}.bin"
        )
        t_int = T_INT_OVER_TDEP * tdep
        force_params = dict(BASE_FORCE_PARAMS)
        force_params["tdep"] = tdep
        force_params["use_gap"] = 1 if gap else 0
        force_params["k_gap_dep"] = 0.0     # empty gap when use_gap=1
        force_params["f_e"] = f_e
        force_params["t_int"] = t_int
        force_params["dt_int"] = t_int / 5000.0
        force_params["output_dir"] = str(OUTPUT_DIR / gap_tag)
        force_params["embryo_true_anomalies"] = EMBRYO_TRUE_ANOMALIES
        runs.append((
            fname,
            LEGACY_HILL_K,            # exactly 6 mutual Hill radii
            1,                        # n_j
            N_EMB,
            MSTAR,
            RSTAR,
            np.array([m_j]),
            np.array([R_JUP]),
            np.array([A_JUP]),
            np.array([e_jup]),
            M_EMB,
            R_EMB,
            a_emb,
            0.0,                      # e_p2
            0,                        # mutual-Hill spacing
            1,                        # open_disk
            1,                        # radial eccentricity damping only
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


def maybe_submit_continuation(shard: int) -> None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return
    if os.environ.get("SURF_NO_AUTO_RESUBMIT") == "1":
        print(f"shard {shard}: auto-resubmission disabled by SURF_NO_AUTO_RESUBMIT=1")
        return
    marker = REPO_DIR / "logs" / f".surf_continuation_{job_id}"
    if marker.exists():
        print(f"shard {shard}: continuation already queued for job {job_id} (requeue) -- skipping")
        return
    chain = int(os.environ.get("SURF_CHAIN", "0"))
    if chain >= MAX_CHAIN:
        print(f"shard {shard}: chain cap reached ({chain} continuations): not resubmitting")
        return
    ntasks = len(cases_for_shard(shard))
    cmd = [
        "sbatch",
        f"--ntasks={ntasks}",
        f"--dependency=afterany:{job_id}",
        f"--export=ALL,SURF_SHARD={shard},SURF_CHAIN={chain + 1}",
        str(SCRIPT_PATH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        marker.touch()
        print(f"shard {shard}: queued continuation job (chain {chain + 1}): {result.stdout.strip()}")
    else:
        print(f"WARNING: shard {shard}: continuation sbatch failed: {result.stderr.strip()}")


def dispatch_shards() -> None:
    """Submit NSHARDS sbatch jobs, each sized to its own share of the runs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for shard in range(NSHARDS):
        ntasks = len(cases_for_shard(shard))
        cmd = [
            "sbatch",
            f"--ntasks={ntasks}",
            f"--export=ALL,SURF_SHARD={shard}",
            str(SCRIPT_PATH),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"submitted shard {shard} ({ntasks} tasks): {result.stdout.strip()}")
        else:
            print(f"WARNING: sbatch failed for shard {shard}: {result.stderr.strip()}")


def run_shard(shard: int) -> None:
    runs = build_surfing_runs(cases_for_shard(shard))

    unfinished = [run for run in runs if not run_is_finished(run)]
    if not unfinished:
        print(f"shard {shard}: all {len(runs)} surfing-search runs finished -- nothing to do")
        return

    maybe_submit_continuation(shard)

    ncpus = min(parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE")), len(unfinished))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"shard {shard}: running {len(unfinished)}/{len(runs)} surfing-search runs with {ncpus} workers")
    for run in unfinished:
        fp = run[17]
        print(f"  {os.path.basename(fp['output_dir'])}/{run[0]}  t_int={fp['t_int']:.1e}")

    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(BASELINE.run_simulation, unfinished):
            print(f"finished: {result}", flush=True)


def main() -> None:
    shard_env = os.environ.get("SURF_SHARD")
    if shard_env is None:
        dispatch_shards()
        return
    run_shard(int(shard_env))


if __name__ == "__main__":
    main()
