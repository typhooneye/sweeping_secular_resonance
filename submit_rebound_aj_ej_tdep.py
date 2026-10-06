#!/usr/bin/env python

#SBATCH --job-name=ssr-aj-ej-tdep
#SBATCH --output=logs/ssr-aj-ej-tdep_%j.out
#SBATCH --error=logs/ssr-aj-ej-tdep_%j.err
#SBATCH --account=pi-abbot
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1

"""Submit the single-giant SSR REBOUND grid in parallel.

Default grid:
    a_J = 5, 10, 20, 30 AU
    e_J = 0.05, 0.5, 0.8
    T_dep = 1 Myr, 10 Myr

The default uses one Earth-mass embryo starting at 1 AU.  Increase
--n-embryos to build a chain starting from 1 AU.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


REPO_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent)).resolve()
BASELINE = SourceFileLoader(
    "rebound_ssr_type1_baseline",
    str(REPO_DIR / "rebound+SSR+type1_baseline.py"),
).load_module()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-giants", default="5,10,20,30")
    parser.add_argument("--e-giants", default="0.05,0.5,0.8")
    parser.add_argument("--tdeps", default="1e6,1e7", help="Comma-separated depletion times in yr.")
    parser.add_argument("--output-dir", default="output/rebound_aj_ej_tdep")
    parser.add_argument("--ncpus", type=int, default=None)
    parser.add_argument("--n-embryos", type=int, default=1)
    parser.add_argument(
        "--spacing",
        choices=["single", "hill", "2to1", "3to2", "random"],
        default="single",
        help="Embryo layout. 'single' requires --n-embryos 1.",
    )
    parser.add_argument("--embryo-a0", type=float, default=1.0)
    parser.add_argument("--embryo-mass-earth", type=float, default=1.0)
    parser.add_argument("--embryo-radius-earth", type=float, default=1.0)
    parser.add_argument("--embryo-e", type=float, default=0.0)
    parser.add_argument("--mstar", type=float, default=1.0)
    parser.add_argument("--rstar", type=float, default=1.0)
    parser.add_argument("--m-giant-mjup", type=float, default=1.0)
    parser.add_argument("--r-giant-rjup", type=float, default=1.0)
    parser.add_argument("--open-disk", type=int, choices=[0, 1], default=1)
    parser.add_argument("--open-typeI", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--k-hill", type=float, default=8.0)
    parser.add_argument("--k-gap-dep", type=float, default=0.0)
    parser.add_argument("--f-e", type=float, default=1.0)
    parser.add_argument("--t-int", type=float, default=2.0e7)
    parser.add_argument("--dt-int", type=float, default=1.0e3)
    parser.add_argument(
        "--dt-fraction",
        type=float,
        default=0.01,
        help="REBOUND timestep as a fraction of the initial embryo period (default P/100).",
    )
    parser.add_argument("--include-gr", type=int, choices=[0, 1], default=1)
    return parser.parse_args()


def spacing_mode(name: str) -> int:
    if name == "hill":
        return 0
    if name == "2to1":
        return 1
    if name == "3to2":
        return 2
    if name == "random":
        return 100
    return 1


def tdep_label(tdep: float) -> str:
    if tdep >= 1.0e6:
        return "%05.1fMyr" % (tdep / 1.0e6)
    if tdep >= 1.0e3:
        return "%05.1fkyr" % (tdep / 1.0e3)
    return "%05.1fyr" % tdep


def build_runs(args: argparse.Namespace) -> list[tuple]:
    if args.spacing == "single" and args.n_embryos != 1:
        raise ValueError("--spacing single requires --n-embryos 1")

    output_dir = str(Path(args.output_dir).resolve())
    hill_or_mmr = spacing_mode(args.spacing)
    runs = []

    for a_j in parse_float_list(args.a_giants):
        for e_j in parse_float_list(args.e_giants):
            for tdep in parse_float_list(args.tdeps):
                fname = (
                    "single_giant_aj_%05.2f_ej_%04.2f_tdep_%s_"
                    "nemb_%d_aemb_%04.2f.bin"
                    % (a_j, e_j, tdep_label(tdep), args.n_embryos, args.embryo_a0)
                )
                force_params = {
                    "tdep": tdep,
                    "k_gap_dep": args.k_gap_dep,
                    "f_e": args.f_e,
                    "include_gr": args.include_gr,
                    "use_gap": 1 if args.open_disk == 1 else 0,
                    "t_int": args.t_int,
                    "dt_int": args.dt_int,
                    "dt_fraction": getattr(args, "dt_fraction", 0.01),
                    "output_dir": output_dir,
                }
                runs.append(
                    (
                        fname,
                        args.k_hill,
                        1,
                        args.n_embryos,
                        args.mstar,
                        args.rstar,
                        np.array([args.m_giant_mjup]),
                        np.array([args.r_giant_rjup]),
                        np.array([a_j]),
                        np.array([e_j]),
                        args.embryo_mass_earth,
                        args.embryo_radius_earth,
                        args.embryo_a0,
                        args.embryo_e,
                        hill_or_mmr,
                        args.open_disk,
                        args.open_typeI,
                        force_params,
                    )
                )
    return runs


def parse_slurm_cpus(value: str | None) -> int:
    if not value:
        return multiprocessing.cpu_count()

    total = 0
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "(x" in item:
            cpus, repeat = item.split("(x", 1)
            total += int(cpus) * int(repeat.rstrip(")"))
        else:
            total += int(item)
    return total if total > 0 else multiprocessing.cpu_count()


def main() -> None:
    args = parse_args()
    runs = build_runs(args)
    ncpus = args.ncpus
    if ncpus is None:
        ncpus = parse_slurm_cpus(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"running {len(runs)} simulations with {ncpus} workers")
    print(f"output_dir={Path(args.output_dir).resolve()}")

    with multiprocessing.Pool(ncpus) as pool:
        for result in pool.starmap(BASELINE.run_simulation, runs):
            print(result, flush=True)


if __name__ == "__main__":
    main()
