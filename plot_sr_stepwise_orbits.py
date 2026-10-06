#!/usr/bin/env python
"""Plot a, a(1-e), a(1+e) time series from diagnose_sr_stepwise.py archives.

Reads the same .bin SimulationArchives as plot_sr_stepwise_results.py and, for
the tracked body of each test, plots the semimajor axis together with the
perihelion a(1-e) and apohelion a(1+e) envelope. Tolerates missing or
still-growing archives. Does not run any simulation.
"""

from __future__ import annotations

import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rebound

REPO_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent)).resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

STEPWISE = SourceFileLoader(
    "diagnose_sr_stepwise_for_orbit_plot",
    str(REPO_DIR / "diagnose_sr_stepwise.py"),
).load_module()


def read_orbit_series(bin_path: Path, track_index: int):
    if not bin_path.exists():
        print(f"  {bin_path.name}: not found, skipping")
        return None
    try:
        sa = rebound.Simulationarchive(str(bin_path))
    except Exception as exc:
        print(f"  {bin_path.name}: could not open ({exc}), skipping")
        return None

    times, a, e = [], [], []
    try:
        for sim_snap in sa:
            orbit = sim_snap.particles[track_index].orbit(primary=sim_snap.particles[0])
            times.append(sim_snap.t)
            a.append(orbit.a)
            e.append(orbit.e)
    except Exception as exc:
        print(f"  {bin_path.name}: read stopped after {len(times)} snapshots ({exc})")

    if len(times) < 2:
        print(f"  {bin_path.name}: only {len(times)} snapshot(s), skipping")
        return None
    print(f"  {bin_path.name}: {len(times)} snapshots, t=[{times[0]:.3g}, {times[-1]:.3g}] yr")
    return np.array(times), np.array(a), np.array(e)


def main() -> None:
    args = STEPWISE.parse_args()
    output_dir = args.output_dir.resolve()
    configs = STEPWISE.build_configs(args)

    print(f"reading archives from {output_dir}")
    series = []
    for config in configs:
        data = read_orbit_series(output_dir / config.fname, config.track_index)
        if data is not None:
            series.append((config.name, *data))

    if not series:
        print("no usable archives found -- nothing to plot")
        return

    fig, axes = plt.subplots(
        1, len(series), figsize=(4.5 * len(series), 4.5), constrained_layout=True
    )
    for axis, (name, time, a, e) in zip(np.atleast_1d(axes), series):
        peri = a * (1.0 - e)
        apo = a * (1.0 + e)
        axis.plot(time, a, color="tab:blue", lw=1.5, label=r"$a$")
        axis.plot(time, peri, color="tab:orange", lw=1.0, label=r"$a(1-e)$")
        axis.plot(time, apo, color="tab:green", lw=1.0, label=r"$a(1+e)$")
        axis.fill_between(time, peri, apo, color="tab:blue", alpha=0.08)
        axis.set_title(name)
        axis.set_xlabel("Time [yr]")
        axis.set_ylabel("Distance [AU]")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure_path = REPO_DIR / "figure" / "diagnose_sr_stepwise_orbits.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    print(f"figure={figure_path}")


if __name__ == "__main__":
    main()
