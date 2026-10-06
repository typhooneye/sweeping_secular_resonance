#!/usr/bin/env python
"""Plot apsidal angles from a single REBOUND SimulationArchive."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rebound


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--giant-index", type=int, default=1)
    parser.add_argument("--embryo-index", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = rebound.Simulationarchive(str(args.archive))
    times = []
    giant_varpi = []
    embryo_varpi = []
    embryo_e = []
    embryo_a = []
    separation = []

    for sim in archive:
        star = sim.particles[0]
        giant = sim.particles[args.giant_index]
        embryo = sim.particles[args.embryo_index]
        giant_orbit = giant.orbit(primary=star)
        embryo_orbit = embryo.orbit(primary=star)
        dx = embryo.x - giant.x
        dy = embryo.y - giant.y
        dz = embryo.z - giant.z
        times.append(sim.t)
        giant_varpi.append(giant_orbit.pomega)
        embryo_varpi.append(embryo_orbit.pomega)
        embryo_e.append(embryo_orbit.e)
        embryo_a.append(embryo_orbit.a)
        separation.append(np.sqrt(dx * dx + dy * dy + dz * dz))

    times = np.asarray(times)
    giant_varpi = np.unwrap(np.asarray(giant_varpi))
    embryo_varpi = np.unwrap(np.asarray(embryo_varpi))
    delta_varpi = np.unwrap(embryo_varpi - giant_varpi)
    embryo_e = np.asarray(embryo_e)
    embryo_a = np.asarray(embryo_a)
    separation = np.asarray(separation)

    output = args.output
    if output is None:
        output = args.archive.with_name(args.archive.stem + "_omega_p.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    axes[0].plot(times / 1e6, embryo_varpi, label="rocky planet", lw=1.2)
    axes[0].plot(times / 1e6, giant_varpi, label="giant planet", lw=1.2)
    axes[0].set_ylabel(r"unwrapped $\varpi$ [rad]")
    axes[0].set_title("Apsidal angles")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(times / 1e6, delta_varpi, color="tab:purple", lw=1.2)
    axes[1].set_ylabel(r"$\Delta\varpi=\varpi_{\rm rocky}-\varpi_J$ [rad]")
    axes[1].set_title("Relative apsidal angle")
    axes[1].grid(alpha=0.25)

    axes[2].plot(times / 1e6, embryo_e, color="tab:orange", label="rocky eccentricity")
    axes[2].set_ylabel("eccentricity")
    axes[2].set_xlabel("Time [Myr]")
    axes[2].grid(alpha=0.25)
    separation_axis = axes[2].twinx()
    separation_axis.plot(
        times / 1e6,
        separation,
        color="tab:gray",
        alpha=0.75,
        label="planet separation",
    )
    separation_axis.set_ylabel("rocky-giant separation [AU]")

    fig.suptitle(
        f"{args.archive.name}: a={embryo_a[0]:.3f} AU initially, "
        f"min separation={separation.min():.3f} AU"
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)

    np.savetxt(
        output.with_suffix(".csv"),
        np.column_stack(
            (times, giant_varpi, embryo_varpi, delta_varpi, embryo_e, embryo_a, separation)
        ),
        delimiter=",",
        header="time_yr,giant_varpi_rad,embryo_varpi_rad,delta_varpi_rad,embryo_e,embryo_a_au,separation_au",
        comments="",
    )
    print(f"archive snapshots: {len(times)}")
    print(f"minimum planet separation: {separation.min():.6f} AU")
    print(f"embryo eccentricity range: {embryo_e.min():.6e} to {embryo_e.max():.6e}")
    print(f"figure: {output.resolve()}")


if __name__ == "__main__":
    main()
