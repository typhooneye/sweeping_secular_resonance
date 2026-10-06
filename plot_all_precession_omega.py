#!/usr/bin/env python
"""Plot omega(t) for every timestep and both precession tests in one figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_summary(path: Path) -> dict[tuple[str, float], dict[str, float]]:
    summary: dict[tuple[str, float], dict[str, float]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["test"], float(row["dt_yr"]))
            summary[key] = {
                "analytical_rate": float(row["analytical_rad_per_yr"]),
            }
    return summary


def read_series(path: Path) -> dict[float, dict[str, np.ndarray]]:
    data = np.load(path)
    series: dict[float, dict[str, np.ndarray]] = {}
    for key in data.files:
        if not key.startswith("dt_"):
            continue
        _, dt_text, quantity = key.split("_", 2)
        dt = float(dt_text)
        series.setdefault(dt, {})[quantity] = np.asarray(data[key])
    return series


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("output/precession_timestep_convergence"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figure/precession_omega_all_cases.png"),
    )
    args = parser.parse_args()

    summary = read_summary(args.input_dir / "precession_timestep_convergence.csv")
    giant = read_series(args.input_dir / "giant_only_omega.npz")
    disk = read_series(args.input_dir / "static_disk_only_omega.npz")

    timesteps = sorted(set(giant) | set(disk), reverse=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(timesteps)))

    fig, giant_axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    disk_axis = giant_axis.twinx()

    for color, dt in zip(colors, timesteps):
        if dt in giant:
            time = giant[dt]["time_yr"]
            omega = giant[dt]["omega_rad"]
            giant_axis.plot(
                time,
                omega - omega[0],
                color=color,
                lw=1.5,
                ls="-",
                label=f"Giant, dt={dt:g} yr",
            )
            rate = summary[("Giant only", dt)]["analytical_rate"]
            giant_axis.plot(time, rate * time, color=color, lw=0.9, ls=":")

        if dt in disk:
            time = disk[dt]["time_yr"]
            omega = disk[dt]["omega_rad"]
            disk_axis.plot(
                time,
                omega - omega[0],
                color=color,
                lw=1.5,
                ls="--",
                label=f"Disk, dt={dt:g} yr",
            )
            rate = summary[("Static disk only", dt)]["analytical_rate"]
            disk_axis.plot(time, rate * time, color=color, lw=0.9, ls="-.")

    giant_axis.set_xlabel("Time [yr]")
    giant_axis.set_ylabel(r"Giant-only $\omega(t)-\omega(0)$ [rad]")
    disk_axis.set_ylabel(r"Disk-only $\omega(t)-\omega(0)$ [rad]")
    giant_axis.set_title("Apsidal angle time series for all timestep cases")
    giant_axis.grid(alpha=0.25)

    handles_1, labels_1 = giant_axis.get_legend_handles_labels()
    handles_2, labels_2 = disk_axis.get_legend_handles_labels()
    giant_axis.legend(handles_1 + handles_2, labels_1 + labels_2, fontsize=8, ncol=2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
