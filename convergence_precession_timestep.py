#!/usr/bin/env python
"""Run the isolated precession tests for several timesteps.

The saved ``omega`` is REBOUND's planar longitude of periapse ``pomega``.
For these coplanar simulations this is the apsidal angle used to measure
secular precession.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import diagnose_precession_rates as diagnostic


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def save_time_series(
    records: dict[str, list[dict[str, np.ndarray | float]]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for test_name, test_records in records.items():
        safe_name = test_name.lower().replace(" ", "_")
        payload: dict[str, np.ndarray] = {}
        for record in test_records:
            key = f"dt_{record['dt']:.8g}"
            payload[f"{key}_time_yr"] = np.asarray(record["time"])
            payload[f"{key}_omega_rad"] = np.asarray(record["omega"])
            payload[f"{key}_eccentricity"] = np.asarray(record["eccentricity"])
        np.savez(output_dir / f"{safe_name}_omega.npz", **payload)


def save_summary(
    records: dict[str, list[dict[str, np.ndarray | float]]],
    output_dir: Path,
) -> None:
    with (output_dir / "precession_timestep_convergence.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "test",
                "dt_yr",
                "measured_rad_per_yr",
                "analytical_rad_per_yr",
                "relative_error",
            ]
        )
        for test_name, test_records in records.items():
            for record in test_records:
                writer.writerow(
                    [
                        test_name,
                        record["dt"],
                        record["measured_rate"],
                        record["analytical_rate"],
                        record["relative_error"],
                    ]
                )


def plot_time_series(
    records: dict[str, list[dict[str, np.ndarray | float]]],
    output_dir: Path,
) -> None:
    for test_name, test_records in records.items():
        fig, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
        for record in test_records:
            time = np.asarray(record["time"])
            omega = np.asarray(record["omega"])
            axis.plot(time, omega - omega[0], lw=1.2, label=f"dt={record['dt']:g} yr")

        analytical_rate = float(test_records[0]["analytical_rate"])
        longest_time = max(float(np.asarray(record["time"])[-1]) for record in test_records)
        axis.plot(
            [0.0, longest_time],
            [0.0, analytical_rate * longest_time],
            "k--",
            lw=1.5,
            label="Analytical",
        )
        axis.set_title(f"{test_name}: timestep convergence")
        axis.set_xlabel("Time [yr]")
        axis.set_ylabel(r"$\omega(t)-\omega(0)$ [rad]")
        axis.grid(alpha=0.25)
        axis.legend()
        safe_name = test_name.lower().replace(" ", "_")
        fig.savefig(output_dir / f"{safe_name}_omega_vs_time.png", dpi=220)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timesteps",
        default="0.1,0.05,0.02,0.01,0.005,0.0025",
        help="Comma-separated REBOUND timesteps in years.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output/precession_timestep_convergence"))
    parser.add_argument("--giant-duration", type=float, default=1.0e5)
    parser.add_argument("--disk-duration", type=float, default=5.0e3)
    parser.add_argument("--giant-sample-interval", type=float, default=20.37)
    parser.add_argument("--disk-sample-interval", type=float, default=1.73)
    parser.add_argument("--trim-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timesteps = parse_float_list(args.timesteps)
    if not timesteps or any(dt <= 0.0 for dt in timesteps):
        raise ValueError("All timesteps must be positive.")

    records: dict[str, list[dict[str, np.ndarray | float]]] = {
        "Giant only": [],
        "Static disk only": [],
    }

    for dt in timesteps:
        test_args = argparse.Namespace(
            rocky_a=1.0,
            rocky_e=0.01,
            rocky_mass_earth=1.0,
            giant_a=5.0,
            giant_mass_mjup=1.0,
            sigma0_cgs=1700.0,
            disk_k=1.5,
            z_k=1.094,
            disk_inner_cutoff=0.01,
            n_terms=30,
            dt=dt,
            giant_duration=args.giant_duration,
            disk_duration=args.disk_duration,
            giant_sample_interval=args.giant_sample_interval,
            disk_sample_interval=args.disk_sample_interval,
            trim_fraction=args.trim_fraction,
        )

        giant = diagnostic.run_giant_test(test_args)
        disk = diagnostic.run_disk_test(test_args)
        for result, test_name in ((giant, "Giant only"), (disk, "Static disk only")):
            records[test_name].append(
                {
                    "dt": dt,
                    "time": result.time,
                    "omega": result.varpi,
                    "eccentricity": result.eccentricity,
                    "measured_rate": result.measured_rate,
                    "analytical_rate": result.analytical_rate,
                    "relative_error": result.relative_error,
                }
            )
            print(
                f"{test_name}, dt={dt:g} yr: "
                f"measured={result.measured_rate:.8e}, "
                f"analytical={result.analytical_rate:.8e}, "
                f"relative_error={result.relative_error:+.3%}",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_time_series(records, args.output_dir)
    save_summary(records, args.output_dir)
    plot_time_series(records, args.output_dir)
    print(f"results={args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
