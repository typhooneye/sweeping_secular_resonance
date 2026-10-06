#!/usr/bin/env python
"""Plot the precession-rate figure from existing diagnose_sr_stepwise.py output.

Reads whatever .bin SimulationArchives are already on disk under
--output-dir (as written by diagnose_sr_stepwise.py) and regenerates the
summary CSV and figure from them. Does not run or resubmit any simulation.

Tolerates an archive that is missing, empty, or still being actively written
by a running Slurm job: a test whose bin can't be read (or has fewer than 2
snapshots) is skipped with a warning rather than aborting the whole plot, so
you can re-run this at any time to see partial progress.

Usage:
    python plot_sr_stepwise_results.py --output-dir output/diagnose_sr_stepwise
Pass the same physical-parameter flags (--rocky-a, --giant-a, ...) used for
the run being plotted if they were not left at diagnose_sr_stepwise.py's
defaults, so the analytical comparison lines match.
"""

from __future__ import annotations

import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import rebound

REPO_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", Path(__file__).resolve().parent)).resolve()
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

STEPWISE = SourceFileLoader(
    "diagnose_sr_stepwise_for_plotting",
    str(REPO_DIR / "diagnose_sr_stepwise.py"),
).load_module()


def try_read_archive(bin_path: Path, track_index: int):
    """Read (time, pomega_raw, eccentricity) from a possibly-incomplete archive.

    Returns None if the file is missing, unreadable, or has under 2 usable
    snapshots. If the archive is being appended to concurrently, reads
    whatever complete snapshots are available and returns those.
    """
    if not bin_path.exists():
        print(f"  {bin_path.name}: not found yet, skipping")
        return None
    try:
        sa = rebound.Simulationarchive(str(bin_path))
    except Exception as exc:
        print(f"  {bin_path.name}: could not open archive ({exc}), skipping")
        return None

    times, varpi, ecc = [], [], []
    try:
        for sim_snap in sa:
            orbit = sim_snap.particles[track_index].orbit(primary=sim_snap.particles[0])
            times.append(sim_snap.t)
            varpi.append(orbit.pomega)
            ecc.append(orbit.e)
    except Exception as exc:
        print(f"  {bin_path.name}: archive read stopped after {len(times)} snapshots ({exc})")

    if len(times) < 2:
        print(f"  {bin_path.name}: only {len(times)} snapshot(s) available, skipping (job likely still running)")
        return None

    print(f"  {bin_path.name}: {len(times)} snapshots, t=[{times[0]:.3g}, {times[-1]:.3g}] yr")
    return np.array(times), np.array(varpi), np.array(ecc)


def analyze_available(config, output_dir: Path, trim_fraction: float):
    bin_path = output_dir / config.fname
    data = try_read_archive(bin_path, config.track_index)
    if data is None:
        return None
    time, varpi_raw, ecc = data
    varpi_unwrapped = np.unwrap(varpi_raw)
    measured = STEPWISE.fit_rate(time, varpi_unwrapped, trim_fraction)
    return STEPWISE.StepResult(
        name=config.name,
        time=time,
        varpi_raw=varpi_raw,
        varpi_unwrapped=varpi_unwrapped,
        eccentricity=ecc,
        measured_rate=measured,
        analytical_rate=config.analytical_rate_rad_per_yr,
        bin_path=bin_path,
    )


def main() -> None:
    args = STEPWISE.parse_args()
    output_dir = args.output_dir.resolve()
    configs = STEPWISE.build_configs(args)

    print(f"reading archives from {output_dir}")
    results = []
    for config in configs:
        result = analyze_available(config, output_dir, args.trim_fraction)
        if result is not None:
            results.append(result)

    if not results:
        print("no usable archives found -- nothing to plot")
        return

    STEPWISE.write_summary(results, output_dir / "summary.csv")
    STEPWISE.plot_results(results, args.figure)

    for result in results:
        print(
            f"{result.name}: measured={result.measured_rate:.6e} rad/yr, "
            f"analytical={result.analytical_rate:.6e} rad/yr, "
            f"relative_error={result.relative_error:+.3%}"
        )
    skipped = {c.name for c in configs} - {r.name for r in results}
    if skipped:
        print(f"skipped (no usable data yet): {sorted(skipped)}")
    print(f"summary={output_dir / 'summary.csv'}")
    print(f"figure={args.figure.resolve()}")


if __name__ == "__main__":
    main()
