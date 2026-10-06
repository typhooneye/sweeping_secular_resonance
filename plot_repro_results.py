#!/usr/bin/env python
"""Plot the HR5183 (hr5183_1pt1M_repro) and NLT05 (nlt05_repro) reproduction runs.

Unlike plot_rebound_run_results.py (which is tied to the single_giant_* demo
grid), this script recognizes both reproduction filename families and builds
the analytic secular-resonance overlay with the parameters each set was
actually run with:

  HR5183_1pt1M_aemb_*.bin   giant 3.23 Mjup, a=18 AU, e=0.84 around a
                            1.07 Msun star; disk sigma0=50 g/cm^2, k=1.01,
                            tdep=1e6 yr, EMPTY gap (k_gap_dep=0).
                            The gr0/gr1 parent directory names the GR setting.
  NLT05_memb_*_aemb_*_ejup_*_tdep_*.bin
                            Jupiter (1 Mjup, 5.203 AU, e_J from filename)
                            around 1 Msun; MMSN disk sigma0=1700 g/cm^2,
                            k=1.5, tdep from filename, NO gap.

Each figure has two panels: (top) embryo a, periapse q, apoapse Q vs time
with the analytic sweeping-SR curve t_SR(a) and the predicted crossing time
of the embryo's initial a; (bottom) embryo eccentricity vs time. For HR5183,
the gr0/gr1 parent directory selects whether GR is omitted from or
included in the analytical overlay.

Usage (from sweeping_secular_resonance/):
    python plot_repro_results.py                          # both sets
    python plot_repro_results.py --input-dir output/nlt05_repro
Figures mirror the archive tree under --output-dir (default figure/).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_rebound_run_results import read_embryo_tracks
from ssr_single_giant import DiskParams, GiantParams, gap_edges, resonance_time


HR5183_RE = re.compile(
    r"HR5183_1pt1M_aemb_(?P<a_emb>[0-9.]+?)(?P<tag>_noforce)?\.bin$"
)
NLT05_RE = re.compile(
    r"NLT05_memb_(?P<m_emb>[^_]+)_aemb_(?P<a_emb>[0-9.]+)_"
    r"ejup_(?P<e_jup>[^_]+)_tdep_(?P<tdep>[^_]+)\.bin$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="output",
                        help="searched recursively for HR5183_*.bin and NLT05_*.bin")
    parser.add_argument("--output-dir", default="figure")
    parser.add_argument("--max-snapshots", type=int, default=2500)
    parser.add_argument("--n-sr-grid", type=int, default=1500)
    return parser.parse_args()


def classify(path: Path) -> dict | None:
    """Return giant/disk/plot metadata for one archive, or None if unknown."""
    match = HR5183_RE.match(path.name)
    if match is not None:
        gr = path.parent.name  # gr0 / gr1
        # The analytic overlay uses the same GR setting as the run it is
        # plotted over, so the curve is a prediction of that run; the GR
        # effect is read by comparing gr0 and gr1 figures.
        include_gr = gr == "gr1"
        noforce = match.group("tag") is not None
        return {
            "family": "hr5183",
            "a_emb": float(match.group("a_emb")),
            "giant": GiantParams(a=18.0, e=0.84, mass_mjup=3.23, mstar=1.07),
            "disk": DiskParams(sigma0_cgs=50.0, k=1.01, z_k=None,
                               t_dep=1.0e6, disk_inner_cutoff=0.0,
                               k_gap_dep=0.0, include_gr=include_gr),
            "gap": True,
            # the noforce case has no disk, so no SR overlay applies
            "overlay": not noforce,
            "sr_label": f"analytic SR ({'with' if include_gr else 'no'} GR)",
            "title": (rf"HR 5183 repro, $a_{{emb}}={float(match.group('a_emb')):g}$ AU, "
                      f"{gr}" + (" (no forces)" if noforce else "")),
        }
    match = NLT05_RE.match(path.name)
    if match is not None:
        return {
            "family": "nlt05",
            "a_emb": float(match.group("a_emb")),
            "giant": GiantParams(a=5.203, e=float(match.group("e_jup")),
                                 mass_mjup=1.0, mstar=1.0),
            # include_gr=False: the NLT05 runs have no GR force.
            "disk": DiskParams(sigma0_cgs=1700.0, k=1.5, z_k=1.094,
                               t_dep=float(match.group("tdep")),
                               disk_inner_cutoff=0.0, k_gap_dep=0.0,
                               include_gr=False),
            "gap": False,
            "overlay": True,
            "sr_label": "analytic SR (no GR)",
            "title": (rf"NLT05 repro, $m={float(match.group('m_emb')):g}\,M_\oplus$, "
                      rf"$a_{{emb}}={float(match.group('a_emb')):g}$ AU, "
                      rf"$e_J={float(match.group('e_jup')):g}$, "
                      rf"$\tau_n={float(match.group('tdep')) / 1.0e6:g}$ Myr"),
        }
    return None


def analytic_sr_curve(info: dict, n_grid: int) -> tuple[np.ndarray, np.ndarray]:
    giant, disk = info["giant"], info["disk"]
    if info["gap"]:
        r_in, _ = gap_edges(giant)
        a_max = r_in * 0.999
    else:
        a_max = min(3.0, 0.6 * giant.a)
    a_grid = np.linspace(0.02, a_max, n_grid)
    if info["gap"]:
        t_sr = resonance_time(a_grid, giant, disk, gap=True, gap_depletion=0.0)
    else:
        t_sr = resonance_time(a_grid, giant, disk, gap=False)
    mask = np.isfinite(t_sr) & (t_sr >= 0.0)
    return t_sr[mask], a_grid[mask]


def crossing_time(info: dict) -> float:
    a = np.array([info["a_emb"]])
    if info["gap"]:
        t = resonance_time(a, info["giant"], info["disk"], gap=True, gap_depletion=0.0)
    else:
        t = resonance_time(a, info["giant"], info["disk"], gap=False)
    return float(t[0])


def plot_archive(path: Path, input_dir: Path, output_dir: Path,
                 args: argparse.Namespace) -> Path | None:
    info = classify(path)
    if info is None:
        print(f"skip {path}: unrecognized filename")
        return None

    times, tracks = read_embryo_tracks(path, args.max_snapshots)
    time_myr = times / 1.0e6

    fig, (ax_a, ax_e) = plt.subplots(
        2, 1, sharex=True, figsize=(8.5, 7.5),
        gridspec_kw={"height_ratios": [3, 2]},
    )

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(tracks), 1)))
    for color, (particle_idx, track) in zip(colors, tracks.items()):
        label = f"embryo {particle_idx - 1}"
        ax_a.plot(time_myr, track["a"], color=color, linewidth=1.8,
                  label=rf"{label}: $a$")
        ax_a.plot(time_myr, track["q"], color=color, linestyle="--",
                  linewidth=1.0, alpha=0.8)
        ax_a.plot(time_myr, track["Q"], color=color, linestyle="--",
                  linewidth=1.0, alpha=0.8)
        with np.errstate(invalid="ignore", divide="ignore"):
            ecc = (track["Q"] - track["q"]) / (track["Q"] + track["q"])
        ax_e.plot(time_myr, ecc, color=color, linewidth=1.5, label=label)

    a_sr = np.array([])
    t_sr_myr = np.array([])
    if info["overlay"]:
        t_sr, a_sr = analytic_sr_curve(info, args.n_sr_grid)
        if len(t_sr) > 0:
            t_sr_myr = t_sr / 1.0e6
            ax_a.plot(t_sr_myr, a_sr, color="black", linewidth=2.0,
                      label=info["sr_label"])
        t_cross = crossing_time(info)
        if np.isfinite(t_cross) and t_cross >= 0.0:
            t_cross_myr = t_cross / 1.0e6
            for ax in (ax_a, ax_e):
                ax.axvline(t_cross_myr, color="black", linestyle=":",
                           linewidth=1.2, alpha=0.8)
            ax_a.plot(t_cross_myr, info["a_emb"], marker="o", markersize=5,
                      color="black", zorder=5)
            ax_e.text(t_cross_myr, 0.95, "  predicted crossing",
                      transform=ax_e.get_xaxis_transform(),
                      fontsize=8, va="top", clip_on=True)

    ax_a.set_ylabel("$a$, $q$, $Q$ [AU]")
    ax_a.set_title(info["title"])
    ax_a.grid(True, alpha=0.3)
    ax_a.legend(loc="best", fontsize=8)

    ax_e.set_xlabel("time [Myr]")
    ax_e.set_ylabel("eccentricity")
    ax_e.set_ylim(bottom=0.0)
    ax_e.grid(True, alpha=0.3)

    finite_values = []
    for track in tracks.values():
        finite_values.extend(track["q"][np.isfinite(track["q"])])
        finite_values.extend(track["Q"][np.isfinite(track["Q"])])
    if len(a_sr) > 0:
        finite_values.extend(a_sr)
    if finite_values:
        upper = np.nanpercentile(finite_values, 99.5)
        ax_a.set_ylim(bottom=0.0, top=max(1.2, upper * 1.15))
    finite_t = time_myr[np.isfinite(time_myr)]
    time_right_candidates = []
    if len(finite_t) > 0:
        time_right_candidates.append(float(np.nanmax(finite_t)))
    if len(t_sr_myr) > 0:
        time_right_candidates.append(float(np.nanmax(t_sr_myr)))
    if time_right_candidates:
        ax_a.set_xlim(left=0.0, right=max(time_right_candidates))

    relative = path.relative_to(input_dir)
    output_path = (output_dir / relative).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    archives = sorted(
        list(input_dir.rglob("HR5183_*.bin")) + list(input_dir.rglob("NLT05_*.bin"))
    )
    if not archives:
        raise FileNotFoundError(
            f"no HR5183_*.bin or NLT05_*.bin archives under {input_dir}"
        )
    print(f"plotting {len(archives)} archives")
    for archive in archives:
        output = plot_archive(archive, input_dir, output_dir, args)
        if output is not None:
            print(output)


if __name__ == "__main__":
    main()
