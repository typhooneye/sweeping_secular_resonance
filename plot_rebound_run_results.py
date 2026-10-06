#!/usr/bin/env python
"""Plot embryo orbital evolution from REBOUND archives with analytical SR overlay.

Recognizes two archive filename families:
  single_giant_aj_*_ej_*_tdep_*_nemb_*_aemb_*.bin   the aj/ej/tdep demo grid
  SURF_mJ_*_aemb_*_[multi-embryo fields]_fe_*_ej_*_tdep_*.bin
                                                       the surfing-search grid
                                                     (output/surfing_search/{nogap,gap}/*)
For SURF archives the giant is fixed at a=10 AU and mstar=1; giant mass and
eccentricity, embryo a, e-damping factor f_e, and tau_n come from the filename;
whether the giant opens an empty gap is read from the immediate parent
directory name ("gap" vs "nogap").
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rebound

from ssr_single_giant import DiskParams, GiantParams, gap_edges, resonance_time


ARCHIVE_RE = re.compile(
    r"single_giant_aj_(?P<a_j>[0-9.]+)_ej_(?P<e_j>[0-9.]+)_"
    r"tdep_(?P<tdep_value>[0-9.]+)(?P<tdep_unit>Myr|kyr|yr)_nemb_(?P<n_embryos>[0-9]+)_"
    r"aemb_(?P<a_embryo>[0-9.]+)\.bin$"
)

SURF_RE = re.compile(
    r"SURF_mJ_(?P<m_j>[0-9.]+)_aemb_(?P<a_embryo>[0-9.]+)_"
    r"(?:nemb_(?P<n_embryos>[0-9]+)_memb_(?P<m_embryo>[0-9.]+)_kmut_(?P<k_mutual>[0-9.]+)_)?"
    r"fe_(?P<f_e>[0-9.]+)_ej_(?P<e_j>[0-9.]+)_"
    r"tdep_(?P<tdep>[0-9.]+e[+-][0-9]+)\.bin$"
)
SURF_A_JUP = 10.0  # AU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="output/rebound_aj_ej_tdep")
    parser.add_argument("--output-dir", default="figure/rebound_aj_ej_tdep")
    parser.add_argument("--pattern", default="*.bin",
                        help="glob under --input-dir; unrecognized filenames are skipped")
    parser.add_argument("--m-giant-mjup", type=float, default=1.0)
    parser.add_argument("--mstar", type=float, default=1.0)
    parser.add_argument("--k-gap-dep", type=float, default=0.0)
    parser.add_argument("--disk-inner-cutoff", type=float, default=0.0)
    parser.add_argument("--sigma0-cgs", type=float, default=1700.0)
    parser.add_argument("--disk-k", type=float, default=1.5)
    parser.add_argument("--z-k", type=float, default=None,
                        help="Override the local-disk coefficient; default: exact z_k(k).")
    parser.add_argument("--n-sr-grid", type=int, default=1500)
    parser.add_argument("--max-snapshots", type=int, default=2500)
    parser.add_argument("--include-no-gap-sr", action="store_true")
    return parser.parse_args()


def parse_archive_metadata(path: Path) -> dict[str, float | int | str | bool] | None:
    match = ARCHIVE_RE.match(path.name)
    if match is not None:
        tdep_value = float(match.group("tdep_value"))
        tdep_unit = match.group("tdep_unit")
        if tdep_unit == "Myr":
            tdep = tdep_value * 1.0e6
        elif tdep_unit == "kyr":
            tdep = tdep_value * 1.0e3
        else:
            tdep = tdep_value
        return {
            "family": "single_giant",
            "a_j": float(match.group("a_j")),
            "e_j": float(match.group("e_j")),
            "tdep": tdep,
            "n_embryos": int(match.group("n_embryos")),
            "a_embryo": float(match.group("a_embryo")),
        }

    match = SURF_RE.match(path.name)
    if match is not None:
        gap = path.parent.name == "gap"
        return {
            "family": "surf",
            "a_j": SURF_A_JUP,
            "e_j": float(match.group("e_j")),
            "m_j": float(match.group("m_j")),
            "tdep": float(match.group("tdep")),
            "f_e": float(match.group("f_e")),
            "n_embryos": int(match.group("n_embryos") or 1),
            "a_embryo": float(match.group("a_embryo")),
            "gap": gap,
        }

    return None


def snapshot_indices(n_snapshots: int, max_snapshots: int) -> np.ndarray:
    if n_snapshots <= max_snapshots:
        return np.arange(n_snapshots)
    return np.unique(np.linspace(0, n_snapshots - 1, max_snapshots, dtype=int))


def read_embryo_tracks(path: Path, max_snapshots: int) -> tuple[np.ndarray, dict[int, dict[str, np.ndarray]]]:
    warnings.filterwarnings(
        "ignore",
        message="You have to reset function pointers after creating a reb_simulation struct with a binary file.",
        category=RuntimeWarning,
    )
    archive = rebound.Simulationarchive(str(path))
    indices = snapshot_indices(len(archive), max_snapshots)
    first = archive[int(indices[0])]
    embryo_indices = list(range(2, first.N))

    times = np.full(len(indices), np.nan)
    tracks = {
        idx: {
            "a": np.full(len(indices), np.nan),
            "q": np.full(len(indices), np.nan),
            "Q": np.full(len(indices), np.nan),
        }
        for idx in embryo_indices
    }

    for out_idx, archive_idx in enumerate(indices):
        sim = archive[int(archive_idx)]
        times[out_idx] = sim.t
        primary = sim.particles[0]
        for particle_idx in embryo_indices:
            if particle_idx >= sim.N:
                continue
            try:
                orbit = sim.particles[particle_idx].orbit(primary=primary)
            except Exception:
                continue
            if not np.isfinite(orbit.a) or orbit.a <= 0.0:
                continue
            tracks[particle_idx]["a"][out_idx] = orbit.a
            tracks[particle_idx]["q"][out_idx] = orbit.a * (1.0 - orbit.e)
            tracks[particle_idx]["Q"][out_idx] = orbit.a * (1.0 + orbit.e)

    return times, tracks


def analytical_sr_curve(
    a_j: float,
    e_j: float,
    tdep: float,
    args: argparse.Namespace,
    gap: bool = True,
    m_giant_mjup: float | None = None,
    k_gap_dep: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    giant = GiantParams(
        a=a_j, e=e_j,
        mass_mjup=m_giant_mjup if m_giant_mjup is not None else args.m_giant_mjup,
        mstar=args.mstar,
    )
    disk = DiskParams(
        sigma0_cgs=args.sigma0_cgs,
        k=args.disk_k,
        z_k=args.z_k,
        t_dep=tdep,
        disk_inner_cutoff=args.disk_inner_cutoff,
        k_gap_dep=k_gap_dep if k_gap_dep is not None else args.k_gap_dep,
    )
    r_in, _ = gap_edges(giant)
    a_min = max(args.disk_inner_cutoff * 1.01, 0.02)
    a_max = max(r_in * 0.999, a_min * 1.1)
    a_grid = np.linspace(a_min, a_max, args.n_sr_grid)
    if gap:
        t_sr = resonance_time(a_grid, giant, disk, gap=True, gap_depletion=disk.k_gap_dep)
    else:
        t_sr = resonance_time(a_grid, giant, disk, gap=False)
    mask = np.isfinite(t_sr) & (t_sr >= 0.0)
    return t_sr[mask], a_grid[mask]


def plot_archive(path: Path, input_dir: Path, output_dir: Path, args: argparse.Namespace) -> Path | None:
    metadata = parse_archive_metadata(path)
    if metadata is None:
        print(f"skip {path}: filename does not match grid pattern")
        return None

    times, tracks = read_embryo_tracks(path, args.max_snapshots)
    time_myr = times / 1.0e6
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(tracks), 1)))
    for color, (particle_idx, track) in zip(colors, tracks.items()):
        label = f"embryo {particle_idx - 1}"
        ax.plot(time_myr, track["a"], color=color, linewidth=1.8, label=rf"{label}: $a$")
        ax.plot(time_myr, track["q"], color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.plot(time_myr, track["Q"], color=color, linestyle="--", linewidth=1.0, alpha=0.8)

    family = metadata.get("family", "single_giant")
    m_giant_mjup = metadata.get("m_j")
    gap = bool(metadata["gap"]) if family == "surf" else True
    # the surfing-search grid uses an empty gap (k_gap_dep=0.0) when open
    k_gap_dep = 0.0 if family == "surf" else None

    t_sr, a_sr = analytical_sr_curve(
        float(metadata["a_j"]),
        float(metadata["e_j"]),
        float(metadata["tdep"]),
        args,
        gap=gap,
        m_giant_mjup=m_giant_mjup,
        k_gap_dep=k_gap_dep,
    )
    sr_label = "analytical SR" if gap else "analytical SR, no gap"
    if len(t_sr) > 0:
        ax.plot(t_sr / 1.0e6, a_sr, color="black", linewidth=2.0, label=sr_label)

    if args.include_no_gap_sr and gap:
        t_sr_ng, a_sr_ng = analytical_sr_curve(
            float(metadata["a_j"]),
            float(metadata["e_j"]),
            float(metadata["tdep"]),
            args,
            gap=False,
            m_giant_mjup=m_giant_mjup,
        )
        if len(t_sr_ng) > 0:
            ax.plot(t_sr_ng / 1.0e6, a_sr_ng, color="0.45", linestyle=":", linewidth=1.8, label="analytical SR, no gap")

    ax.set_xlabel("time [Myr]")
    ax.set_ylabel("semimajor axis / periapse / apoapse [AU]")
    if family == "surf":
        ax.set_title(
            rf"$m_J={float(metadata['m_j']):g}\,M_{{\rm Jup}}$, "
            rf"$e_J={float(metadata['e_j']):g}$, "
            rf"$a_{{emb}}={float(metadata['a_embryo']):g}$ AU, "
            rf"$f_e={float(metadata['f_e']):g}$, "
            rf"$\tau_n={float(metadata['tdep']) / 1.0e3:g}$ kyr, "
            f"{'empty gap' if gap else 'no gap'}"
        )
    else:
        ax.set_title(
            rf"$a_J={float(metadata['a_j']):g}$ AU, "
            rf"$e_J={float(metadata['e_j']):g}$, "
            rf"$T_{{dep}}={float(metadata['tdep']) / 1.0e6:g}$ Myr"
        )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    finite_values = []
    for track in tracks.values():
        finite_values.extend(track["q"][np.isfinite(track["q"])])
        finite_values.extend(track["Q"][np.isfinite(track["Q"])])
    if len(a_sr) > 0:
        finite_values.extend(a_sr)
    if finite_values:
        upper = np.nanpercentile(finite_values, 99.5)
        ax.set_ylim(bottom=0.0, top=max(1.2, upper * 1.15))
    ax.set_xlim(left=0.0, right=max(time_myr[np.isfinite(time_myr)]))

    # Mirror the archive's subfolder structure under output_dir so archives
    # with identical filenames in different subfolders do not collide.
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
    # rglob searches input_dir and all of its subfolders.
    archives = sorted(input_dir.rglob(args.pattern))
    if not archives:
        raise FileNotFoundError(f"no archives matching {args.pattern} under {input_dir}")

    print(f"plotting {len(archives)} archives")
    for archive in archives:
        output = plot_archive(archive, input_dir, output_dir, args)
        if output is not None:
            print(output)


if __name__ == "__main__":
    main()
