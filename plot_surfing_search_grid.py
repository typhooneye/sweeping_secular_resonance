#!/usr/bin/env python
"""Multi-panel diagnostic grid for the surfing-search archives.

One figure per (tau_n, gap, e_J) combination.  Within a figure: rows =
innermost a_emb, columns = f_e.  Each cell is a two-row sub-panel -- top shows a(t)
with its periapse/apoapse envelope a(1-e)/a(1+e) and the test-particle
analytical SR-crossing curve; bottom shows e(t).  Giant mass is encoded by
color and embryo identity by marker, with all four embryos in the same cell.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_rebound_run_results import (
    analytical_sr_curve,
    read_embryo_tracks,
)

A_JUP = 10.0  # AU, matches submit_rebound_surfing_search.py
N_EMB = 4
M_EMB = 1.0
MUTUAL_HILL_SPACING = 6.0
MJ_LIST = (1.0, 10.0)
A_EMB_LIST = (1.0, 1.5, 2.0)
FE_LIST = (1.0, 0.5, 0.0)
E_JUP_LIST = (0.1, 0.3, 0.6)
TDEP_LIST = (3.0e5, 1.0e6, 3.0e6)
EMBRYO_MARKERS = ("o", "s", "^", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="output/surfing_search_4emb_m1_kmut6")
    parser.add_argument("--output-dir", default="figure/surfing_search_grid_4emb_m1_kmut6")
    parser.add_argument("--max-snapshots", type=int, default=2500)
    parser.add_argument("--n-sr-grid", type=int, default=1500)
    return parser.parse_args()


def archive_path(input_dir: Path, gap: bool, m_j: float, a_emb: float, f_e: float, e_jup: float, tdep: float) -> Path:
    gap_tag = "gap" if gap else "nogap"
    fname = (
        f"SURF_mJ_{m_j:g}_aemb_{a_emb:.2f}_nemb_{N_EMB}_memb_{M_EMB:g}_"
        f"kmut_{MUTUAL_HILL_SPACING:g}_fe_{f_e:g}_ej_{e_jup:g}_tdep_{tdep:.0e}.bin"
    )
    return input_dir / gap_tag / fname


def analytic_args(n_sr_grid: int) -> SimpleNamespace:
    # Matches BASE_FORCE_PARAMS in submit_rebound_surfing_search.py.
    return SimpleNamespace(
        mstar=1.0,
        sigma0_cgs=1700.0,
        disk_k=1.5,
        z_k=None,
        disk_inner_cutoff=0.0,
        k_gap_dep=0.0,
        n_sr_grid=n_sr_grid,
    )


def plot_figure(input_dir: Path, output_dir: Path, gap: bool, tdep: float, e_jup: float, args: argparse.Namespace) -> tuple[Path, bool]:
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(MJ_LIST)))
    a_args = analytic_args(args.n_sr_grid)

    sr_curves = {}
    for m_j in MJ_LIST:
        t_sr, a_sr = analytical_sr_curve(
            A_JUP, e_jup, tdep, a_args, gap=gap, m_giant_mjup=m_j, k_gap_dep=0.0,
        )
        sr_curves[m_j] = (t_sr / 1.0e6, a_sr)

    n_rows, n_cols = len(A_EMB_LIST), len(FE_LIST)
    fig = plt.figure(figsize=(4.2 * n_cols, 3.9 * n_rows))
    outer_gs = fig.add_gridspec(n_rows, n_cols, hspace=0.55, wspace=0.3)

    handles = [
        plt.Line2D([0], [0], color=c, linewidth=2.0, label=rf"$m_J={m_j:g}\,M_{{\rm Jup}}$")
        for m_j, c in zip(MJ_LIST, colors)
    ]
    embryo_handles = [
        plt.Line2D([0], [0], color="0.3", marker=marker, linestyle="none", markersize=4, label=f"embryo {i + 1}")
        for i, marker in enumerate(EMBRYO_MARKERS)
    ]
    style_handles = [
        plt.Line2D([0], [0], color="0.3", linewidth=1.6, label=r"$a(t)$"),
        plt.Line2D([0], [0], color="0.3", linewidth=1.0, linestyle="--", label=r"$a(1\pm e)$"),
        plt.Line2D([0], [0], color="0.3", linewidth=1.3, linestyle=":", label="test-particle analytic SR"),
    ]

    any_data = False
    for i, a_emb in enumerate(A_EMB_LIST):
        for j, f_e in enumerate(FE_LIST):
            inner_gs = outer_gs[i, j].subgridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.08)
            ax_a = fig.add_subplot(inner_gs[0])
            ax_e = fig.add_subplot(inner_gs[1], sharex=ax_a)
            cell_tracks = []

            for m_j, color in zip(MJ_LIST, colors):
                path = archive_path(input_dir, gap, m_j, a_emb, f_e, e_jup, tdep)
                if not path.exists():
                    continue
                times, tracks = read_embryo_tracks(path, args.max_snapshots)
                if not tracks:
                    continue
                any_data = True
                t_myr = times / 1.0e6

                for embryo_index, track in enumerate(tracks.values()):
                    cell_tracks.append(track)
                    alpha = 1.0 - 0.15 * embryo_index
                    marker = EMBRYO_MARKERS[embryo_index % len(EMBRYO_MARKERS)]
                    markevery = max(len(t_myr) // 18, 1)
                    ax_a.plot(t_myr, track["a"], color=color, linewidth=1.2, alpha=alpha,)
                              #amarker=marker, markevery=markevery, markersize=2.5)
                    ax_a.plot(t_myr, track["q"], color=color, linewidth=0.7, linestyle="--", alpha=0.45 * alpha)
                    ax_a.plot(t_myr, track["Q"], color=color, linewidth=0.7, linestyle="--", alpha=0.45 * alpha)

                    with np.errstate(invalid="ignore", divide="ignore"):
                        ecc = (track["Q"] - track["q"]) / (track["Q"] + track["q"])
                    ax_e.plot(t_myr, ecc, color=color, linewidth=1.0, alpha=alpha),
                              #marker=marker, markevery=markevery, markersize=2.5)

                t_sr, a_sr = sr_curves[m_j]
                if len(t_sr):
                    ax_a.plot(t_sr, a_sr, color=color, linewidth=1.3, linestyle=":")
            if cell_tracks:
                ax_a.set_ylim(
                    min(np.nanmin(track["q"]) for track in cell_tracks) - 0.3,
                    max(np.nanmax(track["Q"]) for track in cell_tracks) + 0.3,
                )
            ax_a.set_title(rf"$a_{{inner}}={a_emb:g}$ AU, $f_e={f_e:g}$", fontsize=9)
            ax_a.grid(True, alpha=0.25)
            ax_e.grid(True, alpha=0.25)
            ax_e.set_ylim(bottom=0.0)
            plt.setp(ax_a.get_xticklabels(), visible=False)
            if i == n_rows - 1:
                ax_e.set_xlabel("time [Myr]", fontsize=8)
            if j == 0:
                ax_a.set_ylabel("a, q, Q [AU]", fontsize=8)
                ax_e.set_ylabel("e", fontsize=8)
            if i == 0 and j == 0:
                ax_a.legend(handles=handles + embryo_handles + style_handles,
                            loc="upper right", fontsize=5.5, framealpha=0.85)

    gap_label = "empty gap" if gap else "no gap"
    fig.suptitle(
        rf"Surfing search: $a_J={A_JUP:g}$ AU, $e_J={e_jup:g}$, "
        rf"$\tau_n={tdep:.0e}$ yr, $4\times1M_\oplus$ embryos at $6R_{{H,m}}$, {gap_label}",
        fontsize=13,
    )

    gap_tag = "gap" if gap else "nogap"
    out_path = output_dir / f"surfing_grid_ej_{e_jup:g}_tdep_{tdep:.0e}_{gap_tag}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path, any_data


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    for e_jup in E_JUP_LIST:
        for tdep in TDEP_LIST:
            for gap in (False, True):
                out_path, any_data = plot_figure(input_dir, output_dir, gap, tdep, e_jup, args)
                if any_data:
                    print(out_path)
                else:
                    print(f"skip {out_path.name}: no archives found")


if __name__ == "__main__":
    main()
