#!/usr/bin/env python3
"""Figure for the Hopf transient-truncation test.

Panel A: raw longest-H1 persistence vs theta for each transient truncation.
Panel B: the same data squared -- a Hopf gives amplitude ~ sqrt(theta_H - theta),
so pers^2 is linear below the Hopf and its zero locates the bifurcation.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runlog import open_run_log

HERE = Path(__file__).resolve().parent
# Categorical slots 1-4. Slots 3/4 sit below 3:1 on the light surface, so the
# relief rule applies: every series is direct-labelled, never colour alone.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
C_INK, C_INK2, C_GRID = "#0b0b0b", "#52514e", "#dedcd5"


def main(argv=None):
    p = argparse.ArgumentParser(description="Plot the Hopf transient test.")
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, default=HERE / "figures" / "hopf_transient.png")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--fit-from", type=float, default=4.60,
                   help="left edge of the pers^2 linear fit window")
    p.add_argument("--fit-to", type=float, default=4.70,
                   help="right edge of the fit window. Deliberately a round "
                        "number, NOT the Hopf angle: bounding the fit by the "
                        "answer would make the extrapolation circular.")
    args = p.parse_args(argv)

    log = open_run_log("plot_hopf_test", args)
    try:
        rows = [r for r in csv_reader(args.csv)]
        th = np.array([float(r["theta"]) for r in rows])
        fracs, series = [], {}
        for k in rows[0]:
            if k.endswith("_h1_max_pers"):
                tag = k[: -len("_h1_max_pers")]
                fracs.append(tag)
                series[tag] = np.array([float(r[k]) if r[k] else np.nan for r in rows])
        HOPF = 3 * np.pi / 2

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 8.4), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1],
                                                   "hspace": 0.13})
        fig.patch.set_facecolor("#fcfcfb")
        for a in (ax, ax2):
            a.set_facecolor("#fcfcfb")
            a.grid(True, color=C_GRID, lw=0.8, zorder=0)
            a.set_axisbelow(True)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                a.spines[side].set_color(C_GRID)
            a.tick_params(colors=C_INK2, labelsize=11)
            a.axvline(HOPF, color=C_INK2, lw=1.5, ls=":", zorder=1)

        for i, tag in enumerate(fracs):
            y = series[tag]
            lbl = "keep " + tag.replace("keep", "") .replace("0.", ".")
            ax.semilogy(th, y, color=COLORS[i % 4], lw=1.8, marker="o", ms=3.4,
                        mec="#fcfcfb", mew=0.6, zorder=3, label=lbl, alpha=0.9)
        ax.annotate("Hopf\n(analytic)\n4.71239", (HOPF, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(6, -34), fontsize=9,
                    color=C_INK2)
        ax.set_ylabel("longest H1 persistence (raw, log)", fontsize=11.5, color=C_INK)
        ax.set_title("Raw H1 persistence: smooth through the Hopf, no kink;\n"
                     "all four transient truncations coincide",
                     fontsize=12.5, color=C_INK, loc="left", pad=8)
        ax.legend(loc="lower left", frameon=True, facecolor="#fcfcfb",
                  edgecolor=C_GRID, fontsize=9.5, labelcolor=C_INK2,
                  framealpha=1.0, ncol=4, title="transient truncation",
                  title_fontsize=9.5)

        # Panel B: sqrt scaling
        for i, tag in enumerate(fracs):
            ax2.plot(th, series[tag] ** 2, color=COLORS[i % 4], lw=1.6,
                     marker="o", ms=3.2, mec="#fcfcfb", mew=0.6, zorder=3,
                     alpha=0.9)
        m = (th >= args.fit_from) & (th <= args.fit_to)
        zeros = []
        for i, tag in enumerate(fracs):
            A = np.polyfit(th[m], series[tag][m] ** 2, 1)
            zeros.append(-A[1] / A[0])
        z = float(np.mean(zeros))
        xs = np.linspace(args.fit_from, args.fit_to + 0.03, 50)
        A = np.polyfit(th[m], series[fracs[0]][m] ** 2, 1)
        ax2.plot(xs, np.polyval(A, xs), color=C_INK2, lw=1.6, ls="--", zorder=4)
        ax2.set_ylim(bottom=0)
        ax2.annotate(f"linear fit on [{args.fit_from}, {args.fit_to}]\n"
                     f"  (window not set by the Hopf)\n"
                     f"extrapolated zero: {z:.5f}\n"
                     f"analytic Hopf:     {HOPF:.5f}\n"
                     f"error: {abs(z-HOPF):.5f} rad",
                     (0.60, 0.52), xycoords="axes fraction", fontsize=9.5,
                     color=C_INK2, family="monospace")
        ax2.set_ylabel(r"(longest H1 persistence)$^2$", fontsize=11.5, color=C_INK)
        ax2.set_xlabel(r"parameter-path angle  $\theta$  (rad)", fontsize=12,
                       color=C_INK)
        ax2.set_title(r"Squared, it is linear; extrapolating its zero recovers "
                      r"the Hopf  (amplitude $\sim\sqrt{\theta_H-\theta}$)",
                      fontsize=12.5, color=C_INK, loc="left", pad=10)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.075)
        fig.savefig(args.out, dpi=args.dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Wrote {args.out}")
        print(f"pers^2 zero crossings: {[f'{v:.5f}' for v in zeros]}")
        log.note("figure", str(args.out))
        log.close("ok")
        return 0
    except Exception as exc:
        log.close("error", exc)
        raise
    finally:
        print(f"Run log: {log.path}")


def csv_reader(path):
    with path.open() as fh:
        yield from csv.DictReader(l for l in fh if not l.startswith("#"))


if __name__ == "__main__":
    raise SystemExit(main())
