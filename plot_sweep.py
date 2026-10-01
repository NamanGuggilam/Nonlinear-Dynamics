#!/usr/bin/env python3
"""Plot a sweep CSV: b0 and b1 against theta, with UNDEFINED sectors shaded.

Both series are integer counts on the same scale, so they share one axis --
no second y-scale.  Sectors with no bounded attractor are drawn as shaded
bands with a break in the lines, never as zeros.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from runlog import open_run_log

HERE = Path(__file__).resolve().parent
UNDEFINED = "UNDEFINED"

# Categorical slots 1 and 2 of the reference palette; validated as a pair
# (CVD dE 24.7, normal-vision dE 33.6, both >= 3:1 on the light surface).
C_B0 = "#2a78d6"      # blue
C_B1 = "#eb6834"      # orange
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_GRID = "#dedcd5"
C_UNDEF = "#b4b1a6"
C_PERS = "#1baf7a"    # categorical slot 3; sits below 3:1 on the light surface,
                      # so the relief rule applies -- its panel carries a named
                      # y-axis label and a direct annotation, never colour alone.


def read_sweep(path):
    """Return (meta dict from '#' header lines, list of row dicts)."""
    meta, data_lines = {}, []
    with path.open() as fh:
        for line in fh:
            if line.startswith("#"):
                body = line[1:].strip()
                if ": " in body:
                    k, v = body.split(": ", 1)
                    meta[k.strip()] = v.strip()
            else:
                data_lines.append(line)
    rows = list(csv.DictReader(data_lines))
    return meta, rows


def to_series(rows):
    """theta, b0, b1 (NaN where UNDEFINED), and the undefined boolean mask."""
    theta = np.array([float(r["theta"]) for r in rows])
    undef = np.array([r["status"] == UNDEFINED for r in rows])

    def col(name):
        out = np.full(len(rows), np.nan)
        for i, r in enumerate(rows):
            if r["status"] != UNDEFINED and r[name] not in ("", UNDEFINED):
                out[i] = float(r[name])
        return out

    return theta, col("b0"), col("b1_betti"), undef


def undefined_spans(theta, undef, period=2 * np.pi):
    """Contiguous runs of undefined angles -> (start, end) shading bands.

    Bands extend to the midpoint between neighbouring samples so the shading
    meets the defined region instead of stopping at the sample point.
    """
    if len(theta) == 0:
        return []
    step = (theta[1] - theta[0]) if len(theta) > 1 else period
    spans, start = [], None
    for i, flag in enumerate(undef):
        if flag and start is None:
            start = theta[i] - step / 2.0
        elif not flag and start is not None:
            spans.append((start, theta[i] - step / 2.0))
            start = None
    if start is not None:
        spans.append((start, theta[-1] + step / 2.0))
    return spans


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Plot Betti numbers vs theta from a sweep CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csv", type=Path, help="sweep CSV produced by sweep.py")
    p.add_argument("--out", type=Path, default=None,
                   help="output image (default: figures/<csv stem>.png)")
    p.add_argument("--dpi", type=int, default=200, help="figure resolution")
    p.add_argument("--figsize", type=float, nargs=2, default=[9.0, 5.0],
                   metavar=("W", "H"), help="figure size in inches")
    p.add_argument("--title", default=None, help="override the plot title")
    p.add_argument("--with-persistence", action="store_true",
                   help="add a lower panel showing the raw persistence of the "
                        "longest H1 bar vs theta (needs the h1_max_pers column)")
    args = p.parse_args(argv)

    if args.out is None:
        args.out = HERE / "figures" / f"{args.csv.stem}.png"

    log = open_run_log("plot_sweep", args)
    try:
        meta, rows = read_sweep(args.csv)
        if not rows:
            raise SystemExit(f"no data rows in {args.csv}")
        theta, b0, b1, undef = to_series(rows)
        spans = undefined_spans(theta, undef)

        # Raw longest-H1 persistence, recorded independently of any threshold.
        h1p = np.full(len(rows), np.nan)
        for i, r in enumerate(rows):
            v = r.get("h1_max_pers", "")
            if r["status"] != UNDEFINED and v not in ("", UNDEFINED):
                h1p[i] = float(v)
        floor = np.full(len(rows), np.nan)
        for i, r in enumerate(rows):
            v = r.get("pers_floor", "")
            if v not in ("", UNDEFINED):
                floor[i] = float(v)
        want_pers = args.with_persistence and np.any(np.isfinite(h1p))

        args.out.parent.mkdir(parents=True, exist_ok=True)
        if want_pers:
            # A Betti count and a persistence are different units, so they get
            # separate panels sharing x -- never two y-scales on one axis.
            w, h = args.figsize
            fig, (ax, axp) = plt.subplots(
                2, 1, figsize=(w, h * 1.55), sharex=True,
                gridspec_kw={"height_ratios": [1.0, 0.72], "hspace": 0.12})
            axp.set_facecolor("#fcfcfb")
        else:
            fig, ax = plt.subplots(figsize=tuple(args.figsize))
            axp = None
        fig.patch.set_facecolor("#fcfcfb")
        ax.set_facecolor("#fcfcfb")

        for target in ([ax] if axp is None else [ax, axp]):
            for lo, hi in spans:
                target.axvspan(lo, hi, color=C_UNDEF, alpha=0.28, lw=0, zorder=0)
                target.axvspan(lo, hi, facecolor="none", edgecolor=C_UNDEF,
                               hatch="///", lw=0, alpha=0.5, zorder=0)

        # NaN at undefined angles breaks the line rather than drawing zero.
        ax.plot(theta, b0, color=C_B0, lw=2.0, marker="o", ms=5.5,
                mec="#fcfcfb", mew=1.2, zorder=3, label=r"$b_0$  (components)")
        ax.plot(theta, b1, color=C_B1, lw=2.0, marker="s", ms=5.5,
                mec="#fcfcfb", mew=1.2, zorder=3, label=r"$b_1$  (loops)")

        xaxis_owner = ax if axp is None else axp
        xaxis_owner.set_xlabel(r"parameter-path angle  $\theta$  (rad)",
                               fontsize=12, color=C_INK)
        ax.set_ylabel("Betti number", fontsize=12, color=C_INK)
        # Full-circle sweeps get pi-fraction ticks; a sub-interval sweep gets
        # plain radians over its own range, so a narrow window stays readable.
        span = float(theta[-1] - theta[0]) if len(theta) > 1 else 0.0
        pad = max(0.05 * span, 0.02) if span else 0.05
        if span > 1.9 * np.pi:
            ax.set_xlim(-0.05, 2 * np.pi + 0.05)
            ax.set_xticks(np.arange(0, 2.01, 0.25) * np.pi)
            ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$",
                                r"$5\pi/4$", r"$3\pi/2$", r"$7\pi/4$", r"$2\pi$"])
        else:
            ax.set_xlim(theta[0] - pad, theta[-1] + pad)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8))
        finite = np.concatenate([b0[np.isfinite(b0)], b1[np.isfinite(b1)]])
        top = max(2.0, float(finite.max()) if finite.size else 2.0)
        ax.set_ylim(-0.35, top + 0.9)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.grid(True, color=C_GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C_GRID)
        ax.tick_params(colors=C_INK2, labelsize=11)

        if axp is not None:
            axp.plot(theta, h1p, color=C_PERS, lw=2.0, marker="o", ms=4.5,
                     mec="#fcfcfb", mew=1.0, zorder=3)
            if np.any(np.isfinite(floor)) and np.nanmax(floor) > 0:
                axp.plot(theta, floor, color=C_INK2, lw=1.6, ls="--", zorder=2)
                j = int(np.nanargmax(np.where(np.isfinite(floor), floor, -np.inf)))
                axp.annotate("persistence floor", (theta[j], floor[j]),
                             textcoords="offset points", xytext=(6, -14),
                             fontsize=9, color=C_INK2)
            k = int(np.nanargmax(np.where(np.isfinite(h1p), h1p, -np.inf)))
            axp.annotate("longest H1 bar", (theta[k], h1p[k]),
                         textcoords="offset points", xytext=(6, 4),
                         fontsize=9, color=C_INK2)
            axp.set_ylabel("longest H1\npersistence (raw)", fontsize=11, color=C_INK)
            axp.grid(True, color=C_GRID, lw=0.8, zorder=0)
            axp.set_axisbelow(True)
            axp.set_ylim(bottom=0)
            for side in ("top", "right"):
                axp.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                axp.spines[side].set_color(C_GRID)
            axp.tick_params(colors=C_INK2, labelsize=11)

        system = meta.get("system", "?")
        eps = meta.get("eps", "?")
        rule = meta.get("betti_rule", "?")
        title = args.title or (
            f"Betti numbers along the anisotropic path — {system} normal form"
        )
        ax.set_title(title, fontsize=14, color=C_INK, pad=34, loc="left")
        n_undef = int(undef.sum())
        ax.text(0.0, 1.015,
                fr"$\varepsilon={eps}$,  $b_1=\varepsilon^4\cos\theta$, "
                fr"$b_2=\varepsilon^2\sin\theta$;  rule: {rule};  "
                f"{n_undef}/{len(rows)} sectors undefined",
                transform=ax.transAxes, fontsize=10, color=C_INK2, va="bottom")

        handles = [
            Line2D([], [], color=C_B0, lw=2.0, marker="o", ms=5.5,
                   label=r"$b_0$  (connected components)"),
            Line2D([], [], color=C_B1, lw=2.0, marker="s", ms=5.5,
                   label=r"$b_1$  (independent loops)"),
        ]
        if spans:
            handles.append(Patch(facecolor=C_UNDEF, alpha=0.35, hatch="///",
                                 edgecolor=C_UNDEF,
                                 label="UNDEFINED (no bounded attractor)"))
        ax.legend(handles=handles, loc="upper left", frameon=True,
                  facecolor="#fcfcfb", edgecolor=C_GRID, fontsize=10,
                  labelcolor=C_INK2, framealpha=1.0).set_zorder(5)

        if axp is None:
            fig.tight_layout()
        else:
            # tight_layout cannot handle the shared-x gridspec; place manually.
            fig.subplots_adjust(left=0.11, right=0.985, top=0.86, bottom=0.09,
                                hspace=0.12)
        fig.savefig(args.out, dpi=args.dpi, facecolor=fig.get_facecolor())
        plt.close(fig)

        print(f"{len(rows)} angles, {n_undef} UNDEFINED, "
              f"{len(spans)} shaded sector(s)")
        print(f"Wrote {args.out}")
        log.note("figure", str(args.out))
        log.note("n_undefined", n_undef)
        log.close("ok")
        return 0
    except Exception as exc:
        log.close("error", exc)
        raise
    finally:
        print(f"Run log: {log.path}")


if __name__ == "__main__":
    raise SystemExit(main())
