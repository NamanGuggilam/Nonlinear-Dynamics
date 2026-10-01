"""Plot b0 and b1 vs theta from a sweep.py CSV.

Undefined sectors (no bounded attractor) are shaded gray with hatching so they
read in print and for colorblind readers; Betti values there are absent, not
zero. Saves to figures/ at print resolution.
"""

import argparse
import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from runlog import start_run_log

FIG_DIR = pathlib.Path(__file__).resolve().parent / "figures"

# Colorblind-distinguishable pair; identity is also carried by linestyle+marker.
COLOR_B0 = "#2563eb"  # blue
COLOR_B1 = "#d97706"  # orange


def read_sweep_csv(path):
    """Return (header_params, thetas, b0, b1, defined) from a sweep.py CSV.

    b0/b1 are float arrays with NaN where undefined.
    """
    params = {}
    rows = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = None
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                text = row[0].lstrip("# ")
                if "=" in text:
                    key, _, value = text.partition("=")
                    params[key.strip()] = value.strip()
                continue
            if header is None:
                header = row
                continue
            rows.append(row)
    thetas = np.array([float(r[0]) for r in rows])
    defined = np.array([r[3] == "1" for r in rows])
    b0 = np.array([float(r[1]) if r[1] else np.nan for r in rows])
    b1 = np.array([float(r[2]) if r[2] else np.nan for r in rows])
    return params, thetas, b0, b1, defined


def undefined_sectors(thetas, defined):
    """Merge contiguous undefined samples into (start, end) angular sectors.

    Each sample owns a sector of half the local spacing on each side.
    """
    if len(thetas) < 2:
        dtheta = 2.0 * np.pi
    else:
        dtheta = thetas[1] - thetas[0]
    sectors = []
    start = None
    for theta, ok in zip(thetas, defined):
        if not ok and start is None:
            start = theta - dtheta / 2.0
        elif ok and start is not None:
            sectors.append((start, theta - dtheta / 2.0))
            start = None
    if start is not None:
        sectors.append((start, thetas[-1] + dtheta / 2.0))
    return sectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=pathlib.Path, help="CSV produced by sweep.py")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="output PNG (default figures/<csv stem>_betti.png)")
    args = parser.parse_args()
    logger = start_run_log("plot_sweep", vars(args))

    params, thetas, b0, b1, defined = read_sweep_csv(args.csv)
    sectors = undefined_sectors(thetas, defined)
    logger.info("read %d thetas from %s (%d undefined, %d sectors)",
                len(thetas), args.csv, int(np.sum(~defined)), len(sectors))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lo, hi in sectors:
        ax.axvspan(lo, hi, facecolor="0.85", edgecolor="0.6",
                   hatch="///", linewidth=0, zorder=0)
    ax.plot(thetas, b0, color=COLOR_B0, linestyle="-", marker="o",
            markersize=4, linewidth=2, drawstyle="steps-mid",
            label=r"$b_0$", zorder=3)
    ax.plot(thetas, b1, color=COLOR_B1, linestyle="--", marker="s",
            markersize=4, linewidth=2, drawstyle="steps-mid",
            label=r"$b_1$", zorder=3)

    ax.set_xlim(0.0, 2.0 * np.pi)
    ax.set_xticks(np.arange(0, 2.01 * np.pi, np.pi / 2))
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    finite = np.concatenate([b0[defined], b1[defined]]) if defined.any() else np.array([0.0])
    ymax = int(np.nanmax(finite)) if finite.size else 1
    ax.set_yticks(np.arange(0, ymax + 2))
    ax.set_ylim(-0.5, ymax + 1)
    ax.set_xlabel(r"$\theta$ (parameter-circle angle, rad)")
    ax.set_ylabel("Betti number")
    system = params.get("system", "?")
    eps = params.get("eps", "?")
    ax.set_title(f"Betti numbers along parameter circle — {system}, eps={eps}")
    ax.grid(True, color="0.9", linewidth=0.8, zorder=1)
    handles, labels = ax.get_legend_handles_labels()
    if sectors:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="0.85",
                                     edgecolor="0.6", hatch="///"))
        labels.append("undefined (divergent)")
    ax.legend(handles, labels, loc="upper right", framealpha=0.95)
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    out = args.out or FIG_DIR / f"{args.csv.stem}_betti.png"
    fig.savefig(out, dpi=200)
    logger.info("saved plot to %s", out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
