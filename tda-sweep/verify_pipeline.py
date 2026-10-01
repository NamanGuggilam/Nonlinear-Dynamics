"""Sanity check that the ripser/persim stack works.

Samples a noisy unit circle, runs ripser, prints the H0/H1 barcodes, saves the
persistence diagram to figures/circle_check.png, and asserts that exactly one
H1 bar has persistence > 0.5. This must pass before anything else runs.
"""

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import persim
from ripser import ripser

from runlog import start_run_log

FIG_DIR = pathlib.Path(__file__).resolve().parent / "figures"


def sample_noisy_circle(n, sigma, seed):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points + rng.normal(scale=sigma, size=points.shape)


def print_barcode(name, diagram):
    print(f"{name} barcode ({len(diagram)} bars):")
    for birth, death in diagram:
        pers = death - birth
        print(f"  [{birth:.4f}, {death:.4f})  persistence = {pers:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=300)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-persistence", type=float, default=0.5,
                        help="required persistence of the single H1 bar")
    args = parser.parse_args()
    logger = start_run_log("verify_pipeline", vars(args))

    points = sample_noisy_circle(args.n_points, args.sigma, args.seed)
    diagrams = ripser(points)["dgms"]
    h0, h1 = diagrams[0], diagrams[1]

    print_barcode("H0", h0)
    print_barcode("H1", h1)

    FIG_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    persim.plot_diagrams(diagrams, ax=ax)
    ax.set_title(f"Noisy circle (n={args.n_points}, sigma={args.sigma})")
    fig.tight_layout()
    out_path = FIG_DIR / "circle_check.png"
    fig.savefig(out_path, dpi=200)
    logger.info("saved persistence diagram to %s", out_path)

    persistences = h1[:, 1] - h1[:, 0]
    n_big = int(np.sum(persistences > args.min_persistence))
    logger.info("H1 bars with persistence > %.2f: %d", args.min_persistence, n_big)
    assert n_big == 1, (
        f"expected exactly one H1 bar with persistence > {args.min_persistence}, "
        f"got {n_big}"
    )
    print("PASS: exactly one H1 bar with persistence "
          f"> {args.min_persistence}; ripser stack is working.")


if __name__ == "__main__":
    main()
