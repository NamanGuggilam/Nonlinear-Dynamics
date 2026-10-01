#!/usr/bin/env python3
"""Sanity check for the TDA pipeline.

Samples a noisy unit circle, runs ripser, prints the H0/H1 barcodes, saves
the persistence diagram, and asserts that exactly one H1 bar has
persistence > 0.5.  Nothing else in this project should be trusted until
this passes.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from persim import plot_diagrams
from ripser import ripser

from runlog import open_run_log

HERE = Path(__file__).resolve().parent


def sample_noisy_circle(n_points, sigma, seed):
    """n_points uniformly-in-angle on the unit circle + isotropic Gaussian noise."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n_points)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    pts += rng.normal(loc=0.0, scale=sigma, size=pts.shape)
    return pts


def describe_barcode(dgm, label, max_print):
    """Print a barcode, longest-lived first. Returns finite persistences."""
    if dgm.size == 0:
        print(f"  {label}: (empty)")
        return np.empty(0)

    births, deaths = dgm[:, 0], dgm[:, 1]
    pers = deaths - births
    order = np.argsort(-np.where(np.isinf(pers), np.inf, pers))

    print(f"  {label}: {len(dgm)} bars")
    for rank, i in enumerate(order[:max_print]):
        d = deaths[i]
        d_str = "   inf" if np.isinf(d) else f"{d:6.4f}"
        p_str = "   inf" if np.isinf(pers[i]) else f"{pers[i]:6.4f}"
        print(f"    [{rank:>3}] birth={births[i]:6.4f}  death={d_str}  persistence={p_str}")
    if len(order) > max_print:
        print(f"    ... {len(order) - max_print} shorter bars not shown")

    return pers[np.isfinite(pers)]


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Sanity-check the ripser pipeline on a noisy unit circle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-points", type=int, default=300, help="points sampled on the circle")
    p.add_argument("--sigma", type=float, default=0.05, help="Gaussian noise std dev")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for the sample")
    p.add_argument("--persistence-threshold", type=float, default=0.5,
                   help="an H1 bar counts as significant above this persistence")
    p.add_argument("--expected-h1", type=int, default=1,
                   help="how many significant H1 bars must be present")
    p.add_argument("--maxdim", type=int, default=1, help="max homology dimension for ripser")
    p.add_argument("--max-print", type=int, default=10,
                   help="longest-lived bars to print per dimension")
    p.add_argument("--figure", type=Path, default=HERE / "figures" / "circle_check.png",
                   help="output path for the persistence diagram")
    p.add_argument("--dpi", type=int, default=160, help="figure resolution")
    args = p.parse_args(argv)

    log = open_run_log("verify_pipeline", args)
    try:
        print(f"Sampling {args.n_points} points on the unit circle "
              f"(sigma={args.sigma}, seed={args.seed})")
        pts = sample_noisy_circle(args.n_points, args.sigma, args.seed)

        print("Running ripser...")
        dgms = ripser(pts, maxdim=args.maxdim)["dgms"]

        print("\nBarcodes:")
        describe_barcode(dgms[0], "H0", args.max_print)
        h1_pers = describe_barcode(dgms[1], "H1", args.max_print)

        significant = h1_pers[h1_pers > args.persistence_threshold]
        print(f"\nH1 bars with persistence > {args.persistence_threshold}: "
              f"{len(significant)} (expected {args.expected_h1})")
        for v in np.sort(significant)[::-1]:
            print(f"    persistence = {v:.4f}")

        args.figure.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        axes[0].scatter(pts[:, 0], pts[:, 1], s=8, alpha=0.75)
        axes[0].set_aspect("equal")
        axes[0].set_title(f"Noisy unit circle (n={args.n_points}, sigma={args.sigma})")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        plot_diagrams(dgms, ax=axes[1])
        axes[1].set_title("Persistence diagram")
        fig.tight_layout()
        fig.savefig(args.figure, dpi=args.dpi)
        plt.close(fig)
        print(f"\nSaved persistence diagram to {args.figure}")

        log.note("n_h1_significant", int(len(significant)))
        log.note("h1_persistences", [float(v) for v in np.sort(significant)[::-1]])
        log.note("figure", str(args.figure))

        assert len(significant) == args.expected_h1, (
            f"expected exactly {args.expected_h1} H1 bar(s) with persistence "
            f"> {args.persistence_threshold}, found {len(significant)}: "
            f"{np.sort(significant)[::-1]}"
        )

        print("\nPASS: pipeline sanity check succeeded.")
        log.close("ok")
        return 0

    except AssertionError as exc:
        print(f"\nFAIL: {exc}", file=sys.stderr)
        log.close("assertion_failed", exc)
        return 1
    except Exception as exc:
        log.close("error", exc)
        raise
    finally:
        print(f"Run log: {log.path}")


if __name__ == "__main__":
    raise SystemExit(main())
