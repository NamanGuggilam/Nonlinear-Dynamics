#!/usr/bin/env python3
"""Diagnose the Betti thresholding rules at a single theta.

Prints the H0 death-time distribution -- the quantity the median rule scales --
so the shape driving b0 is visible rather than inferred.  Optionally saves a
histogram.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ripser import ripser
from scipy.spatial import cKDTree

import sweep
from normal_forms import BLOWUP_WEIGHTS, RHS, blowup, param_path
from runlog import open_run_log

HERE = Path(__file__).resolve().parent


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Print the death-time distribution at one theta.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--system", choices=sorted(RHS), default="bt")
    p.add_argument("--theta", type=float, required=True, help="angle to inspect (rad)")
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--blowup-weights", default="auto", help="'auto' or 'p,q'")
    p.add_argument("--box", type=float, nargs=4, default=[-1.5, 2.0, -1.5, 1.5])
    p.add_argument("--box-coords", choices=["blown", "raw"], default="blown")
    p.add_argument("--nx", type=int, default=12)
    p.add_argument("--ny", type=int, default=12)
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=300.0)
    p.add_argument("--n-samples", type=int, default=400)
    p.add_argument("--keep-frac", type=float, default=0.3)
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--escape-radius", type=float, default=50.0)
    p.add_argument("--max-cloud", type=int, default=1200)
    p.add_argument("--subsample-seed", type=int, default=1234)
    p.add_argument("--backward-time", action="store_true")
    p.add_argument("--eps-scale", type=float, default=2.0)
    p.add_argument("--gap-max-candidates", type=int, default=12)
    p.add_argument("--percentiles", type=float, nargs="*",
                   default=[0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
    p.add_argument("--hist", type=Path, default=None, help="optional histogram path")
    args = p.parse_args(argv)

    weights = (BLOWUP_WEIGHTS[args.system] if args.blowup_weights == "auto"
               else tuple(float(v) for v in str(args.blowup_weights).split(",")))

    log = open_run_log("diagnose_threshold", args)
    try:
        b1, b2 = param_path(args.theta, args.eps)
        ics = sweep.initial_conditions(args.box, args.nx, args.ny,
                                       coords=args.box_coords, eps=args.eps)
        cloud, n_bounded, _, _ = sweep.integrate_grid(
            RHS[args.system], b1, b2, ics, (args.t0, args.t1), args.n_samples,
            args.keep_frac, args.rtol, args.atol, args.escape_radius)
        if args.backward_time:
            span = args.t1 - args.t0
            cb, _, _, _ = sweep.integrate_grid(
                RHS[args.system], b1, b2, ics, (args.t0, args.t0 - span),
                args.n_samples, args.keep_frac, args.rtol, args.atol,
                args.escape_radius)
            if len(cb):
                cloud = np.vstack([cloud, cb]) if len(cloud) else cb

        cloud = sweep.subsample(cloud, args.max_cloud, args.subsample_seed)
        uv = blowup(cloud, args.eps, weights=weights)
        print(f"system={args.system} theta={args.theta} b1={b1:.6g} b2={b2:.6g}")
        print(f"blowup weights={weights}  bounded={n_bounded}/{len(ics)}  "
              f"cloud={len(uv)} points")
        ext = uv.max(axis=0) - uv.min(axis=0)
        print(f"cloud extent (u,v) = ({ext[0]:.4g}, {ext[1]:.4g})  "
              f"diameter = {np.linalg.norm(ext):.4g}")

        dgms = ripser(uv, maxdim=1)["dgms"]
        h0 = dgms[0][np.isfinite(dgms[0][:, 1])]
        deaths = np.sort(h0[:, 1])          # H0 births are all 0
        print(f"\nH0: {len(dgms[0])} bars ({len(deaths)} finite + 1 infinite)")
        print("H0 death-time percentiles (= persistence, births are 0):")
        for q in args.percentiles:
            print(f"  p{q:>5.1f}  {np.percentile(deaths, q):.6g}")

        med = float(np.median(deaths))
        thr_med = args.eps_scale * med
        print(f"\n  median death        = {med:.6g}")
        print(f"  median rule thresh  = {args.eps_scale} * median = {thr_med:.6g}")
        print(f"  bars above it       = {int((deaths > thr_med).sum())}"
              f"  -> b0 = {int((deaths > thr_med).sum()) + 1}")

        # Why: consecutive samples of one trajectory are far closer together
        # than distinct trajectories are to each other.
        tree = cKDTree(uv)
        nn = tree.query(uv, k=2)[0][:, 1]
        print(f"\n  nearest-neighbour distance: median {np.median(nn):.6g}, "
              f"max {nn.max():.6g}")
        print(f"  ratio max(death)/median(death) = {deaths.max() / med:.4g}")

        gap = sweep.betti_at_threshold(dgms, args.eps_scale, rule="gap",
                                       max_candidates=args.gap_max_candidates)
        mrule = sweep.betti_at_threshold(dgms, args.eps_scale, rule="median")
        print(f"\n  rule=median -> b0={mrule['betti'][0]:<5} b1={mrule['betti'][1]}"
              f"   (thresholds {mrule['thresholds'][0]:.4g}, {mrule['thresholds'][1]:.4g})")
        print(f"  rule=gap    -> b0={gap['betti'][0]:<5} b1={gap['betti'][1]}"
              f"   (thresholds {gap['thresholds'][0]:.4g}, {gap['thresholds'][1]:.4g})")

        if args.hist:
            args.hist.parent.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
            fig.patch.set_facecolor("#fcfcfb")
            for ax in axes:
                ax.set_facecolor("#fcfcfb")
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
            axes[0].hist(deaths, bins=60, color="#2a78d6")
            axes[0].axvline(thr_med, color="#eb6834", lw=2,
                            label=f"median rule thr = {thr_med:.3g}")
            axes[0].set_xlabel("H0 death time"); axes[0].set_ylabel("count")
            axes[0].set_title("linear bins", loc="left")
            axes[0].legend(fontsize=9)
            pos = deaths[deaths > 0]
            axes[1].hist(pos, bins=np.logspace(np.log10(pos.min()),
                                               np.log10(pos.max()), 60),
                         color="#2a78d6")
            axes[1].axvline(thr_med, color="#eb6834", lw=2)
            axes[1].set_xscale("log")
            axes[1].set_xlabel("H0 death time (log)")
            axes[1].set_title("log bins", loc="left")
            fig.suptitle(f"H0 death-time distribution — {args.system}, "
                         f"theta={args.theta}, weights={weights}", x=0.01, ha="left")
            fig.tight_layout()
            fig.savefig(args.hist, dpi=180, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"\nWrote {args.hist}")
            log.note("hist", str(args.hist))

        log.note("b0_median", mrule["betti"][0])
        log.note("b0_gap", gap["betti"][0])
        log.close("ok")
        return 0
    except Exception as exc:
        log.close("error", exc)
        raise
    finally:
        print(f"Run log: {log.path}")


if __name__ == "__main__":
    raise SystemExit(main())
