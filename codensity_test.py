#!/usr/bin/env python3
"""Codensity diagnostic: is the attractor cloud uniformly sparse or cored?

For each theta, the cloud is built EXACTLY as sweep.py builds it -- same IC
grid and box, same t_span, same keep_frac, same subsample, same blowup -- by
calling sweep.py's own functions.  Nothing in the existing pipeline is
modified or re-tuned here; this script only looks.

codensity(points, k) is the distance from each point to its k-th nearest
neighbour.  Small codensity = locally dense, large = locally sparse.  The
distribution answers the shape question; sweeping a threshold s and keeping
only points with codensity <= s answers whether a dense core carries the H1
loop or whether the loop lives in the sparse outskirts.

Report only: no threshold, floor, or angle is tuned toward any expected
answer, and the longest H1 persistence is reported raw.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ripser import ripser
from scipy.spatial import cKDTree

from normal_forms import BLOWUP_WEIGHTS, RHS, blowup, param_path
from runlog import open_run_log
from sweep import initial_conditions, integrate_grid, subsample

HERE = Path(__file__).resolve().parent

#: Angles requested for the BT diagnostic.  4.65 and 4.76 bracket 4.712.
DEFAULT_THETAS = [4.65, 4.712, 4.76]


def codensity(points, k=12):
    """Distance from each point to its k-th nearest neighbour.

    points : (N, 2) array (any dimension works)
    k      : neighbour rank, 1-based and EXCLUDING the point itself, so k=12
             means the 12th-nearest other point.

    Returns a (N,) array of distances.  Raises if the cloud has fewer than
    k+1 points, since the k-th neighbour would not exist.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError(f"expected a (N, d) point cloud, got shape {pts.shape}")
    if len(pts) <= k:
        raise ValueError(f"cloud has {len(pts)} points; need > k = {k}")
    # query k+1 because the first neighbour returned is the point itself (d=0)
    dists, _ = cKDTree(pts).query(pts, k=k + 1)
    return dists[:, k]


def longest_h1(points, maxdim=1):
    """Longest finite H1 persistence of a point cloud, raw and unthresholded.

    Returns (persistence, n_finite_h1_bars).  A cloud with no finite H1 bar
    gives (0.0, 0).
    """
    dgms = ripser(points, maxdim=maxdim)["dgms"]
    if len(dgms) < 2 or len(dgms[1]) == 0:
        return 0.0, 0
    h1 = np.asarray(dgms[1], dtype=float)
    h1 = h1[np.isfinite(h1[:, 1])]
    if len(h1) == 0:
        return 0.0, 0
    return float(np.max(h1[:, 1] - h1[:, 0])), int(len(h1))


def build_cloud(theta, args, rhs, ics):
    """The pipeline's cloud at one theta, in blowup coordinates.

    Mirrors analyse_angle()'s construction order: integrate, subsample with
    the pipeline's seed, then blow up.  Returns (uv, info).
    """
    b1, b2 = param_path(theta, args.eps)
    cloud, n_bounded, n_diverged, n_failed = integrate_grid(
        rhs, b1, b2, ics,
        t_span=(args.t0, args.t1), n_samples=args.n_samples,
        keep_frac=args.keep_frac, rtol=args.rtol, atol=args.atol,
        escape_radius=args.escape_radius,
    )
    info = {"b1": b1, "b2": b2, "n_bounded": n_bounded,
            "n_diverged": n_diverged, "n_failed": n_failed,
            "n_raw": len(cloud)}
    if len(cloud) == 0:
        return np.empty((0, 2)), info

    cloud = subsample(cloud, args.max_cloud, args.subsample_seed)
    uv = blowup(cloud, args.eps, weights=args.blowup_weights)
    info["n_cloud"] = len(uv)
    if len(uv):
        info["diameter"] = float(np.linalg.norm(uv.max(axis=0) - uv.min(axis=0)))
    return uv, info


def analyse_angle(theta, args, rhs, ics):
    """Codensity distribution plus the threshold sweep at one theta."""
    uv, info = build_cloud(theta, args, rhs, ics)
    if len(uv) <= args.k:
        return {"theta": theta, "info": info, "cod": None, "sweep": [],
                "note": f"cloud too small ({len(uv)} points <= k={args.k})"}

    cod = codensity(uv, k=args.k)
    pcts = np.linspace(args.s_min_pct, args.s_max_pct, args.n_s)
    s_values = np.percentile(cod, pcts)

    rows = []
    for pct, s in zip(pcts, s_values):
        keep = uv[cod <= s]
        if len(keep) < args.min_kept:
            rows.append({"pct": pct, "s": s, "n_kept": len(keep),
                         "h1_pers": None, "n_h1": 0,
                         "note": f"< {args.min_kept} points"})
            continue
        pers, n_h1 = longest_h1(keep)
        rows.append({"pct": pct, "s": s, "n_kept": len(keep),
                     "h1_pers": pers, "n_h1": n_h1, "note": ""})
    return {"theta": theta, "info": info, "cod": cod, "sweep": rows, "note": ""}


def build_parser():
    p = argparse.ArgumentParser(
        description="Codensity diagnostic on the BT attractor cloud "
                    "(report only; changes nothing in the sweep pipeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--system", choices=sorted(RHS), default="bt")
    p.add_argument("--thetas", type=float, nargs="+", default=DEFAULT_THETAS,
                   help="angles to analyse (rad)")
    p.add_argument("--k", type=int, default=12,
                   help="neighbour rank for codensity")
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--blowup-weights", default="auto",
                   help="'p,q' or 'auto' for the per-system default")

    p.add_argument("--box", type=float, nargs=4,
                   default=[-1.5, 2.0, -1.5, 1.5],
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
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

    p.add_argument("--s-min-pct", type=float, default=10.0,
                   help="lowest codensity percentile used as a threshold")
    p.add_argument("--s-max-pct", type=float, default=90.0,
                   help="highest codensity percentile used as a threshold")
    p.add_argument("--n-s", type=int, default=9,
                   help="number of threshold steps")
    p.add_argument("--min-kept", type=int, default=20,
                   help="skip ripser below this many surviving points")
    p.add_argument("--out", default="results/codensity_bt.csv",
                   help="CSV of the threshold sweep")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.blowup_weights == "auto":
        args.blowup_weights = BLOWUP_WEIGHTS[args.system]
    else:
        args.blowup_weights = tuple(
            int(v) for v in str(args.blowup_weights).split(","))

    log = open_run_log("codensity_test", args)
    try:
        rhs = RHS[args.system]
        ics = initial_conditions(args.box, args.nx, args.ny,
                                 coords=args.box_coords, eps=args.eps)

        results = []
        for theta in args.thetas:
            res = analyse_angle(theta, args, rhs, ics)
            results.append(res)
            print(f"theta = {theta:.5f}  b1 = {res['info']['b1']:+.6e}  "
                  f"b2 = {res['info']['b2']:+.6e}  "
                  f"bounded = {res['info']['n_bounded']}/{len(ics)}  "
                  f"cloud = {res['info'].get('n_cloud', 0)}  "
                  f"diam = {res['info'].get('diameter', float('nan')):.5f}")
            if res["note"]:
                print(f"  NOTE: {res['note']}")
        print()

        # ---- 3. codensity distribution per angle ----
        print(f"CODENSITY DISTRIBUTION (k = {args.k}, blowup coords)")
        print(f"{'theta':>9}  {'min':>10}  {'median':>10}  {'p90':>10}  "
              f"{'max':>10}  {'max/med':>8}  {'diam':>8}")
        for res in results:
            cod = res["cod"]
            if cod is None:
                print(f"{res['theta']:9.5f}  (no distribution: {res['note']})")
                continue
            mn, md = float(cod.min()), float(np.median(cod))
            p90, mx = float(np.percentile(cod, 90)), float(cod.max())
            print(f"{res['theta']:9.5f}  {mn:10.6f}  {md:10.6f}  {p90:10.6f}  "
                  f"{mx:10.6f}  {mx / md:8.2f}  "
                  f"{res['info'].get('diameter', float('nan')):8.5f}")
            log.note(f"codensity_{res['theta']:.5f}",
                     {"min": mn, "median": md, "p90": p90, "max": mx})
        print()

        # ---- 4/5. threshold sweep, theta down the rows ----
        pcts = np.linspace(args.s_min_pct, args.s_max_pct, args.n_s)
        print("LONGEST H1 PERSISTENCE vs CODENSITY THRESHOLD s "
              "(raw, no floor; s = percentile of that angle's codensity)")
        header = f"{'theta':>9}  " + "  ".join(f"p{p:02.0f}".rjust(8)
                                               for p in pcts)
        print(header)
        for res in results:
            cells = []
            for row in res["sweep"]:
                cells.append("      --" if row["h1_pers"] is None
                             else f"{row['h1_pers']:8.5f}")
            if not cells:
                cells = ["      --"] * len(pcts)
            print(f"{res['theta']:9.5f}  " + "  ".join(cells))

        print()
        print("points surviving each threshold (same layout)")
        print(header)
        for res in results:
            cells = [f"{row['n_kept']:8d}" for row in res["sweep"]]
            if not cells:
                cells = ["      --"] * len(pcts)
            print(f"{res['theta']:9.5f}  " + "  ".join(cells))

        out = HERE / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["theta", "b1", "b2", "k", "n_cloud", "diameter",
                        "cod_min", "cod_median", "cod_p90", "cod_max",
                        "s_pct", "s", "n_kept", "h1_max_pers", "n_h1_bars",
                        "note"])
            for res in results:
                cod = res["cod"]
                stats = (["", "", "", ""] if cod is None else
                         [cod.min(), np.median(cod),
                          np.percentile(cod, 90), cod.max()])
                for row in res["sweep"]:
                    w.writerow([
                        res["theta"], res["info"]["b1"], res["info"]["b2"],
                        args.k, res["info"].get("n_cloud", 0),
                        res["info"].get("diameter", ""), *stats,
                        row["pct"], row["s"], row["n_kept"],
                        "" if row["h1_pers"] is None else row["h1_pers"],
                        row["n_h1"], row["note"],
                    ])
        print(f"\nwrote {out}")
        log.note("csv", str(out))
        log.close()
    except Exception as exc:
        log.close(status="error", error=exc)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
