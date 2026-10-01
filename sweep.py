#!/usr/bin/env python3
"""Sweep the anisotropic parameter path and compute Betti numbers via ripser.

For each theta on [0, 2pi):
  1. map theta -> (b1, b2) with normal_forms.param_path
  2. integrate a grid of initial conditions (solve_ivp, LSODA)
  3. drop the transient, union the retained points into one cloud
  4. subsample, apply the blowup, run ripser
  5. read off (b0, b1) with a selectable thresholding rule

Angles where the trajectories escape instead of settling onto a bounded
attractor are recorded as UNDEFINED -- never as zero.  "No attractor to
measure" and "an attractor with no holes" are different facts.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from ripser import ripser
from scipy.integrate import solve_ivp

from normal_forms import BLOWUP_WEIGHTS, RHS, blowup, param_path
from runlog import open_run_log, timestamp

HERE = Path(__file__).resolve().parent

#: Sentinel written to the CSV where no bounded attractor exists.
UNDEFINED = "UNDEFINED"


# --------------------------------------------------------------------------
# Betti extraction
# --------------------------------------------------------------------------

def _finite(dgm):
    """Finite (birth, death) rows of a ripser diagram, plus the count of
    infinite bars (H0 always has exactly one: the whole space)."""
    if dgm is None or len(dgm) == 0:
        return np.empty((0, 2)), 0
    dgm = np.asarray(dgm, dtype=float)
    inf_mask = ~np.isfinite(dgm[:, 1])
    return dgm[~inf_mask], int(inf_mask.sum())


def _median_threshold(dgm, eps_scale):
    """Rule (a): threshold = eps_scale * median finite death time."""
    fin, _ = _finite(dgm)
    if len(fin) == 0:
        return 0.0
    return float(eps_scale) * float(np.median(fin[:, 1]))


def _gap_threshold(dgm, eps_scale, max_candidates):
    """Rule (b): longest-bar gap heuristic.

    Sort persistences descending, look at the successive drops among the top
    `max_candidates` bars, and cut at the largest one.  The threshold is the
    midpoint of that gap, so the bars above it are the "long" ones.  Falls
    back to the median rule when there are too few bars for a gap to exist.
    """
    fin, _ = _finite(dgm)
    if len(fin) < 2:
        return _median_threshold(dgm, eps_scale)

    pers = np.sort(fin[:, 1] - fin[:, 0])[::-1]
    top = pers[: max(2, int(max_candidates))]
    drops = top[:-1] - top[1:]
    k = int(np.argmax(drops))            # cut between top[k] and top[k+1]
    return float(0.5 * (top[k] + top[k + 1]))


def betti_at_threshold(diagrams, eps_scale, rule="median", max_candidates=12,
                       diameter=None, min_persistence_frac=0.0):
    """Count bars whose persistence exceeds a data-derived threshold.

    diagrams    : list of ripser diagrams, [H0, H1, ...]
    eps_scale   : scale factor -- a multiple of the median death time under
                  rule "median"; a fallback scale under rule "gap"
    rule        : "median" (fixed multiple of the median death time) or
                  "gap" (longest-bar gap heuristic).  Both are exposed on the
                  command line via --betti-rule.
    diameter    : diameter of the point cloud the diagrams came from; required
                  for the persistence floor to do anything
    min_persistence_frac
                : absolute floor applied ON TOP of whichever rule is selected --
                  a bar counts only if its persistence also exceeds
                  min_persistence_frac * diameter.  Applied to every dimension.
                  The effective threshold is max(rule threshold, floor), so the
                  floor can only ever remove bars, never add them.

    Returns {"betti": [...], "thresholds": [...], "n_bars": [...],
             "rule_thresholds": [...], "floor": float,
             "max_persistence": [...]}.
    Infinite bars always count (the one infinite H0 bar is the ambient
    connected component) -- infinite persistence clears any finite floor.
    """
    if rule not in ("median", "gap"):
        raise ValueError(f"unknown betti rule {rule!r}; use 'median' or 'gap'")

    floor = 0.0
    if diameter is not None and min_persistence_frac > 0:
        floor = float(min_persistence_frac) * float(diameter)

    betti, thresholds, rule_thresholds, n_bars, max_pers = [], [], [], [], []
    for dgm in diagrams:
        fin, n_inf = _finite(dgm)
        thr_rule = (_median_threshold(dgm, eps_scale) if rule == "median"
                    else _gap_threshold(dgm, eps_scale, max_candidates))
        thr = max(float(thr_rule), floor)
        pers = fin[:, 1] - fin[:, 0] if len(fin) else np.empty(0)
        betti.append(int(n_inf + int(np.sum(pers > thr))))
        thresholds.append(float(thr))
        rule_thresholds.append(float(thr_rule))
        n_bars.append(int(len(fin) + n_inf))
        max_pers.append(float(pers.max()) if len(pers) else 0.0)
    return {"betti": betti, "thresholds": thresholds,
            "rule_thresholds": rule_thresholds, "floor": floor,
            "n_bars": n_bars, "max_persistence": max_pers}


# --------------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------------

def initial_conditions(box, nx, ny, coords="blown", eps=None):
    """Regular grid of initial conditions over box = (xmin, xmax, ymin, ymax).

    coords="raw"   : the box is in phase-space (x, y).
    coords="blown" : the box is in blowup coordinates (u, v) and is mapped
                     back with x = u*eps^2, y = v*eps^3.  This is the default
                     because the interesting dynamics sit at x ~ eps^2,
                     y ~ eps^3, so a fixed raw box is either far too big or
                     far too small as eps changes.
    """
    xmin, xmax, ymin, ymax = box
    if coords == "blown":
        if eps is None:
            raise ValueError("eps is required for blown-up box coordinates")
        xmin, xmax = xmin * eps ** 2, xmax * eps ** 2
        ymin, ymax = ymin * eps ** 3, ymax * eps ** 3
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel()])


def integrate_grid(rhs, b1, b2, ics, t_span, n_samples, keep_frac,
                   rtol, atol, escape_radius):
    """Integrate every initial condition; return the union of retained points.

    A trajectory is 'diverged' if it leaves escape_radius (a terminal event)
    or if the stiff solver fails on it.  Diverged trajectories contribute no
    points at all -- their unbounded tail would otherwise dominate the cloud.

    Returns (cloud (M,2), n_bounded, n_diverged, n_failed).
    """

    def escaped(t, state, *params):
        return escape_radius - float(np.hypot(state[0], state[1]))

    escaped.terminal = True
    escaped.direction = -1.0

    t_eval = np.linspace(t_span[0], t_span[1], n_samples)
    n_keep = max(2, int(round(keep_frac * n_samples)))

    kept, n_bounded, n_diverged, n_failed = [], 0, 0, 0
    for ic in ics:
        try:
            sol = solve_ivp(
                rhs, t_span, ic, method="LSODA", args=(b1, b2),
                t_eval=t_eval, rtol=rtol, atol=atol, events=escaped,
            )
        except Exception:
            n_failed += 1
            n_diverged += 1
            continue

        # status 1 == a terminal event fired, i.e. the trajectory escaped.
        if sol.status == 1:
            n_diverged += 1
            continue
        if not sol.success or sol.y.shape[1] < n_keep:
            n_failed += 1
            n_diverged += 1
            continue

        traj = sol.y[:, -n_keep:].T          # keep the last keep_frac of time
        if not np.all(np.isfinite(traj)):
            n_failed += 1
            n_diverged += 1
            continue

        kept.append(traj)
        n_bounded += 1

    cloud = np.vstack(kept) if kept else np.empty((0, 2))
    return cloud, n_bounded, n_diverged, n_failed


def subsample(cloud, max_points, seed):
    """Uniformly subsample the cloud to at most max_points, fixed seed."""
    if len(cloud) <= max_points:
        return cloud
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(cloud), size=max_points, replace=False)
    return cloud[np.sort(idx)]


# --------------------------------------------------------------------------
# One angle
# --------------------------------------------------------------------------

def analyse_angle(theta, args, rhs, ics):
    """Full pipeline at one theta.  Returns a CSV-ready row dict."""
    b1, b2 = param_path(theta, args.eps)

    cloud, n_bounded, n_diverged, n_failed = integrate_grid(
        rhs, b1, b2, ics,
        t_span=(args.t0, args.t1), n_samples=args.n_samples,
        keep_frac=args.keep_frac, rtol=args.rtol, atol=args.atol,
        escape_radius=args.escape_radius,
    )

    # Backward time: the same IC grid integrated with t reversed, so the
    # retained tail is the backward-asymptotic end.  This exposes REPELLING
    # invariant sets (an unstable limit cycle is an attractor in reverse
    # time), which forward integration can never see.
    #
    # Backward escapes deliberately do NOT feed the UNDEFINED test below:
    # trajectories running away from a stable attractor in reverse time is
    # the expected behaviour, not evidence that no attractor exists.  The
    # bounded/divergent verdict stays a forward-time statement.
    n_bounded_bwd, row_note = 0, ""
    if args.backward_time:
        span = args.t1 - args.t0
        cloud_bwd, n_bounded_bwd, _, _ = integrate_grid(
            rhs, b1, b2, ics,
            t_span=(args.t0, args.t0 - span), n_samples=args.n_samples,
            keep_frac=args.keep_frac, rtol=args.rtol, atol=args.atol,
            escape_radius=args.escape_radius,
        )
        # A backward cloud built from a handful of trajectories is a few thin
        # arcs, and unioning it in creates loops between those arcs and the
        # forward attractor that no invariant set corresponds to.  Observed
        # directly: at 4 bounded backward trajectories the Bautin sweep
        # reported b1 = 2 and even 5 at angles far outside the analytic
        # two-cycle window.  Below the floor the backward sample is dropped.
        row_note = ""
        if n_bounded_bwd < args.min_bounded_bwd:
            row_note = (f"backward cloud dropped: {n_bounded_bwd} bounded "
                        f"trajectories < {args.min_bounded_bwd}")
        elif len(cloud_bwd):
            cloud = np.vstack([cloud, cloud_bwd]) if len(cloud) else cloud_bwd

    row = {
        "theta": theta, "b1": b1, "b2": b2,
        "n_bounded": n_bounded, "n_diverged": n_diverged,
        "n_failed": n_failed, "n_ics": len(ics),
        "n_bounded_bwd": n_bounded_bwd,
        "bounded_frac": n_bounded / len(ics),
        "n_cloud": 0, "diameter": "", "b0": UNDEFINED, "b1_betti": UNDEFINED,
        "thr_h0": "", "thr_h1": "", "rule_thr_h0": "", "rule_thr_h1": "",
        "pers_floor": "", "h0_max_pers": "", "h1_max_pers": "",
        "n_bars_h0": "", "n_bars_h1": "",
        "status": UNDEFINED, "reason": "",
    }

    # -- the load-bearing distinction: no bounded attractor => UNDEFINED --
    # A *small basin* is not the same as *no attractor*, so this is an
    # absolute count, not a majority vote: near a saddle-node the basin can
    # be a few percent of the box while the attractor is perfectly real.
    if n_bounded < args.min_bounded_traj:
        row["reason"] = (f"divergent: only {n_bounded}/{len(ics)} trajectories "
                         f"bounded (< {args.min_bounded_traj})")
        return row
    if row["bounded_frac"] < args.min_bounded_frac:
        row["reason"] = (f"divergent: bounded fraction {row['bounded_frac']:.3f} "
                         f"< {args.min_bounded_frac}")
        return row

    cloud = subsample(cloud, args.max_cloud, args.subsample_seed)
    row["n_cloud"] = len(cloud)
    if len(cloud) < args.min_cloud:
        row["reason"] = (f"insufficient points: {len(cloud)} retained "
                         f"(< {args.min_cloud})")
        return row

    uv = blowup(cloud, args.eps, weights=args.blowup_weights)
    if not np.all(np.isfinite(uv)):
        row["reason"] = "non-finite coordinates after blowup"
        return row

    # Degenerate case: every trajectory has collapsed onto a single point to
    # within solver precision.  The cloud is then a ball of roundoff, and
    # ripser will happily find "structure" in it -- at the cusp Hopf angle a
    # cloud of diameter 1.8e-8 was yielding a spurious b1=1.  This is a
    # *defined* answer (a point attractor is bounded), so it is OK, not
    # UNDEFINED: the distinction is between "nothing to measure" and "a
    # measurable thing with no holes", and this is the latter.
    diameter = float(np.linalg.norm(uv.max(axis=0) - uv.min(axis=0)))
    row["diameter"] = diameter
    if diameter < args.degenerate_tol:
        row.update({
            "b0": 1, "b1_betti": 0, "status": "OK",
            "reason": (f"point attractor: cloud diameter {diameter:.3e} < "
                       f"{args.degenerate_tol:g}, below solver precision"),
        })
        return row

    dgms = ripser(uv, maxdim=1)["dgms"]
    res = betti_at_threshold(dgms, args.eps_scale, rule=args.betti_rule,
                             max_candidates=args.gap_max_candidates,
                             diameter=diameter,
                             min_persistence_frac=args.min_persistence_frac)

    row.update({
        "b0": res["betti"][0], "b1_betti": res["betti"][1],
        "thr_h0": res["thresholds"][0], "thr_h1": res["thresholds"][1],
        "rule_thr_h0": res["rule_thresholds"][0],
        "rule_thr_h1": res["rule_thresholds"][1],
        "pers_floor": res["floor"],
        # Raw, un-thresholded persistence of the longest finite bar in each
        # dimension -- recorded independently of any rule or floor so the
        # underlying signal can be plotted against theta directly.
        "h0_max_pers": res["max_persistence"][0],
        "h1_max_pers": res["max_persistence"][1],
        "n_bars_h0": res["n_bars"][0], "n_bars_h1": res["n_bars"][1],
        "status": "OK", "reason": row_note,
    })
    return row


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

FIELDS = ["idx", "theta", "b1", "b2", "status", "b0", "b1_betti",
          "h1_max_pers", "h0_max_pers", "pers_floor",
          "thr_h0", "thr_h1", "rule_thr_h0", "rule_thr_h1",
          "n_bars_h0", "n_bars_h1", "n_cloud",
          "n_ics", "n_bounded", "n_bounded_bwd", "n_diverged", "n_failed",
          "bounded_frac", "diameter", "reason"]


def build_parser():
    p = argparse.ArgumentParser(
        description="Sweep theta over the anisotropic parameter path and "
                    "compute Betti numbers of the attractor point cloud.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("system")
    g.add_argument("--system", choices=sorted(RHS), default="bt",
                   help="normal form to integrate")
    g.add_argument("--eps", type=float, default=0.3,
                   help="path radius; b1=eps^4*cos(theta), b2=eps^2*sin(theta)")
    g.add_argument("--n-theta", type=int, default=36,
                   help="number of angles sampled on the theta interval")
    g.add_argument("--theta-min", type=float, default=0.0,
                   help="start of the theta interval (rad)")
    g.add_argument("--theta-max", type=float, default=2.0 * np.pi,
                   help="end of the theta interval (rad); the endpoint is "
                        "included for a sub-interval and excluded for a full "
                        "2pi sweep, where it would duplicate theta-min")
    g.add_argument("--blowup-weights", default="auto",
                   help="blowup exponents 'p,q' for u=x/eps^p, v=y/eps^q; "
                        "'auto' uses the per-system default "
                        f"({', '.join(f'{k}={v}' for k, v in BLOWUP_WEIGHTS.items())})")

    g = p.add_argument_group("initial conditions")
    g.add_argument("--box", type=float, nargs=4, default=[-1.5, 2.0, -1.5, 1.5],
                   metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="box of initial conditions")
    g.add_argument("--box-coords", choices=["blown", "raw"], default="blown",
                   help="interpret --box in blowup coords (u,v) or raw phase "
                        "space (x,y); blown-up keeps the box meaningful as eps varies")
    g.add_argument("--nx", type=int, default=12, help="grid points in x")
    g.add_argument("--ny", type=int, default=12, help="grid points in y")

    g = p.add_argument_group("integration (LSODA, stiff-aware)")
    g.add_argument("--t0", type=float, default=0.0, help="integration start time")
    g.add_argument("--t1", type=float, default=300.0, help="integration end time")
    g.add_argument("--n-samples", type=int, default=400,
                   help="time samples per trajectory")
    g.add_argument("--keep-frac", type=float, default=0.3,
                   help="fraction of each trajectory retained (the last "
                        "keep-frac of samples); the rest is transient")
    g.add_argument("--rtol", type=float, default=1e-8, help="solver relative tolerance")
    g.add_argument("--atol", type=float, default=1e-10, help="solver absolute tolerance")
    g.add_argument("--escape-radius", type=float, default=50.0,
                   help="|state| beyond this counts as divergence (terminal event)")
    g.add_argument("--min-bounded-bwd", type=int, default=8,
                   help="minimum bounded backward trajectories before the "
                        "backward cloud is unioned in; below this it is too "
                        "sparse to be topologically meaningful and is dropped")
    g.add_argument("--backward-time", action="store_true",
                   help="also integrate the IC grid with t reversed and union "
                        "the backward-asymptotic points into the same cloud; "
                        "reveals repelling sets. Off by default. Does not "
                        "affect the bounded/UNDEFINED verdict.")

    g = p.add_argument_group("point cloud")
    g.add_argument("--max-cloud", type=int, default=1200,
                   help="max points passed to ripser after subsampling")
    g.add_argument("--degenerate-tol", type=float, default=1e-6,
                   help="blown-up cloud diameter below this means a point "
                        "attractor: b0=1, b1=0 (still OK, never UNDEFINED) -- "
                        "guards against ripser reading solver roundoff as topology")
    g.add_argument("--min-cloud", type=int, default=50,
                   help="fewer retained points than this => UNDEFINED")
    g.add_argument("--subsample-seed", type=int, default=1234,
                   help="fixed seed for subsampling")
    g.add_argument("--min-bounded-traj", type=int, default=3,
                   help="fewer bounded trajectories than this => no bounded "
                        "attractor => UNDEFINED (absolute count, so a small "
                        "basin is not mistaken for divergence)")
    g.add_argument("--min-bounded-frac", type=float, default=0.0,
                   help="optional extra bounded-fraction floor; 0 disables it")

    g = p.add_argument_group("Betti thresholding")
    g.add_argument("--betti-rule", choices=["median", "gap"], default="gap",
                   help="(a) median: fixed multiple of the median death time; "
                        "(b) gap: longest-bar gap heuristic")
    g.add_argument("--eps-scale", type=float, default=2.0,
                   help="multiplier for the median rule (and gap fallback)")
    g.add_argument("--min-persistence-frac", type=float, default=0.1,
                   help="absolute floor applied on top of the selected rule, in "
                        "BOTH H0 and H1: a bar counts only if its persistence "
                        "also exceeds this fraction of the point cloud's "
                        "diameter. 0 disables it.")
    g.add_argument("--gap-max-candidates", type=int, default=12,
                   help="how many longest bars the gap rule considers")

    g = p.add_argument_group("output")
    g.add_argument("--out", type=Path, default=None,
                   help="CSV path (default: results/sweep_<system>_<timestamp>.csv)")
    g.add_argument("--quiet", action="store_true", help="suppress per-angle progress")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    stamp = timestamp()
    if args.out is None:
        args.out = HERE / "results" / f"sweep_{args.system}_{stamp}.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    log = open_run_log("sweep", args, stamp=stamp)
    try:
        if not 0.0 < args.keep_frac <= 1.0:
            raise SystemExit("--keep-frac must be in (0, 1]")

        if args.blowup_weights == "auto":
            args.blowup_weights = BLOWUP_WEIGHTS[args.system]
        else:
            try:
                pq = tuple(float(v) for v in str(args.blowup_weights).split(","))
            except ValueError:
                raise SystemExit("--blowup-weights must be 'auto' or 'p,q'")
            if len(pq) != 2:
                raise SystemExit("--blowup-weights must be 'auto' or 'p,q'")
            args.blowup_weights = pq

        rhs = RHS[args.system]
        ics = initial_conditions(args.box, args.nx, args.ny,
                                 coords=args.box_coords, eps=args.eps)
        # A full-circle sweep must exclude the endpoint (theta_max == theta_min
        # mod 2pi would duplicate a sample); a sub-interval should include it.
        full_circle = abs((args.theta_max - args.theta_min) - 2.0 * np.pi) < 1e-12
        thetas = np.linspace(args.theta_min, args.theta_max, args.n_theta,
                             endpoint=not full_circle)

        print(f"system={args.system}  eps={args.eps}  n_theta={args.n_theta}  "
              f"ics={len(ics)}  rule={args.betti_rule}  eps_scale={args.eps_scale}")
        print(f"theta in [{args.theta_min:.4f}, {args.theta_max:.4f}]"
              f"{'' if full_circle else ' (endpoint included)'}  "
              f"blowup weights={tuple(args.blowup_weights)}  "
              f"backward_time={args.backward_time}")
        print(f"LSODA rtol={args.rtol} atol={args.atol} t_span=({args.t0}, {args.t1}) "
              f"keep_frac={args.keep_frac}")

        rows = []
        for i, th in enumerate(thetas):
            row = analyse_angle(float(th), args, rhs, ics)
            row["idx"] = i
            rows.append(row)
            if not args.quiet:
                if row["status"] == "OK":
                    print(f"  [{i:>3}/{args.n_theta}] theta={th:6.4f}  "
                          f"b0={row['b0']}  b1={row['b1_betti']}  "
                          f"cloud={row['n_cloud']}  bounded={row['n_bounded']}/{row['n_ics']}")
                else:
                    print(f"  [{i:>3}/{args.n_theta}] theta={th:6.4f}  "
                          f"{UNDEFINED}  ({row['reason']})")

        n_undef = sum(r["status"] == UNDEFINED for r in rows)
        with args.out.open("w", newline="") as fh:
            fh.write(f"# tda-normal-forms sweep\n")
            fh.write(f"# timestamp_utc: {stamp}\n")
            fh.write(f"# run_log: {log.path.name}\n")
            for k, v in sorted(vars(args).items()):
                fh.write(f"# {k}: {v}\n")
            fh.write(f"# n_undefined: {n_undef} / {len(rows)}\n")
            fh.write(f"# note: status={UNDEFINED} means no bounded attractor "
                     f"(or too few points); b0/b1 are {UNDEFINED}, NOT zero\n")
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

        print(f"\n{len(rows)} angles, {n_undef} UNDEFINED")
        print(f"Wrote {args.out}")
        log.note("csv", str(args.out))
        log.note("n_undefined", n_undef)
        log.note("n_angles", len(rows))
        log.close("ok")
        return 0
    except Exception as exc:
        log.close("error", exc)
        raise
    finally:
        print(f"Run log: {log.path}")


if __name__ == "__main__":
    raise SystemExit(main())
