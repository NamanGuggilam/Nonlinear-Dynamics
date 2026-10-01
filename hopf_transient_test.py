#!/usr/bin/env python3
"""Does the raw H1 persistence curve develop a kink at the Hopf point when the
transient is truncated harder?

BT normal form. For each theta the IC grid is integrated ONCE at the full
t_span, then several transient truncations are sliced from the same
trajectories -- so the keep-fraction is the only thing that varies between
columns, and the integration is not repeated four times.

Reports the RAW longest-H1 persistence. No persistence floor is applied
anywhere in this script.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from ripser import ripser
from scipy.integrate import solve_ivp

from normal_forms import RHS, blowup, param_path
from runlog import open_run_log, timestamp
from sweep import initial_conditions, subsample

HERE = Path(__file__).resolve().parent


def integrate_full(rhs, b1, b2, ics, t_span, n_samples, rtol, atol, escape_radius):
    """Integrate every IC once and return the FULL bounded trajectories.

    Same divergence handling as sweep.integrate_grid (terminal escape event,
    solver failure, non-finite states) -- only the return shape differs, so
    the tails can be sliced afterwards.
    """
    def escaped(t, state, *params):
        return escape_radius - float(np.hypot(state[0], state[1]))
    escaped.terminal = True
    escaped.direction = -1.0

    t_eval = np.linspace(t_span[0], t_span[1], n_samples)
    trajs = []
    for ic in ics:
        try:
            sol = solve_ivp(rhs, t_span, ic, method="LSODA", args=(b1, b2),
                            t_eval=t_eval, rtol=rtol, atol=atol, events=escaped)
        except Exception:
            continue
        if sol.status != 0 or not sol.success or sol.y.shape[1] < n_samples:
            continue
        traj = sol.y.T
        if not np.all(np.isfinite(traj)):
            continue
        trajs.append(traj)
    return trajs


def h1_stats(trajs, keep_frac, n_samples, eps, weights, max_cloud, seed, min_cloud):
    """Raw longest-H1 persistence for one transient truncation."""
    n_keep = max(2, int(round(keep_frac * n_samples)))
    tails = [t[-n_keep:] for t in trajs]
    if not tails:
        return None
    cloud = np.vstack(tails)
    cloud = subsample(cloud, max_cloud, seed)
    if len(cloud) < min_cloud:
        return None
    uv = blowup(cloud, eps, weights=weights)
    ext = uv.max(axis=0) - uv.min(axis=0)
    diameter = float(np.linalg.norm(ext))
    dgms = ripser(uv, maxdim=1)["dgms"]
    h1 = dgms[1]
    if len(h1) == 0:
        pers = 0.0
    else:
        fin = h1[np.isfinite(h1[:, 1])]
        pers = float((fin[:, 1] - fin[:, 0]).max()) if len(fin) else 0.0
    return {"h1_max_pers": pers, "diameter": diameter, "n_cloud": len(cloud),
            "n_keep": n_keep}


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Transient-truncation sweep of the raw H1 persistence curve.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--system", choices=sorted(RHS), default="bt")
    p.add_argument("--theta-min", type=float, default=4.60)
    p.add_argument("--theta-max", type=float, default=4.85)
    p.add_argument("--n-theta", type=int, default=60)
    p.add_argument("--eps", type=float, default=0.3)
    p.add_argument("--keep-fracs", type=float, nargs="+",
                   default=[0.30, 0.10, 0.03, 0.01])
    p.add_argument("--t0", type=float, default=0.0)
    p.add_argument("--t1", type=float, default=1200.0,
                   help="4x the sweep.py default of 300")
    p.add_argument("--n-samples", type=int, default=6000,
                   help="high enough that the smallest keep-fraction still "
                        "yields more points than --max-cloud, so cloud size is "
                        "controlled across columns")
    p.add_argument("--box", type=float, nargs=4, default=[-1.5, 2.0, -1.5, 1.5])
    p.add_argument("--box-coords", choices=["blown", "raw"], default="blown")
    p.add_argument("--nx", type=int, default=12)
    p.add_argument("--ny", type=int, default=12)
    p.add_argument("--rtol", type=float, default=1e-8)
    p.add_argument("--atol", type=float, default=1e-10)
    p.add_argument("--escape-radius", type=float, default=50.0)
    p.add_argument("--max-cloud", type=int, default=400)
    p.add_argument("--min-cloud", type=int, default=100)
    p.add_argument("--subsample-seed", type=int, default=1234)
    p.add_argument("--blowup-weights", default="2,3")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    weights = tuple(float(v) for v in str(args.blowup_weights).split(","))
    stamp = timestamp()
    if args.out is None:
        args.out = HERE / "results" / f"hopf_transient_{stamp}.csv"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    log = open_run_log("hopf_transient_test", args, stamp=stamp)
    try:
        rhs = RHS[args.system]
        ics = initial_conditions(args.box, args.nx, args.ny,
                                 coords=args.box_coords, eps=args.eps)
        thetas = np.linspace(args.theta_min, args.theta_max, args.n_theta)
        fracs = list(args.keep_fracs)

        print(f"system={args.system} eps={args.eps} t_span=({args.t0},{args.t1}) "
              f"n_samples={args.n_samples} ics={len(ics)}")
        print(f"keep-fracs={fracs}  max_cloud={args.max_cloud}  weights={weights}")
        print("NO persistence floor is applied; h1_max_pers is raw.\n")

        rows = []
        for i, th in enumerate(thetas):
            b1, b2 = param_path(float(th), args.eps)
            trajs = integrate_full(rhs, b1, b2, ics, (args.t0, args.t1),
                                   args.n_samples, args.rtol, args.atol,
                                   args.escape_radius)
            row = {"idx": i, "theta": float(th), "b1": b1, "b2": b2,
                   "n_bounded": len(trajs)}
            parts = []
            for f in fracs:
                st = h1_stats(trajs, f, args.n_samples, args.eps, weights,
                              args.max_cloud, args.subsample_seed, args.min_cloud)
                tag = f"keep{f:g}"
                if st is None:
                    row[f"{tag}_h1_max_pers"] = ""
                    row[f"{tag}_diameter"] = ""
                    row[f"{tag}_n_cloud"] = 0
                    parts.append(f"{tag}=--")
                else:
                    row[f"{tag}_h1_max_pers"] = st["h1_max_pers"]
                    row[f"{tag}_diameter"] = st["diameter"]
                    row[f"{tag}_n_cloud"] = st["n_cloud"]
                    parts.append(f"{tag}={st['h1_max_pers']:.5f}")
            rows.append(row)
            print(f"  [{i:>3}/{len(thetas)}] theta={th:.5f} bounded={len(trajs):>3}  "
                  + "  ".join(parts))

        fields = ["idx", "theta", "b1", "b2", "n_bounded"]
        for f in fracs:
            tag = f"keep{f:g}"
            fields += [f"{tag}_h1_max_pers", f"{tag}_diameter", f"{tag}_n_cloud"]
        with args.out.open("w", newline="") as fh:
            fh.write("# BT Hopf transient-truncation test\n")
            fh.write(f"# timestamp_utc: {stamp}\n")
            for k, v in sorted(vars(args).items()):
                fh.write(f"# {k}: {v}\n")
            fh.write("# note: h1_max_pers is the RAW longest finite H1 bar; "
                     "no persistence floor applied\n")
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out}")
        log.note("csv", str(args.out))
        log.close("ok")
        return 0
    except Exception as exc:
        log.close("error", exc)
        raise
    finally:
        print(f"Run log: {log.path}")


if __name__ == "__main__":
    raise SystemExit(main())
