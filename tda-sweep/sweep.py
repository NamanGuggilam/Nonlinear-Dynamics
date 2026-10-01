"""Sweep theta over the parameter circle and record Betti numbers of the
attractor point cloud for a chosen normal form.

For each theta: integrate a grid of initial conditions (LSODA, stiff-aware),
drop the transient, union the retained points, subsample, apply blowup(), run
ripser, and read off (b0, b1) via betti_at_threshold(). Thetas where no
trajectory stays bounded are recorded as undefined (empty Betti fields in the
CSV), never as zero. All parameters go into the CSV header and a timestamped
log file.

The mathematical content (vector fields, param_path, blowup,
betti_at_threshold) is written by the project owner; this file only
orchestrates.
"""

import argparse
import csv
import datetime
import pathlib

import numpy as np
from ripser import ripser
from scipy.integrate import solve_ivp

from normal_forms import bautin_rhs, blowup, bt_rhs, cusp_rhs, param_path
from runlog import start_run_log

RHS_FUNCS = {"bt": bt_rhs, "cusp": cusp_rhs, "bautin": bautin_rhs}


def betti_at_threshold(diagrams, eps_scale):
    """Betti numbers of the Rips complex at a threshold derived from eps_scale.

    Contract (threshold rule written by the project owner):
        diagrams  : list of ndarrays as returned by ripser(...)["dgms"];
                    diagrams[k] has shape (n_k, 2) of (birth, death) pairs,
                    H0 may contain one death = inf bar.
        eps_scale : float > 0, the scale from which the filtration threshold
                    is computed.

    Returns:
        (b0, b1) : tuple of two non-negative ints.
    """
    raise NotImplementedError("betti_at_threshold: written by the project owner")


def grid_initial_conditions(box, n_per_axis):
    """Regular n_per_axis x n_per_axis grid over box = (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = box
    xs = np.linspace(xmin, xmax, n_per_axis)
    ys = np.linspace(ymin, ymax, n_per_axis)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def integrate_trajectory(rhs, ic, b1, b2, t_span, n_samples, rtol, atol, bound):
    """Integrate one trajectory with LSODA.

    Returns (points, diverged): points is an (n_samples, 2) array of states at
    uniform sample times, or None when the trajectory diverged. Divergence
    means: the norm of the state exceeded `bound` (caught early by a terminal
    event), the solver failed, or non-finite values appeared.
    """

    def escape(t, state):
        return np.linalg.norm(state) - bound

    escape.terminal = True
    escape.direction = 1

    t_eval = np.linspace(t_span[0], t_span[1], n_samples)
    try:
        sol = solve_ivp(rhs, t_span, ic, method="LSODA", t_eval=t_eval,
                        rtol=rtol, atol=atol, args=(b1, b2), events=escape)
    except (ValueError, FloatingPointError, OverflowError):
        return None, True
    if not sol.success or sol.t_events[0].size > 0:
        return None, True
    points = sol.y.T
    if points.shape[0] < n_samples or not np.all(np.isfinite(points)):
        return None, True
    return points, False


def cloud_for_theta(rhs, b1, b2, ics, args, rng):
    """Build the attractor point cloud for one parameter point.

    Returns (cloud, n_diverged): cloud is an (m, 2) ndarray of retained,
    subsampled points (before blowup), or None when every trajectory diverged.
    """
    keep_from = int(round(args.n_samples * (1.0 - args.keep_frac)))
    retained = []
    n_diverged = 0
    for ic in ics:
        points, diverged = integrate_trajectory(
            rhs, ic, b1, b2, (args.t0, args.t1), args.n_samples,
            args.rtol, args.atol, args.divergence_bound)
        if diverged:
            n_diverged += 1
        else:
            retained.append(points[keep_from:])
    if not retained:
        return None, n_diverged
    cloud = np.vstack(retained)
    if cloud.shape[0] > args.max_cloud:
        idx = rng.choice(cloud.shape[0], size=args.max_cloud, replace=False)
        cloud = cloud[idx]
    return cloud, n_diverged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(RHS_FUNCS), required=True,
                        help="which normal form to sweep")
    parser.add_argument("--eps", type=float, default=0.1,
                        help="path scale passed to param_path and blowup")
    parser.add_argument("--n-theta", type=int, default=64,
                        help="number of theta samples in [0, 2pi)")
    parser.add_argument("--box", type=float, nargs=4, default=(-1.0, 1.0, -1.0, 1.0),
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                        help="initial-condition box")
    parser.add_argument("--grid-n", type=int, default=5,
                        help="initial conditions per axis (grid-n^2 total)")
    parser.add_argument("--t0", type=float, default=0.0)
    parser.add_argument("--t1", type=float, default=200.0)
    parser.add_argument("--n-samples", type=int, default=400,
                        help="uniform sample times per trajectory")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--keep-frac", type=float, default=0.3,
                        help="fraction of each trajectory kept (from the end)")
    parser.add_argument("--divergence-bound", type=float, default=1e3,
                        help="state norm beyond which a trajectory counts as diverged")
    parser.add_argument("--max-cloud", type=int, default=1500,
                        help="max points handed to ripser per theta")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for subsampling")
    parser.add_argument("--eps-scale", type=float, default=1.0,
                        help="threshold scale passed to betti_at_threshold")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results.csv"))
    args = parser.parse_args()
    logger = start_run_log("sweep", vars(args))

    rhs = RHS_FUNCS[args.system]
    ics = grid_initial_conditions(args.box, args.grid_n)
    thetas = np.linspace(0.0, 2.0 * np.pi, args.n_theta, endpoint=False)
    rng = np.random.default_rng(args.seed)

    rows = []
    for i, theta in enumerate(thetas):
        b1, b2 = param_path(theta, args.eps)
        cloud, n_diverged = cloud_for_theta(rhs, b1, b2, ics, args, rng)
        if cloud is None:
            logger.info("theta[%d]=%.4f: all %d trajectories diverged -> undefined",
                        i, theta, len(ics))
            rows.append((theta, "", "", 0, 0, n_diverged))
            continue
        diagrams = ripser(blowup(cloud, args.eps))["dgms"]
        b0, b1_betti = betti_at_threshold(diagrams, args.eps_scale)
        logger.info("theta[%d]=%.4f: b0=%d b1=%d cloud=%d diverged=%d/%d",
                    i, theta, b0, b1_betti, cloud.shape[0], n_diverged, len(ics))
        rows.append((theta, b0, b1_betti, 1, cloud.shape[0], n_diverged))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        fh.write(f"# sweep.py run {datetime.datetime.now().isoformat()}\n")
        for key, value in sorted(vars(args).items()):
            fh.write(f"# {key} = {value}\n")
        fh.write(f"# n_initial_conditions = {len(ics)}\n")
        writer = csv.writer(fh)
        writer.writerow(["theta", "b0", "b1", "defined", "n_cloud", "n_diverged"])
        writer.writerows(rows)

    n_undefined = sum(1 for row in rows if not row[3])
    logger.info("wrote %s: %d thetas, %d undefined", args.out, len(rows), n_undefined)


if __name__ == "__main__":
    main()
