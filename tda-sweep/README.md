# tda-sweep

Topological data analysis pipeline for 2D dynamical systems: sweep a parameter
circle around a codimension-2 bifurcation point, build attractor point clouds
from grids of trajectories, and track Betti numbers (via ripser) along the
sweep.

The mathematical content lives in stubs and is written by the project owner —
`normal_forms.py` (all functions) and `betti_at_threshold` in `sweep.py` raise
`NotImplementedError` until filled in. Everything else is runnable as-is.

## Setup

```sh
cd ~/projects/tda-sweep
python3 -m venv venv            # already created
venv/bin/pip install -r requirements.txt
```

## 1. Verify the TDA stack (run this first)

```sh
venv/bin/python verify_pipeline.py
```

Samples 300 points on a noisy unit circle (sigma = 0.05), prints the H0/H1
barcodes, saves `figures/circle_check.png`, and asserts exactly one H1 bar
with persistence > 0.5. If this fails, nothing downstream is trustworthy.

## 2. Run a sweep

Requires the stubs in `normal_forms.py` and `betti_at_threshold` in
`sweep.py` to be implemented first.

```sh
venv/bin/python sweep.py --system bt --eps 0.1 --n-theta 64 --out results.csv
```

Key options (see `venv/bin/python sweep.py --help` for all):

- `--system {bt,cusp,bautin}` — which normal form (required)
- `--eps` — path scale for `param_path` and `blowup` (default 0.1)
- `--n-theta` — theta samples in [0, 2pi) (default 64)
- `--box XMIN XMAX YMIN YMAX` / `--grid-n` — initial-condition grid
  (default 5x5 over [-1,1]^2)
- `--t0 --t1 --n-samples --rtol --atol` — LSODA integration controls
  (stiff-aware; RK45 is deliberately not used)
- `--keep-frac` — fraction of each trajectory kept from the end (default 0.3,
  i.e. the first 70% is discarded as transient)
- `--divergence-bound` — state norm treated as divergence (default 1e3);
  thetas where every trajectory diverges are recorded as undefined, not b=0
- `--max-cloud` / `--seed` — subsampling cap and its fixed random seed
- `--eps-scale` — passed to `betti_at_threshold`

Output CSV has every parameter in `#` header lines, then columns
`theta,b0,b1,defined,n_cloud,n_diverged`. Undefined thetas have empty
`b0`/`b1` fields and `defined=0`.

## 3. Plot the sweep

```sh
venv/bin/python plot_sweep.py results.csv
```

Plots b0 and b1 vs theta (step style, distinguishable line styles/markers),
shades undefined angular sectors in hatched gray, and saves
`figures/results_betti.png` (override with `--out`).

## Logs

Every run of every script writes its full parameter set to a timestamped file
in `logs/` (e.g. `logs/sweep_20260831-101500.log`).

## Layout

```
normal_forms.py     vector fields, param_path, blowup       [stubs — owner-written]
sweep.py            theta sweep pipeline; betti_at_threshold [stub inside]
verify_pipeline.py  ripser sanity check                      [runnable now]
plot_sweep.py       CSV -> Betti-vs-theta figure             [runnable now]
runlog.py           shared timestamped run logging
figures/  logs/     outputs
```
