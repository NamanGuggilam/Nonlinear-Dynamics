# TDA on 2D normal forms

A topological-data-analysis pipeline for planar dynamical systems. It integrates a
grid of trajectories, unions the post-transient points into a point cloud, applies
an anisotropic blowup, and reads Betti numbers off the persistence diagram — swept
around a parameter-space circle.

Everything runs from the command line. Every run writes its full parameter set to a
timestamped JSON file in `logs/`.

## Setup

```bash
cd ~/projects/tda-normal-forms
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
```

### ripser

`ripser` installed here from a **prebuilt wheel** (`ripser-0.6.15-cp313-cp313-macosx_11_0_arm64.whl`)
on macOS arm64 / CPython 3.13.5 — no compiler was invoked, so there was no build to
diagnose. Nothing was substituted for it.

If pip *does* fall back to a source build on your platform and it fails, the real
error is almost always one of these, and it should be fixed rather than worked
around:

* `error: command 'clang' failed` / missing `stdio.h` — no C++ toolchain.
  Fix: `xcode-select --install`.
* `fatal error: 'numpy/arrayobject.h' file not found` — numpy headers missing at
  build time. Fix: install numpy first, then `pip install --no-build-isolation ripser`.
* `Cython.Compiler.Errors.CompileError` — a Cython/Python version mismatch.
  Fix: pin `Cython==3.3.0` (already in `requirements.txt`) before building.

To see the actual compiler output rather than a truncated summary:

```bash
./venv/bin/python -m pip install --no-cache-dir --no-binary :all: -v ripser
```

## 1. Verify the pipeline (run this first)

```bash
./venv/bin/python verify_pipeline.py
```

Samples 300 points on a unit circle with Gaussian noise `sigma=0.05`, runs ripser,
prints the H0 and H1 barcodes, writes `figures/circle_check.png`, and asserts that
**exactly one** H1 bar has persistence > 0.5. Exit code 1 if the assertion fails.

Result on this machine:

```
H0: 300 bars — one infinite, all others dead by 0.1307
H1: 30 bars — longest: birth=0.1327 death=1.5666 persistence=1.4339
              next longest persistence = 0.0308
H1 bars with persistence > 0.5: 1 (expected 1)   → PASS
```

Options: `--n-points --sigma --seed --persistence-threshold --expected-h1 --maxdim
--max-print --figure --dpi`.

## 2. Sweep

```bash
# Bogdanov–Takens, 36 angles (~45 s)
./venv/bin/python sweep.py --system bt --n-theta 36

# explicit output path, longer integration, finer grid
./venv/bin/python sweep.py --system bt --n-theta 72 --nx 16 --ny 16 \
    --t1 500 --keep-frac 0.2 --out results/bt_fine.csv

# Bautin, initial-condition box given in raw phase space (~5 min)
./venv/bin/python sweep.py --system bautin --n-theta 36 \
    --box-coords raw --box -0.5 0.5 -0.5 0.5 --t1 200

# cusp normal form
./venv/bin/python sweep.py --system cusp --n-theta 36

# swap the Betti thresholding rule
./venv/bin/python sweep.py --system bt --betti-rule median --eps-scale 2.0
./venv/bin/python sweep.py --system bt --betti-rule gap --gap-max-candidates 12
```

Writes `results/sweep_<system>_<timestamp>.csv` (override with `--out`) with every
parameter in `#`-prefixed header lines.

### Sweeping a sub-interval

`--theta-min` / `--theta-max` restrict the sweep so a narrow feature can be
resolved without paying for the whole circle:

```bash
./venv/bin/python sweep.py --system bt --theta-min 4.50 --theta-max 4.95 \
    --n-theta 80 --betti-rule gap --out results/bt_zoom.csv
./venv/bin/python plot_sweep.py results/bt_zoom.csv --out figures/bt_zoom.png
```

The endpoint is **included** for a sub-interval and **excluded** for a full 2pi
sweep, where `theta_max` would duplicate `theta_min`. `plot_sweep.py` switches
from pi-fraction ticks to plain radians automatically when the range is narrow.

### Backward time

`--backward-time` (off by default) integrates the same IC grid with `t` reversed
and unions the backward-asymptotic points into the same cloud:

```bash
./venv/bin/python sweep.py --system bautin --backward-time \
    --box-coords raw --box -0.5 0.5 -0.5 0.5 --t1 200
```

An unstable invariant set is an attractor in reverse time, so this is how repelling
structure — the inner cycle of a Bautin pair, a separatrix — gets into the cloud at
all. Forward integration can never see it.

Backward escapes are counted separately (`n_bounded_bwd`) and **deliberately do not
feed the UNDEFINED test**: trajectories running away from a stable attractor in
reverse time is expected behaviour, not evidence that no attractor exists. The
bounded/divergent verdict stays a forward-time statement.

### Blowup weights

`blowup(points, eps, weights=(p, q))` computes `u = x/eps^p`, `v = y/eps^q`. The
default is `(2, 3)`, so `blowup(points, eps)` is unchanged — the BT weights are
fixed by the `param_path` scaling and were not touched.

`--blowup-weights` defaults to `auto`, which picks per system from
`normal_forms.BLOWUP_WEIGHTS`:

| System | Weights | Why |
|---|---|---|
| `bt`, `cusp` | `(2, 3)` | the Bogdanov–Takens scaling `param_path` encodes |
| `bautin` | `(1, 1)` | isotropic: `rdot = r*(b1 + b2*r^2 - r^4)` puts cycles at `r^2 ~ b2 ~ eps^2`, i.e. `r ~ eps` in *both* coordinates |

Override with e.g. `--blowup-weights 2,3`.

### Initial-condition box

`--box` defaults to **blowup coordinates** (`--box-coords blown`), mapped back with
`x = u*eps^2`, `y = v*eps^3`. This matters: the interesting dynamics sit at
`x ~ eps^2`, `y ~ eps^3`, so a fixed raw box is either far too large or far too small
as `eps` changes. Use `--box-coords raw` to give the box in phase space instead.

### Betti thresholding rules

Both are selectable at the command line via `--betti-rule`:

| Rule | Flag | Threshold |
|---|---|---|
| (a) fixed multiple of the median death time | `--betti-rule median` | `eps_scale * median(finite death times)` |
| (b) longest-bar gap heuristic | `--betti-rule gap` | midpoint of the largest drop among the top `--gap-max-candidates` persistences |

`gap` is the default. Both are implemented in
`betti_at_threshold(diagrams, eps_scale, rule=...)` in `sweep.py` and are swappable
without touching code. Why the median rule misbehaves is diagnosed below.

### Why the median rule reports b0 in the hundreds

```bash
./venv/bin/python diagnose_threshold.py --system bt --theta 4.60 \
    --hist figures/h0_deaths_bt.png
```

`diagnose_threshold.py` dumps the H0 death-time distribution — the quantity the
median rule scales — at a single angle. At BT `theta=4.60`, 1200 points
(`figures/h0_deaths_bt.png`):

```
H0 death-time percentiles (births are all 0, so death = persistence)
  p 0     0.000188
  p25     0.004898
  p50     0.008252      <- median
  p75     0.014200      <- 2*median = 0.016504 sits just here, ~p79
  p90     0.029866
  p95     0.045730
  p100    0.100655
median rule threshold = 2.0 * 0.008252 = 0.016504
bars above it = 248  ->  b0 = 249
```

**The distribution is smooth and unimodal — there is no gap in it.** That is the
whole problem, and it is not what I assumed before measuring: I expected a bimodal
split between within-trajectory sample spacing and between-trajectory separation.
There is no such split. The retained points fill a 2D annular region roughly
uniformly, so the H0 deaths are just nearest-neighbour distances of a near-uniform
sample: one smooth right-skewed peak spanning ~500x from min to max.

A threshold at a *fixed multiple of the median* therefore lands **inside the bulk**
of that peak — at the 79th percentile here — and mechanically keeps whatever
fraction of bars sits above that quantile. So `b0` is a fixed quantile of the point
count, not a topological quantity. It scales with the cloud size:

| `--max-cloud` | median death | median-rule `b0` | as % of N | gap-rule `b0` |
|---|---|---|---|---|
| 300  | 0.02178 | 54  | 18.0% | 2 |
| 600  | 0.01346 | 116 | 19.3% | 2 |
| 1200 | 0.00825 | 249 | 20.8% | 2 |
| 2400 | 0.00560 | 495 | 20.6% | 2 |

`b0` stays pinned near 20% of N while the gap rule returns 2 throughout. Doubling
the sample doubles the answer, which no Betti number should do.

This is a property of rule (a) on connected clouds, not a bug: reaching `b0 = 2`
here would need `--eps-scale ≈ 11.3` (the 99.9th percentile), and that value is
cloud-specific, so no fixed multiple works across angles. The median rule is
sound only where the death distribution is genuinely bimodal — well-separated
clusters. Use `gap` for attractors that are connected.

### The persistence floor

`--min-persistence-frac` (default `0.1`) applies an absolute floor **on top of**
whichever `--betti-rule` is selected, in **both H0 and H1**: a bar counts only if
its persistence also exceeds that fraction of the point cloud's diameter. The
effective threshold is `max(rule threshold, frac * diameter)`, so the floor can
only ever remove bars, never add them. Set `0` to disable. The infinite H0 bar has
infinite persistence and clears any finite floor, so `b0` never drops below 1.

Both the rule threshold and the floor are recorded per angle
(`rule_thr_h0`, `rule_thr_h1`, `thr_h0`, `thr_h1`, `pers_floor`).

**This fixed the b0 problem that the previous round could not explain.** On BT over
[4.40, 4.98]:

| `--min-persistence-frac` | `b0` range | mean | std | `b0 == 1` |
|---|---|---|---|---|
| 0.05 | 1–10 | 2.73 | 1.94 | 20% |
| 0.10 | 1–1 | 1.00 | 0.00 | **100%** |
| 0.20 | 1–1 | 1.00 | 0.00 | **100%** |

At the default 0.1, `b0 = 1` at all 93 defined angles — the correct answer for a
single limit cycle or single equilibrium. The spurious components were all
low-persistence bars, which a diameter-relative floor removes cleanly. Neither
isotropic blowup weights nor longer integration did this.

### UNDEFINED vs. zero

An angle is recorded as `status=UNDEFINED` with `b0`/`b1` written as the literal
string `UNDEFINED` when there is **no bounded attractor to measure** — this is not
the same fact as an attractor with no holes, and the two are never collapsed.

Triggers, all configurable:

* fewer than `--min-bounded-traj` (default 3) trajectories stay inside
  `--escape-radius` (a terminal `solve_ivp` event);
* bounded fraction below `--min-bounded-frac` (default 0, i.e. off);
* fewer than `--min-cloud` points survive subsampling;
* non-finite coordinates after the blowup.

### Point attractors are not UNDEFINED

If every trajectory collapses onto a single point, the blown-up cloud is a ball of
solver roundoff and ripser will read topology out of it — at the cusp Hopf angle a
cloud of **diameter 1.8e-8** was producing a spurious `b1 = 1`. When the cloud
diameter falls below `--degenerate-tol` (default `1e-6`, in blowup coordinates) the
sweep reports `b0 = 1`, `b1 = 0`, `status=OK`, and records the diameter in the
`diameter` column and the trigger in `reason`.

This is deliberately **OK, not UNDEFINED**: a point attractor is a bounded thing
with no holes, which is a measured answer. UNDEFINED is reserved for "there is no
bounded attractor to measure at all."

The bounded-trajectory test is an **absolute count, not a majority vote** — near a
saddle-node the basin can be a few percent of the box while the attractor is
perfectly real. The `reason` column records which trigger fired.

## 3. Plot

```bash
./venv/bin/python plot_sweep.py results/sweep_bt_demo.csv --out figures/sweep_bt.png
./venv/bin/python plot_sweep.py results/sweep_bautin_demo.csv
```

Plots `b0` and `b1` against `theta` on one shared axis (both are integer counts —
no second y-scale), shades UNDEFINED sectors with a hatched band, and breaks the
lines across them rather than drawing zero.

`--with-persistence` adds a lower panel plotting the raw `h1_max_pers` column
against theta with the persistence floor overlaid — a separate panel rather than a
second y-axis, since a persistence and a Betti count are different units.

Options: `--out --dpi --figsize --title --with-persistence`.

## Results on this machine

**Bogdanov–Takens** (`figures/sweep_bt.png`) — bounded attractor only for
`theta` in roughly `[4.54, 4.89]`, 33 of 36 sectors UNDEFINED, and `b1 = 1`
throughout the bounded wedge: the limit cycle born at the Hopf point.

This matches the analysis. Equilibria satisfy `x^2 + b2*x + b1 = 0` with
`J = [[0,1],[b2+2x, -x]]`, so `trace = -x` and `det = ∓sqrt(b2^2-4b1)`. A stable
equilibrium needs `b2 < 0`, `b1 > 0` and `b2^2 >= 4*b1`, which on the path
`b1 = eps^4 cos(theta)`, `b2 = eps^2 sin(theta)` reduces to `cos(theta) <= sqrt(5)-2`
in the fourth quadrant — i.e. `theta` in `[3*pi/2, 4.951]`. The measured window sits
just below it, where the Hopf has already produced the cycle. The H1 bar shortens
monotonically as the Hopf point is approached (persistence 1.08 → 0.78 → 0.51 →
0.035), which is the cycle shrinking to a point.

**Cusp** — `b0 = 2` inside the bistable wedge and `b0 = 1` outside it, with
`b1 = 0` at every angle. That is the cusp catastrophe read off topologically: the
two-stable-equilibrium region of the unfolding shows up as a second connected
component. Four of eight sampled angles trip the point-attractor guard.

**Bautin** (`figures/sweep_bautin.png`) — 0 sectors UNDEFINED and `b1 = 1` at every
angle, correct for `rdot = r*(b1 + b2*r^2 - r^4)`: forward-time integration finds
the outer *stable* cycle everywhere on this path. Where two cycles coexist the inner
one is repelling and is invisible to forward integration — recovering it needs
backward-time trajectories, which this pipeline does not collect.

### Backward time: recovering the second cycle

`--backward-time` on Bautin (isotropic weights), zoomed to `[1.55, 1.85]` with 24
angles — `figures/bautin_bwd_zoom.png`:

```bash
./venv/bin/python sweep.py --system bautin --backward-time \
    --box-coords raw --box -0.5 0.5 -0.5 0.5 --t1 200 \
    --theta-min 1.55 --theta-max 1.85 --n-theta 24 \
    --out results/bautin_bwd_zoom.csv
```

**Yes — `b1` reaches 2.** Forward-only it is 1 at every angle on the whole circle;
with backward time it hits 2 on a contiguous block `theta` in `[1.7065, 1.7848]`
(7 consecutive angles), and drops back to 1 at 1.7978.

That block sits inside the analytically predicted two-cycle window. From
`rdot = r*(b1 + b2*r^2 - r^4)`, two positive roots of `r^4 - b2*r^2 - b1 = 0`
require `b2 > 0`, `b1 < 0` and `b2^2 + 4*b1 > 0`, which on the path reduces to

```
theta in (pi/2, arccos(2 - sqrt(5)))  =  (1.5708, 1.8091)
```

Detection is **partial within the window** (7 of 18 angles), and that is expected
rather than a miss: the inner cycle radius is `r^2 = [b2 - sqrt(b2^2+4*b1)]/2`,
which shrinks to zero at the lower edge `theta = pi/2`. Near that edge the inner
cycle is far below the sampling resolution, so `b1 = 1`. It becomes resolvable
only in the upper part of the window, before the two cycles merge at 1.8091 —
which is exactly where the detected block sits. Two further `b1 = 2` readings at
`theta` = 1.5500 and 1.5630 fall just *below* `pi/2`, within two grid steps of the
edge; treat those as boundary effects.

#### The sparse-backward-cloud guard

The first unguarded run reported `b1 = 2` at three angles far outside the window
and `b1 = 5` at one of them. All four had only **4 of 144** bounded backward
trajectories: a backward cloud of four thin arcs, whose loops against the forward
attractor correspond to no invariant set. `--min-bounded-bwd` (default 8) drops a
backward sample below that floor and records it in `reason`. With the guard the
spurious readings vanish and the true one survives:

```
b1 values, unguarded: [1, 2, 5]   b1 >= 2 at 4/36 angles (1 in-window, 3 not)
b1 values, guarded:   [1, 2]      b1 >= 2 at 1/36 angles (1 in-window, 0 not)
```

### Blowup weights: what b0 does

Bautin, 36 angles, forward only, everything else identical:

| Weights | `b0` range | mean | median | std | `b0 == 2` | `b1` |
|---|---|---|---|---|---|---|
| `(2, 3)` anisotropic | 2–11 | 4.69 | 4 | 2.61 | 17% | 1 everywhere |
| `(1, 1)` isotropic | 2–9 | 3.81 | 3 | 2.08 | 39% | 1 everywhere |

Isotropic weights **measurably improve `b0` but do not fix it**: the spread narrows
(std 2.61 → 2.08) and the fraction reading the plausible value 2 more than doubles
(17% → 39%), but the range still reaches 9. `b1` is unaffected — 1 at every angle
under both, as it should be for forward-only integration.

So the anisotropy was one real contributor, and removing it recovers part of the
signal. It is not the whole story.

### The b1 transition angle is set by the floor, not by the bifurcation

```bash
for F in 0.05 0.10 0.20; do
  ./venv/bin/python sweep.py --system bt --theta-min 4.40 --theta-max 4.98 \
      --n-theta 120 --betti-rule gap --min-persistence-frac $F \
      --out results/bt_frac$F.csv
done
./venv/bin/python plot_sweep.py results/bt_frac0.20.csv --with-persistence \
    --out figures/bt_frac020.png
```

BT on `[4.40, 4.98]`, 120 angles, gap rule — `figures/bt_frac010.png`,
`figures/bt_frac020.png` (`--with-persistence` adds a lower panel plotting the raw
`h1_max_pers` column against theta, with the floor overlaid):

| `--min-persistence-frac` | `b1` drops 1 → 0 at | status → UNDEFINED at |
|---|---|---|
| 0.05 | never | 4.93613 |
| 0.10 | never | 4.93613 |
| 0.20 | between 4.73143 and **4.73630** | 4.93613 |

**The UNDEFINED onset does not move** — 4.93613 under all three. That is correct by
construction: divergence is decided by bounded-trajectory counts before any
diagram is computed, so a thresholding knob cannot touch it.

**The b1 transition moves drastically, and it is not a bifurcation.** The reason is
visible in the `h1_max_pers` column: as the cycle collapses toward the Hopf point,
the longest H1 bar and the cloud diameter shrink *together*, so their ratio is
scale-invariant and stays in 0.106–0.425 across the whole window.

```
theta     h1_max_pers   diameter   ratio
4.4829      1.06640     2.83583    0.3760
4.6291      0.61869     1.69128    0.3658
4.7168      0.22210     0.94614    0.2347
4.8045      0.05551     0.42310    0.1312
4.9215      0.00393     0.03521    0.1115     <- min over the window: 0.1064
```

A floor set as a *fraction of the diameter* is therefore comparing a scale-free
quantity against a scale-free threshold. It does not detect "the cycle got small";
it detects "the cycle is thin relative to the cloud". So:

* at 0.05 and 0.10 the floor never overtakes the bar and `b1 = 1` throughout;
* at 0.20 it crosses at 4.7363, cleanly (52 angles at 1, then 41 at 0, no flicker).

Two consequences worth knowing before trusting a transition angle:

1. **The default 0.1 sits on a knife edge here.** The minimum ratio over the window
   is 0.1064 — only 6% above the default floor. A slightly different cloud would
   flip `b1` to 0 over part of the range.
2. **No choice of frac recovers the Hopf.** The analytic Hopf is at
   `theta = 3*pi/2 = 4.71239` (where `b1 = eps^4*cos(theta)` changes sign). The
   frac=0.20 transition lands at 4.7363, 0.024 rad late, and the raw persistence
   curve decays *smoothly* through the Hopf with no kink. That is because past the
   Hopf the equilibrium is stable but the retained transient spiral still traces an
   annulus, which reads as a loop. Matching the Hopf would need frac ≈ 0.235, which
   is tuning the knob to the known answer rather than measuring it.

Treat the transition angle as a property of the threshold, not of the system.

## Is the Hopf detectable with attractor-converged sampling?

```bash
./venv/bin/python hopf_transient_test.py \
    --theta-min 4.60 --theta-max 4.85 --n-theta 60 \
    --t1 1200 --n-samples 6000 --keep-fracs 0.30 0.10 0.03 0.01 \
    --max-cloud 400 --out results/hopf_transient.csv
./venv/bin/python plot_hopf_test.py results/hopf_transient.csv
```

`hopf_transient_test.py` integrates the IC grid **once per theta** at 4x the
default `t_span` and slices several transient truncations from the *same*
trajectories, so the keep-fraction is the only thing that varies between columns
and the integration is not repeated. `--n-samples 6000` is chosen so even the 1%
tail exceeds `--max-cloud`: every column is subsampled to exactly 400 points at
every angle, so cloud size cannot confound the comparison. **No persistence floor
is applied anywhere in this script**; `h1_max_pers` is raw.

Figure: `figures/hopf_transient.png`.

### Answer: no kink. The curve decays smoothly, and truncation does not matter.

| theta | keep 30% | keep 10% | keep 3% | keep 1% |
|---|---|---|---|---|
| 4.60000 | 0.97614 | 1.01811 | 1.02628 | 1.00362 |
| 4.66356 | 0.67840 | 0.68326 | 0.68529 | 0.68348 |
| 4.70169 | 0.30330 | 0.29362 | 0.30435 | 0.30855 |
| **4.71441** (Hopf) | **0.18688** | **0.17101** | **0.17155** | **0.18016** |
| 4.72712 | 0.10133 | 0.09522 | 0.09513 | 0.09349 |
| 4.75254 | 0.02107 | 0.02143 | 0.02085 | 0.02129 |
| 4.80339 | 0.00084 | 0.00080 | 0.00083 | 0.00079 |
| 4.84153 | 0.00013 | 0.00006 | 0.00006 | 0.00005 |

The four truncations agree to a median 6% relative spread; `max|keep30% - keep1%|`
is 0.041, and the largest *relative* disagreements are out at theta > 4.82 where
all values are ~1e-4. Truncating the transient 30x harder changes nothing.

No localized feature sits at the Hopf. Steepest descent and maximum |2nd
derivative| land 0.015–0.10 rad away from 4.71239 and **move around
inconsistently between truncations** — the signature of no kink, not of a kink
being missed. Past the Hopf `log(pers)` is linear in theta with R^2 = 0.997–0.998:
a clean exponential decay, not a vanishing.

### Why harder truncation cannot help: critical slowing down

The residual is not leftover transient in the sampling sense — it is the
equilibrium's own convergence rate going to zero at the bifurcation. For the BT
stable focus, `Re(lambda) = -x_-/2`, and `x_- -> 0` at the Hopf. Fitting the
measured persistence against it:

```
log(pers) = 0.9979 * (Re(lambda) * t_end) + const     R^2 = 0.9993
```

A coefficient of 1.00 means the residual loop is exactly `exp(Re(lambda)*t_end)` —
the spiral has simply not finished converging by the end of the run. The required
convergence time diverges at the bifurcation:

| theta | Re(lambda) | 1/|Re(lambda)| | vs t_end = 1200 |
|---|---|---|---|
| 4.85 | -0.00749 | 133 | converged |
| 4.75 | -0.00168 | 595 | marginal |
| 4.72 | -0.00028 | 3530 | **not converged** |
| 4.71239 (Hopf) | -2.0e-07 | infinite | never |

So no truncation *and no feasible integration time* removes the residual near the
Hopf. Keeping the last 1% of a longer run samples the same weakly-converged
spiral, which is exactly what the table shows.

### The Hopf is still recoverable — as a scaling law, not a kink

A supercritical Hopf has amplitude `~ sqrt(theta_H - theta)`, so `pers^2` should be
linear below the bifurcation and cross zero at it. It is (R^2 = 0.992–0.997), and
extrapolating the zero recovers the Hopf.

**On not tuning to the answer:** an earlier version of this fit used the Hopf angle
as the right edge of the fit window, which makes the extrapolation circular. It is
now bounded by round numbers instead. Across seven windows chosen without any
reference to the Hopf:

| fit window | mean zero | error vs 4.71239 |
|---|---|---|
| [4.60, 4.66] | 4.71942 | +0.00703 |
| [4.60, 4.68] | 4.71652 | +0.00413 |
| [4.60, 4.69] | 4.71354 | +0.00115 |
| [4.60, 4.70] | 4.71251 | +0.00012 |
| [4.62, 4.70] | 4.71140 | -0.00099 |
| [4.64, 4.70] | 4.70912 | -0.00327 |
| [4.65, 4.69] | 4.70839 | -0.00400 |
| data-driven (20–80% of max, no Hopf used) | 4.71346 | +0.00107 |

The zero lands within **±0.007 rad** of the analytic Hopf regardless of window —
about 0.15% of the angle — with the spread set by the fit window, not by the
truncation. So the bifurcation is present in the persistence data; it is just
encoded in the *shape* of the decay rather than as a discontinuity. Quoting a
tighter precision than ~±0.005 rad would be reading in the answer.

### Known limitation: b0 on Bautin

`b0` remains noisy under isotropic weights. I tested the two explanations I gave
in the previous round and **both are refuted by measurement** — recording that
rather than leaving the earlier claims standing:

1. **"Residual transients"** — no. Lengthening integration and shrinking the
   retained tail barely moves `b0`, and makes `b1` *worse* by starving the cloud:

   | Setting | `b0` values | mean | std | `b1` values |
   |---|---|---|---|---|
   | `--t1 200 --keep-frac 0.30` | 2,2,2,2,2,3,3,3,4,6,6,8 | 3.58 | 1.93 | [1] |
   | `--t1 600 --keep-frac 0.10` | 2,2,3,3,3,3,3,3,3,4,5,7 | 3.42 | 1.32 | [0, 1] |
   | `--t1 1200 --keep-frac 0.05` | 2,2,2,2,3,3,3,3,3,3,4,11 | 3.42 | 2.36 | [0, 1, 2] |

2. **"True `b0` tracks origin stability"** (`b0 = 2` where `cos(theta) < 0`, since
   the origin is then stable and coexists with the outer cycle) — no. Exact match
   rate 14%, and `b0 = 1` never occurs where the origin is unstable.

What remains is that the gap heuristic systematically finds 1–2 more components
than the phase portrait has, at angles with no obvious common feature. Diagnosing
it properly means looking at the H0 barcode angle by angle with
`diagnose_threshold.py`, which I have not done. `b1` is unaffected throughout.

**Update:** `--min-persistence-frac 0.1` resolves this on BT (`b0 = 1` at 100% of
defined angles, see above). The Bautin runs in the table were made before the floor
existed and have not been re-run with it; the floor is likely to help there too.

## Run logs

Every entry point writes `logs/<script>_<UTC timestamp>.json` containing the resolved
argparse namespace, `sys.argv`, Python and platform strings, versions of numpy/scipy/
matplotlib/ripser/persim, and the run's status and results. The file is written
*before* the work starts, so parameters survive a crash, and rewritten with results
on completion.

## Layout

```
normal_forms.py     bt_rhs, cusp_rhs, bautin_rhs, param_path, blowup
verify_pipeline.py  the noisy-circle sanity gate
sweep.py            integration, betti_at_threshold, the theta sweep, CSV output
plot_sweep.py       CSV -> figure
diagnose_threshold.py  death-time distribution at one theta (threshold forensics)
hopf_transient_test.py transient-truncation sweep of the raw H1 persistence curve
plot_hopf_test.py      figure for the above
codensity_test.py   k-th-nearest-neighbour codensity diagnostic (read-only)
runlog.py           timestamped run-parameter logging (shared)
requirements.txt    pinned
figures/  logs/  results/
tda-sweep/          the earlier stub scaffold, kept for history
```

`cusp_rhs` is the planar suspension `xdot = b1 + b2*x - x^3`, `ydot = -y`. The cusp
normal form is intrinsically one-dimensional; the contracting transverse direction
lets the same planar machinery integrate and sample it.

## Codensity diagnostic

```bash
./venv/bin/python codensity_test.py
```

`codensity_test.py` is a read-only diagnostic: it imports `sweep.py`'s own
`initial_conditions`, `integrate_grid`, and `subsample`, so the cloud at each angle is
built by the same code as the sweep (same box, same blowup, same `t_span`, same seed).
It changes nothing in the pipeline and tunes nothing.

`codensity(points, k=12)` is the distance from each point to its k-th nearest neighbour.
For BT at theta = 4.65, 4.712, 4.76 (1200-point clouds, blowup coordinates):

| theta | min | median | p90 | max | max/med | diameter |
|---|---|---|---|---|---|---|
| 4.65000 | 0.014676 | 0.032477 | 0.068273 | 0.116724 | 3.59 | 1.52229 |
| 4.71200 | 0.016482 | 0.028683 | 0.045300 | 0.060134 | 2.10 | 0.99027 |
| 4.76000 | 0.011951 | 0.019865 | 0.027006 | 0.042365 | 2.13 | 0.67936 |

The clouds are mildly cored rather than uniformly sparse, and 4.65 is the most cored
(spread 3.59 vs 2.10-2.13).

Keeping only points with codensity <= s, s swept over the 10th-90th percentile of each
angle's own codensity, longest H1 persistence raw (no floor). Surviving counts are
120, 240, ... 1080 at every angle, so cloud size does not confound the columns.

| theta | p10 | p20 | p30 | p40 | p50 | p60 | p70 | p80 | p90 |
|---|---|---|---|---|---|---|---|---|---|
| 4.65000 | 0.32348 | 0.50152 | 0.51907 | 0.66833 | 0.73681 | 0.74646 | 0.74582 | 0.77026 | 0.54344 |
| 4.71200 | 0.38038 | 0.42521 | 0.35238 | 0.30004 | 0.26251 | 0.25549 | 0.25393 | 0.24541 | 0.24017 |
| 4.76000 | 0.12294 | 0.14135 | 0.08566 | 0.08423 | 0.07699 | 0.09498 | 0.10798 | 0.10703 | 0.10905 |

Unfiltered baseline on the full cloud: 0.51426 (4.65), 0.23887 (4.712), 0.11034 (4.76).

Observations, not conclusions: at 4.65 filtering to the dense half *raises* the longest
bar above baseline (0.737 at p50, 0.770 at p80 vs 0.514), so the sparse points are
shortening the loop there; at 4.712 and 4.76 filtering leaves it at or below baseline,
and the 4.712 row decays monotonically from p20 on. The 4.712 > 4.76 ordering holds at
every s. Full verbatim run in `results/codensity_bt_results.txt`, per-row data in
`results/codensity_bt.csv`.

## `tda-sweep/` — the earlier scaffold

`tda-sweep/` is the first pass at this project (Aug 2026), kept for history. It has the
same shape — `sweep.py`, `normal_forms.py`, `verify_pipeline.py`, `runlog.py` — but the
mathematical content is deliberately left as stubs: every function in its
`normal_forms.py` and `betti_at_threshold` in its `sweep.py` raise `NotImplementedError`.
The working pipeline is the one at the top level of this repo, which supersedes it. Its
own `tda-sweep/README.md` documents it as it stood.

