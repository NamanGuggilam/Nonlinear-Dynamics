#!/usr/bin/env python3
"""Planar normal forms, the anisotropic parameter path, and the blowup map.

Every right-hand side has the scipy.integrate.solve_ivp signature
f(t, state, b1, b2) with state = (x, y), so they are interchangeable.
"""

import numpy as np

__all__ = ["bt_rhs", "cusp_rhs", "bautin_rhs", "param_path", "blowup",
           "RHS", "BLOWUP_WEIGHTS"]


def bt_rhs(t, state, b1, b2):
    """Bogdanov-Takens normal form.

        xdot = y
        ydot = b1 + b2*x + x^2 - x*y

    Codimension-2 unfolding of a double-zero eigenvalue.  The b1/b2 scaling
    in param_path() is chosen for this system's Newton polygon.
    """
    x, y = state
    return [y, b1 + b2 * x + x * x - x * y]


def cusp_rhs(t, state, b1, b2):
    """Cusp normal form, planar suspension.

        xdot = b1 + b2*x - x^3
        ydot = -y

    The cusp is a codimension-2 bifurcation of *equilibria*; its normal form
    is one-dimensional (xdot = b1 + b2*x - x^3).  It is suspended here with a
    transverse contracting direction so it can be integrated and sampled by
    the same planar machinery as the other two forms.  The x-dynamics are
    exactly the standard cusp unfolding; y decays to the invariant line y=0.
    """
    x, y = state
    return [b1 + b2 * x - x ** 3, -y]


def bautin_rhs(t, state, b1, b2):
    """Bautin (generalized Hopf) normal form, Cartesian coordinates.

        xdot = b1*x - y + b2*x*(x^2+y^2) - x*(x^2+y^2)^2
        ydot = x + b1*y + b2*y*(x^2+y^2) - y*(x^2+y^2)^2

    In polar coordinates this is rdot = r*(b1 + b2*r^2 - r^4), thetadot = 1,
    the codimension-2 unfolding where the first Lyapunov coefficient vanishes.
    Up to two concentric limit cycles coexist, which is what makes H1 counts
    interesting along the parameter path.
    """
    x, y = state
    r2 = x * x + y * y
    return [
        b1 * x - y + b2 * x * r2 - x * r2 * r2,
        x + b1 * y + b2 * y * r2 - y * r2 * r2,
    ]


#: Name -> right-hand side, for argparse dispatch.
RHS = {"bt": bt_rhs, "cusp": cusp_rhs, "bautin": bautin_rhs}


def param_path(theta, eps):
    """Anisotropic circle in parameter space.

        b1 = eps^4 * cos(theta)
        b2 = eps^2 * sin(theta)

    The eps^4 / eps^2 weighting is deliberate: it is the Bogdanov-Takens
    scaling that makes the path cross every sheet of the bifurcation set at
    a nondegenerate angle.  DO NOT change the exponents -- an isotropic
    circle sweeps the wrong region and the Betti sequence loses meaning.
    """
    theta = np.asarray(theta, dtype=float)
    b1 = eps ** 4 * np.cos(theta)
    b2 = eps ** 2 * np.sin(theta)
    if b1.ndim == 0:
        return float(b1), float(b2)
    return b1, b2


#: Per-system blowup exponents (p, q) for u = x/eps^p, v = y/eps^q.
#:
#: bt/cusp keep the Bogdanov-Takens weights (2, 3) -- these are fixed by the
#: BT scaling that param_path encodes and must not be changed.
#:
#: bautin is isotropic (1, 1): its normal form in polar coordinates is
#: rdot = r*(b1 + b2*r^2 - r^4), whose limit cycles sit at r^2 ~ b2 ~ eps^2,
#: i.e. r ~ eps in BOTH coordinates.  Applying the (2, 3) weights to it
#: stretches y by a further factor 1/eps and turns a round cycle into a thin
#: ellipse, which is what wrecks the H0 structure.
BLOWUP_WEIGHTS = {"bt": (2, 3), "cusp": (2, 3), "bautin": (1, 1)}


def blowup(points, eps, weights=(2, 3)):
    """Blowup (rescaling) map.

        u = x / eps^p
        v = y / eps^q        with (p, q) = weights

    Undoes the collapse of the phase portrait toward the origin as eps -> 0,
    so persistent features stay at O(1) scale and are comparable across eps.

    The default (2, 3) is the Bogdanov-Takens scaling matched to param_path's
    anisotropy and is the required behaviour for the BT system -- calling
    blowup(points, eps) with no weights is exactly u = x/eps^2, v = y/eps^3.
    Other systems have other natural scalings; see BLOWUP_WEIGHTS.

    points: (N, 2) array of (x, y).  Returns a new (N, 2) array of (u, v).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) point cloud, got shape {pts.shape}")
    p_exp, q_exp = weights
    return np.column_stack([pts[:, 0] / eps ** p_exp, pts[:, 1] / eps ** q_exp])
