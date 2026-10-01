"""Normal-form vector fields and coordinate maps for the TDA sweep pipeline.

All functions here are mathematical content and are intentionally left
unimplemented — they are written by the project owner. Each docstring states
the contract that sweep.py relies on.
"""

import numpy as np


def bt_rhs(t, state, b1, b2):
    """Bogdanov-Takens normal-form vector field.

    Contract:
        t     : float, time (present for solve_ivp compatibility; the field
                is autonomous, so t is unused).
        state : array-like of shape (2,), the point (x, y).
        b1, b2: floats, unfolding parameters.

    Returns:
        array-like of shape (2,): (dx/dt, dy/dt). Must be finite for finite
        input; sweep.py treats non-finite output as trajectory divergence.
    """
    raise NotImplementedError("bt_rhs: written by the project owner")


def cusp_rhs(t, state, b1, b2):
    """Cusp normal-form vector field.

    Contract:
        Same signature and return contract as bt_rhs: takes (t, state, b1, b2)
        with state of shape (2,), returns (dx/dt, dy/dt) of shape (2,).
    """
    raise NotImplementedError("cusp_rhs: written by the project owner")


def bautin_rhs(t, state, b1, b2):
    """Bautin (generalized Hopf) normal-form vector field.

    Contract:
        Same signature and return contract as bt_rhs: takes (t, state, b1, b2)
        with state of shape (2,), returns (dx/dt, dy/dt) of shape (2,).
    """
    raise NotImplementedError("bautin_rhs: written by the project owner")


def param_path(theta, eps):
    """Map an angle on the parameter circle to unfolding parameters.

    Contract:
        theta : float in [0, 2*pi), angle along the sweep path.
        eps   : float > 0, path scale; the scaling is anisotropic (b1 and b2
                scale with different powers of eps).

    Returns:
        (b1, b2) : tuple of two floats, passed directly to the *_rhs functions.
    """
    raise NotImplementedError("param_path: written by the project owner")


def blowup(points, eps):
    """Coordinate rescaling (blow-up) applied to the point cloud before ripser.

    Contract:
        points : ndarray of shape (n, 2), the union point cloud in the
                 original (x, y) phase-space coordinates.
        eps    : float > 0, the same path scale used in param_path.

    Returns:
        ndarray of shape (n, 2): the rescaled cloud. Must not change n and
        must be finite for finite input.
    """
    raise NotImplementedError("blowup: written by the project owner")
