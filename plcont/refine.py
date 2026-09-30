"""Refining PL zeros with derivative-free approximate Newton iterations (Sections 14--16).

Both iterations use only the values of F already stored at the simplex vertices as a secant
(finite-difference) Jacobian, plus one new evaluation of F per iteration at the current iterate.
"""

from __future__ import annotations

import numpy as np

from .continuation import Cell, PLContinuation, Zero


def refine_on_face(pl: PLContinuation, zero: Zero, iters=3) -> list[np.ndarray]:
    """Approximate Newton restricted to the (n-k)-face (Section 15, equation 43).

    Solves sum_i F(w_i) dt_i = -F(sum_i t_i w_i), sum_i dt_i = 0. Starting from a vertex of the
    face the first step is exactly the PL face test, so this continues that iteration.
    Returns the iterates (the PL zero first).
    """
    W = np.array([pl.lattice.to_x(w) for w in zero.face])
    Fw = np.array([pl.value(w)[0] for w in zero.face])
    M = np.vstack([Fw.T, np.ones(len(W))])
    t = zero.t.copy()
    xs = [t @ W]
    for _ in range(iters):
        r = _residual(pl, xs[-1])
        dt = np.linalg.solve(M, np.concatenate([-r, [0.0]]))
        t = t + dt
        xs.append(t @ W)
    return xs


def secant_jacobian(pl: PLContinuation, cell: Cell) -> np.ndarray:
    """J with J (v_j - v_0) = F(v_j) - F(v_0) for the simplex vertices (equation 41)."""
    X = np.array([pl.lattice.to_x(v) for v in cell.vertices])
    Fv = np.array([pl.value(v)[0] for v in cell.vertices])
    return np.linalg.solve(X[1:] - X[0], Fv[1:] - Fv[0]).T


def refine_orthogonal(pl: PLContinuation, cell: Cell, x, iters=3) -> list[np.ndarray]:
    """Approximate Newton orthogonal to the PL manifold (Section 16).

    The step dx solves J dx = -F(x) subject to T^T dx = 0, where the columns of T span the
    tangent space of the PL manifold in this simplex (the null space of J). This is the
    minimum-norm solution dx = -J^+ F(x), a differenced pseudo-arclength constraint.
    """
    Jp = np.linalg.pinv(secant_jacobian(pl, cell))
    xs = [np.asarray(x, dtype=float)]
    for _ in range(iters):
        xs.append(xs[-1] - Jp @ _residual(pl, xs[-1]))
    return xs


def face_conditioning(pl: PLContinuation, cell: Cell, zero: Zero) -> tuple[float, float]:
    """Error amplification of the face system and of the orthogonal system (Section 16).

    With J the secant Jacobian and E the unit edge vectors w_j - w_0 of the face, the face
    system J E t = -F loses accuracy by sigma_max(J) / sigma_min(J E) relative to the best
    possible frame; for m = 1 this is 1 / |cos| of the angle between the edge and the normal
    to the zero set, and it is large when the face is nearly parallel to the zero set. The
    orthogonal system uses an orthonormal basis Q of the normal space and has
    sigma_max(J) / sigma_min(J Q) = cond(J), which is 1 for m = 1.
    """
    J = secant_jacobian(pl, cell)
    W = np.array([pl.lattice.to_x(w) for w in zero.face])
    E = (W[1:] - W[0]).T
    E = E / np.linalg.norm(E, axis=0)
    sJ = np.linalg.svd(J, compute_uv=False)
    return float(sJ[0] / np.linalg.svd(J @ E, compute_uv=False)[-1]), float(sJ[0] / sJ[-1])


def _residual(pl, x):
    out = pl.F(x)
    r = out[0] if isinstance(out, tuple) else out
    return np.atleast_1d(np.asarray(r, dtype=float))
