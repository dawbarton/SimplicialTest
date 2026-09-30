import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def fig_path(name):
    return os.path.join(FIGDIR, name)


def project_to_manifold(F, x, iters=20, h=1e-7):
    """Minimum-norm Gauss--Newton projection onto F = 0 (validation only; uses FD Jacobian)."""
    x = np.array(x, dtype=float)
    for _ in range(iters):
        r = np.atleast_1d(F(x))
        J = np.array([(np.atleast_1d(F(x + h * e)) - r) / h for e in np.eye(len(x))]).T
        dx = -np.linalg.pinv(J) @ r
        x += dx
        if np.linalg.norm(dx) < 1e-14:
            break
    return x


def render_mesh(ax, P, T, color=(0.55, 0.7, 0.85), light=(0.4, -0.5, 0.8), edges=0.0,
                two_sided=True):
    """Draw a triangle mesh on a 3D axis with simple Lambertian shading."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    tri = P[T]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-300
    L = np.asarray(light, dtype=float)
    L /= np.linalg.norm(L)
    d = nrm @ L
    d = np.abs(d) if two_sided else np.clip(d, 0, None)
    shade = 0.35 + 0.65 * d
    fc = np.clip(np.outer(shade, color), 0, 1)
    pc = Poly3DCollection(tri, facecolors=fc, edgecolor="k" if edges else "none",
                          linewidths=edges)
    ax.add_collection3d(pc)
    lo, hi = P.min(axis=0), P.max(axis=0)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    return pc
