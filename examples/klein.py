"""Appendix A, Klein bottle immersion (Figs. A4, A5): F: R^3 -> R (equation A1), k = 2."""

import numpy as np
import matplotlib.pyplot as plt

from common import fig_path, render_mesh
from plcont import PLContinuation, SurfaceMesh


def klein(p):
    x, y, z = p
    s = x * x + y * y + z * z
    return [(s + 2 * y - 1) * ((s - 2 * y - 1) ** 2 - 8 * z * z) + 16 * x * z * (s - 2 * y - 1)]


def grad(p, h=1e-6):
    return np.array([(klein(p + h * e)[0] - klein(p - h * e)[0]) / (2 * h) for e in np.eye(3)])


# The paper does not give the grain; N * grain^2 is constant (about 943 in this code's
# convention), so 19,576 simplices corresponds to grain ~ 0.22.
runs = {}
for g in (0.22, 0.1, 0.05):
    pl = PLContinuation(klein, [1.0, 0.0, 0.0], grain=g, k=2)
    cells = pl.run()
    mesh = SurfaceMesh(pl, cells)
    chi = mesh.euler_characteristic()
    print(f"grain {g}: {len(cells)} simplices, F evaluations {pl.n_evals}, polygons "
          f"{len(mesh.polygons)} (degenerate {mesh.degenerate}), boundary edges "
          f"{mesh.boundary_edges()}, components {mesh.components()}, Euler characteristic "
          f"{chi} (orientable genus {1 - chi // 2})")
    runs[g] = (pl, cells, mesh)

# Fig. A4
pl, cells, mesh = runs[0.22]
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(projection="3d")
render_mesh(ax, mesh.points, mesh.triangles, color=(0.2, 0.6, 0.9), edges=0.0)
ax.view_init(elev=20, azim=-75); ax.set_axis_off()
ax.set_title(f"Klein bottle (eq. A1), grain 0.22: {len(cells)} simplices (paper: 19,576)")
fig.subplots_adjust(0, 0, 1, 0.95); fig.savefig(fig_path("klein.png"), dpi=150)

# Fig. A5: the double (self-intersection) curve of equation A1 is the circle
# x^2 + (y - 1)^2 = 2, z = 0 (there s = 1 + 2y, so both terms of A1 and grad F vanish).
# Cut the surface with planes normal to that circle: the exact zero set is an X, the PL
# surfaces (which are embedded) resolve each crossing into two folded sheets.
def slice_segments(mesh, c, e1, e2):
    """Intersection of a triangle mesh with the plane through c spanned by e1, e2."""
    nrm = np.cross(e1, e2)
    P = mesh.points[mesh.triangles]
    d = (P - c) @ nrm
    segs = []
    for tri, dd in zip(P, d):
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if dd[a] * dd[b] < 0:
                q = tri[a] + dd[a] / (dd[a] - dd[b]) * (tri[b] - tri[a])
                pts.append([(q - c) @ e1, (q - c) @ e2])
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs)


from matplotlib.collections import LineCollection

box = 0.5
fig, axs = plt.subplots(1, 4, figsize=(14, 3.9))
for ax, phi in zip(axs, np.deg2rad([0, 90, 180, 270])):
    c = np.array([np.sqrt(2) * np.cos(phi), 1 + np.sqrt(2) * np.sin(phi), 0.0])
    e1, e2 = np.array([np.cos(phi), np.sin(phi), 0.0]), np.array([0.0, 0.0, 1.0])
    u = np.linspace(-box, box, 401)
    U, V = np.meshgrid(u, u)
    Fv = klein(c[:, None, None] + e1[:, None, None] * U + e2[:, None, None] * V)[0]
    ax.contour(U, V, Fv, levels=[0], colors="0.6", linewidths=3)
    for g, col in ((0.22, "C3"), (0.1, "C0")):
        segs = slice_segments(runs[g][2], c, e1, e2)
        ax.add_collection(LineCollection(segs, colors=col, linewidths=1.2, label=f"PL, grain {g}"))
    ax.set_xlim(-box, box); ax.set_ylim(-box, box); ax.set_aspect("equal")
    ax.set_title(f"slice at ({c[0]:.2f}, {c[1]:.2f}, 0)")
    ax.set_xlabel("radial offset"); ax.set_ylabel("z")
axs[0].plot([], [], color="0.6", lw=3, label="exact (F = 0)")
axs[0].legend(loc="lower left", fontsize=7)
fig.suptitle("Cross-sections through the double curve (cf. Fig. A5): the PL surface cannot cross itself")
fig.tight_layout(); fig.savefig(fig_path("klein_crossing.png"), dpi=150)
