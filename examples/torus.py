"""Appendix A, torus example (Figs. A2, A3): F: R^3 -> R, k = 2, grain 0.05, start (2.5,0,0)."""

import time

import numpy as np
import matplotlib.pyplot as plt
from common import fig_path, render_mesh
from plcont import PLContinuation, SurfaceMesh

r, R = 0.5, 2.0


def torus(y):
    return [(R - np.sqrt(y[0] ** 2 + y[1] ** 2)) ** 2 + y[2] ** 2 - r ** 2]


def distance(P):
    rho = np.hypot(P[:, 0], P[:, 1])
    return np.abs(np.hypot(rho - R, P[:, 2]) - r)


results = {}
for order in ("fifo", "lifo"):
    pl = PLContinuation(torus, [2.5, 0.0, 0.0], grain=0.05, k=2, order=order)
    t0 = time.time()
    cells = pl.run()
    results[order] = (pl, cells)
    nz = np.bincount([len(c.zeros) for c in cells])
    print(f"[{order}] simplices {len(cells)}, F evaluations {pl.n_evals}, face solves "
          f"{pl.n_solves}, polygon sizes {dict((i, int(v)) for i, v in enumerate(nz) if v)}, "
          f"max frontier {max(h[1] for h in pl.history)}, {time.time() - t0:.1f}s")

pl, cells = results["fifo"]
mesh = SurfaceMesh(pl, cells)
d = distance(mesh.points)
print(f"mesh: {len(mesh.points)} vertices, {len(mesh.polygons)} polygons, "
      f"{len(mesh.triangles)} triangles, degenerate {mesh.degenerate}, "
      f"boundary edges {mesh.boundary_edges()}, components {mesh.components()}, "
      f"Euler characteristic {mesh.euler_characteristic()}")
print(f"vertex distance to torus: max {d.max():.2e}, mean {d.mean():.2e}")
mesh.write_stl(fig_path("torus.stl"))

# Fig. A2
fig = plt.figure(figsize=(8, 5.5))
ax = fig.add_subplot(projection="3d")
render_mesh(ax, mesh.points, mesh.triangles, edges=0.05)
ax.view_init(elev=40, azim=-60); ax.set_axis_off()
ax.set_title(f"Torus, grain 0.05: {len(cells)} simplices (paper: 61,808)")
fig.subplots_adjust(0, 0, 1, 0.95); fig.savefig(fig_path("torus.png"), dpi=150)

# Close-up of the outer equator, showing the triangles and quadrilaterals
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(projection="3d")
sel = [p for p in mesh.polygons if np.all(mesh.points[p][:, 0] > 2.0)
       and np.all(np.abs(mesh.points[p][:, 1]) < 0.3) and np.all(np.abs(mesh.points[p][:, 2]) < 0.3)]
T = np.array([(p[0], p[j], p[j + 1]) for p in sel for j in range(1, len(p) - 1)])
render_mesh(ax, mesh.points, T, edges=0.0)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
ax.add_collection3d(Poly3DCollection([mesh.points[p] for p in sel], facecolor="none",
                                     edgecolor="k", linewidths=0.5))
P = mesh.points[np.unique(T)]
lo, hi = P.min(axis=0), P.max(axis=0)
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
ax.set_box_aspect(hi - lo); ax.view_init(elev=0, azim=0); ax.set_axis_off()
ax.set_title("PL polygons (one per simplex) near (2.5, 0, 0)")
fig.tight_layout(); fig.savefig(fig_path("torus_zoom.png"), dpi=150)

# Fig. A3: cumulative simplices and frontier size against step
fig, ax = plt.subplots(figsize=(8, 3.5))
for order, ls in (("fifo", "-"), ("lifo", "--")):
    h = np.array(results[order][0].history)
    ax.plot(h[:, 0], h[:, 1], ls, label=f"frontier ({order.upper()})")
ax.set_xlabel("step"); ax.set_ylabel("number of boundary simplices")
ax2 = ax.twinx()
ax2.plot(h[:, 0], h[:, 0], color="C2", lw=0.8, label="enumerated simplices")
ax2.set_ylabel("number of manifold simplices")
ax.legend(loc="upper left"); ax2.legend(loc="lower right")
ax.set_title("Torus: enumerated simplices and frontier (cf. Fig. A3)")
fig.tight_layout(); fig.savefig(fig_path("torus_frontier.png"), dpi=150)
