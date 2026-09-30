"""Appendix A, helix example (Fig. A1): F: R^3 -> R^2, k = 1, grain 0.01, arclength 25."""

import numpy as np
import matplotlib.pyplot as plt

from common import fig_path, project_to_manifold
from plcont import PLContinuation, follow_path

a, b = 0.65, 1.35


def helix(y):
    return np.array([y[0] * np.sin(y[2]) - y[1] * np.cos(y[2]),
                     y[0] ** 2 / a ** 2 + y[1] ** 2 / b ** 2 - 1.0])


v0 = np.array([0.459619, 0.954594, 1.122073])
print("|F(v0)| =", np.linalg.norm(helix(v0)))

# user function: distance from the starting point (the paper's nu >= 0 feature)
F = lambda y: (helix(y), [np.linalg.norm(y - v0)])
pl = PLContinuation(F, v0, grain=0.01, k=1)
# The paper does not say which way path_start_a goes; Fig. A1 shows the curve descending in z.
path = follow_path(pl, direction=[0, 0, -1], max_arclength=25.0)
x = path.x
print(f"points {len(x)}, simplices {len(path.simplices)}, F evaluations {pl.n_evals}, "
      f"arclength {path.arclength[-1]:.3f}, z from {x[0, 2]:.3f} to {x[-1, 2]:.3f}")

# accuracy: distance to the exact curve, and interpolated user function vs exact value
err = np.array([np.linalg.norm(p - project_to_manifold(helix, p)) for p in x[::10]])
uerr = np.abs(path.u[:, 0] - np.linalg.norm(x - v0, axis=1))
print(f"distance to exact helix: max {err.max():.2e}, median {np.median(err):.2e} "
      f"(grain 0.01, grain^2 = 1e-4)")
print(f"user-function interpolation error: max {uerr.max():.2e}")

fig = plt.figure(figsize=(9, 4.2))
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.plot(*x.T, lw=1)
ax.scatter(*v0, color="k", s=10)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("PL continuation of the helix (cf. Fig. A1)")
ax2 = fig.add_subplot(1, 2, 2)
ax2.semilogy(path.arclength[::10], err, ".", ms=3, label="distance to exact curve")
ax2.set_xlabel("arclength"); ax2.set_ylabel("error"); ax2.legend()
ax2.set_title("accuracy along the path")
fig.tight_layout()
fig.savefig(fig_path("helix.png"), dpi=150)
np.savetxt(fig_path("helix.txt"), x, fmt="%f")
