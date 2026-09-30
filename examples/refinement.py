"""Sections 15-16: refining PL zeros with the derivative-free approximate Newton iterations.

Torus of Appendix A (n = 3, k = 2): error of the PL zeros and of their refinements against
the exact distance, as a function of the grain, and the conditioning of the face systems.
"""

import numpy as np
import matplotlib.pyplot as plt

from common import fig_path
from plcont import PLContinuation, face_conditioning, refine_on_face, refine_orthogonal

r, R = 0.5, 2.0
torus = lambda y: [(R - np.hypot(y[0], y[1])) ** 2 + y[2] ** 2 - r ** 2]


def dist(P):
    P = np.atleast_2d(P)
    return np.abs(np.hypot(np.hypot(P[:, 0], P[:, 1]) - R, P[:, 2]) - r)


rng = np.random.default_rng(0)
grains = [0.2, 0.1, 0.05, 0.025]
rows = []
for g in grains:
    pl = PLContinuation(torus, [2.5, 0, 0], grain=g, k=2)
    cells = pl.run(max_steps=4000)          # a patch of the torus is enough
    sample = [(c, z) for c in cells for z in c.zeros]
    sample = [sample[i] for i in rng.choice(len(sample), min(1500, len(sample)), replace=False)]
    e0, ef, eo, cf, co = [], [], [], [], []
    for c, z in sample:
        xf = refine_on_face(pl, z, iters=3)
        xo = refine_orthogonal(pl, c, z.x, iters=3)
        e0.append(dist(z.x)[0])
        ef.append(dist(np.array(xf[1:])))
        eo.append(dist(np.array(xo[1:])))
        a, b = face_conditioning(pl, c, z)
        cf.append(a); co.append(b)
    e0, ef, eo, cf, co = map(np.array, (e0, ef, eo, cf, co))
    rows.append((g, e0, ef, eo, cf, co))
    print(f"grain {g}: PL zero error median {np.median(e0):.1e}, max {e0.max():.1e} | "
          f"face Newton (Sec. 15) 1/2/3 steps median "
          f"{'/'.join(f'{v:.1e}' for v in np.median(ef, axis=0))}, max 1 step {ef[:, 0].max():.1e} | "
          f"orthogonal (Sec. 16) median {'/'.join(f'{v:.1e}' for v in np.median(eo, axis=0))}, "
          f"max 1 step {eo[:, 0].max():.1e}")
    print(f"           face amplification median {np.median(cf):.2f}, 99th pct "
          f"{np.percentile(cf, 99):.1e}, max {cf.max():.1e}; orthogonal = {np.max(co):.2f}")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axs[0]
G = np.array(grains)
for j, (lab, key) in enumerate((("PL zero", 1), ("face Newton, 1 step", 2), ("orthogonal Newton, 1 step", 3))):
    med = [np.median(row[key] if key == 1 else row[key][:, 0]) for row in rows]
    mx = [np.max(row[key] if key == 1 else row[key][:, 0]) for row in rows]
    ax.loglog(G, med, "o-", color=f"C{j}", label=f"{lab} (median)")
    ax.loglog(G, mx, "s--", color=f"C{j}", alpha=0.5, label=f"{lab} (max)")
ax.loglog(G, 0.3 * G ** 2, "k:", label="$h^2$")
ax.loglog(G, 0.3 * G ** 3, "k-.", label="$h^3$")
ax.set_xlabel("grain h"); ax.set_ylabel("distance to torus"); ax.legend(fontsize=7)
ax.set_title("Accuracy of PL zeros and one refinement step")
ax = axs[1]
g, e0, ef, eo, cf, co = rows[2]
ax.loglog(cf, ef[:, 0] / e0, ".", ms=2, label="face Newton (Sec. 15)")
ax.loglog(cf, eo[:, 0] / e0, ".", ms=2, label="orthogonal Newton (Sec. 16)")
ax.set_xlabel("face amplification $1/|\\cos\\theta|$ (edge vs normal)")
ax.set_ylabel("error after one step / PL error")
ax.set_title(f"Effect of face conditioning (grain {g})"); ax.legend(fontsize=8, markerscale=4)
fig.tight_layout(); fig.savefig(fig_path("refinement.png"), dpi=150)
