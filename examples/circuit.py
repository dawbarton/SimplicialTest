"""The circuit examples of Section 2 (Figs. 3 and 5), with a simulated circuit.

codim1: unknowns (Vdrv, V1, V2), residual (I1, I2), user function Idrv (Fig. 3).
codim2: unknowns (Vdrv, V1, V2, alpha), residual (I1, I2), user function Idrv (Fig. 5).
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve

from common import fig_path
from circuit_model import R43K, R0, R_of_alpha, currents
from plcont import PLContinuation, SurfaceMesh, follow_path

VMAX = 8.0   # the paper sweeps [0, 7] V; the upper fold of this model is at 7.25 V


def residual(x, R=R43K, noise=None):
    i1, i2, idrv = currents(*x[:3], R=R)
    r, u = np.array([i1, i2]) * 1e3, np.array([idrv * 1e3])       # mA
    if noise is not None:
        r = r + noise[1] * noise[0].standard_normal(2)
        u = u + noise[1] * noise[0].standard_normal(1)
    return r, u


def start_point(vdrv=0.5, R=R43K):
    v = fsolve(lambda y: residual([vdrv, *y], R)[0], [vdrv, 0.05], xtol=1e-13)
    return np.array([vdrv, *v])


def pl_curve(R=R43K, grain=(0.02, 0.01, 0.005), noise=None):
    """Trace the I-V curve both ways from Vdrv = 0.5 with codim1-style path following."""
    pl = PLContinuation(lambda x: residual(x, R, noise), start_point(R=R), grain=grain, k=1,
                        inside=lambda x: 0.0 <= x[0] <= VMAX)
    up = follow_path(pl, direction=[1, 0, 0])
    down = follow_path(pl, direction=[-1, 0, 0])
    X = np.vstack([down.x[::-1], up.x[1:]])
    I = np.concatenate([down.u[::-1, 0], up.u[1:, 0]])
    return X, I, pl.n_evals


def pc_curve(R=R43K, ds=0.02, vmax=VMAX):
    """Reference: pseudo-arclength predictor-corrector continuation with a FD Jacobian."""
    f = lambda x: residual(x, R)[0]

    def jac(x, h=1e-8):
        r = f(x)
        return np.array([(f(x + h * e) - r) / h for e in np.eye(3)]).T

    def correct(x, t, h):
        y = x + h * t
        for _ in range(10):
            dy = np.linalg.solve(np.vstack([jac(y), t]),
                                 -np.concatenate([f(y), [t @ (y - x) - h]]))
            y += dy
            if np.linalg.norm(dy) < 1e-11:
                return y
        return None

    x = start_point(R=R)
    t = np.linalg.svd(jac(x))[2][-1]
    t *= np.sign(t[0])
    xs, h = [x], ds
    while 0.0 <= x[0] <= vmax and len(xs) < 100000:
        y = correct(x, t, h)
        if y is None or np.linalg.norm(y - x - h * t) > 0.2 * h:
            h /= 2                      # reject: corrector failed or moved too far
            if h < 1e-8:
                raise RuntimeError("step size too small")
            continue
        tn = np.linalg.svd(jac(y))[2][-1]
        t = tn * np.sign(tn @ (y - x))
        x, h = y, min(ds, 1.5 * h)
        xs.append(x)
    X = np.array(xs)
    return X, np.array([residual(x, R)[1][0] for x in X])


def folds(X, I):
    """(V, I) at the upper (max Vdrv on the lower branch) and lower (min Vdrv) folds."""
    lo = I < 1.8
    hi = (I > 1.5) & (I < 4.0)
    j, k = np.argmax(np.where(lo, X[:, 0], -np.inf)), np.argmin(np.where(hi, X[:, 0], np.inf))
    return [(X[j, 0], I[j]), (X[k, 0], I[k])]


def n_turning_points(X):
    dv = np.diff(X[:, 0])
    return int(np.sum(np.sign(dv[1:]) != np.sign(dv[:-1])))


# --- Fig. 3 --------------------------------------------------------------------------
t0 = time.time()
Xpl, Ipl, nev = pl_curve()
print(f"PL codim1 (R = 43k): {len(Xpl)} points, {nev} circuit evaluations, "
      f"{time.time() - t0:.1f}s; folds (V, mA): {np.round(folds(Xpl, Ipl), 3).tolist()}")
Xpc, Ipc = pc_curve()
print(f"PC reference: {len(Xpc)} points, {n_turning_points(Xpc)} turning points, "
      f"folds (V, mA): {np.round(folds(Xpc, Ipc), 3).tolist()}")
rng = np.random.default_rng(0)
Xn, In, nevn = pl_curve(noise=(rng, 2e-3))
print(f"PL with 2 uA measurement noise: {len(Xn)} points, {nevn} evaluations; "
      f"folds: {np.round(folds(Xn, In), 3).tolist()}")
sel = Xpl[:, 0] >= 0.5          # the PC reference starts at Vdrv = 0.5 and only goes up
d = np.array([np.min(np.hypot(Xpc[:, 0] - v, Ipc - i))
              for v, i in zip(Xpl[sel][::5, 0], Ipl[sel][::5])])
print(f"PL curve distance to PC curve in the (V, mA) plane: max {d.max():.1e}, median "
      f"{np.median(d):.1e} (PC points spaced up to {np.max(np.hypot(*np.diff(np.c_[Xpc[:, 0], Ipc], axis=0).T)):.1e})")

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axs[0]
ax.plot(Xpc[:, 0], Ipc, color="C2", lw=2.5, alpha=0.6, label="predictor-corrector")
ax.plot(Xpl[:, 0], Ipl, color="C4", lw=1, label="PL continuation (grain 0.02/0.01/0.005 V)")
ax.set_xlim(0, VMAX); ax.set_ylim(0, 5)
ax.set_xlabel("$V_{drv}$ (V)"); ax.set_ylabel("$I_{drv}$ (mA)")
ax.set_title("Simulated I-V curve, R = 43 k$\\Omega$ (cf. Fig. 3)"); ax.legend(fontsize=8)
ax = axs[1]
ax.plot(Xpc[:, 0], Ipc, color="C2", lw=2.5, alpha=0.6, label="noise-free")
ax.plot(Xn[:, 0], In, color="C3", lw=0.8, label="PL, 2 $\\mu$A noise on $I_1, I_2, I_{drv}$")
ax.set_xlim(0, VMAX); ax.set_ylim(0, 5)
ax.set_xlabel("$V_{drv}$ (V)"); ax.set_title("Emulated hardware continuation")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(fig_path("circuit_iv.png"), dpi=150)

# --- where does the hysteresis appear? ------------------------------------------------
def has_fold(R):
    X, I = pc_curve(R=R, ds=0.02, vmax=4.0)
    return n_turning_points(X) > 0


Rc = brentq(lambda R: 0.5 - has_fold(R), 3e3, 5e3, xtol=5.0)
alpha_c = np.log(Rc / R0) / np.log(R43K / R0)
print(f"hysteresis appears for R > {Rc:.0f} Ohm (alpha > {alpha_c:.3f} with the assumed "
      f"R(alpha) = 1k * 43^alpha); the paper says about 5 kOhm and alpha > 0.2")

# --- Fig. 5: codim2 --------------------------------------------------------------------
def residual2(x):
    return residual(x[:3], R=R_of_alpha(x[3]))


x0 = np.append(start_point(vdrv=0.5, R=R_of_alpha(0.5)), 0.5)
pl2 = PLContinuation(residual2, x0, grain=(0.1, 0.05, 0.01, 0.05), k=2,
                     inside=lambda x: 0.0 <= x[0] <= VMAX and 0.0 <= x[3] <= 1.0)
t0 = time.time()
cells = pl2.run()
mesh = SurfaceMesh(pl2, cells)
print(f"PL codim2: {len(cells)} simplices, {pl2.n_evals} circuit evaluations, "
      f"{time.time() - t0:.0f}s; polygon sizes "
      f"{np.bincount([len(p) for p in mesh.polygons]).tolist()}, degenerate {mesh.degenerate}, "
      f"components {mesh.components()}, Euler characteristic {mesh.euler_characteristic()}")

for L in sorted(mesh.boundary_loops(), key=len, reverse=True):
    B = mesh.points[L]
    print(f"  boundary loop of {len(L)} vertices: Vdrv in [{B[:, 0].min():.2f}, "
          f"{B[:, 0].max():.2f}], alpha in [{B[:, 3].min():.3f}, {B[:, 3].max():.3f}], "
          f"|V1 - V2| >= {np.abs(B[:, 1] - B[:, 2]).min():.1e}")
P, U, T = mesh.points, mesh.user[:, 0], mesh.triangles
Q = np.column_stack([P[:, 0], P[:, 3], np.log10(np.clip(U, 1e-6, None) * 1e-3)])
from common import render_mesh
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection="3d")
keep = np.all(Q[T][:, :, 2] > -4.5, axis=1)       # hide Idrv < 30 uA (Vdrv near 0)
render_mesh(ax, Q, T[keep], color=(0.25, 0.35, 0.95), light=(-0.3, -0.6, 0.7))
ax.set_zlim(-4.5, Q[:, 2].max())
ax.set_box_aspect((1.3, 1, 1)); ax.view_init(elev=22, azim=-120)
ax.set_xlabel("$V_{drv}$ (V)"); ax.set_ylabel("$\\alpha$"); ax.set_zlabel("$\\log_{10} I_{drv}$ (A)")
ax.set_title(f"Response surface (cf. Fig. 5): {len(cells)} simplices")
ax = fig.add_subplot(1, 2, 2)
# cross-sections of the surface at fixed alpha (slice the triangles)
for a, col in zip((0.2, 0.35, 0.5, 0.75, 0.95), plt.cm.viridis(np.linspace(0, 0.9, 5))):
    d = P[:, 3] - a
    segs = []
    for tri in T:
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if d[tri[i]] * d[tri[j]] < 0:
                w = d[tri[i]] / (d[tri[i]] - d[tri[j]])
                pts.append([(1 - w) * P[tri[i], 0] + w * P[tri[j], 0],
                            (1 - w) * U[tri[i]] + w * U[tri[j]]])
        if len(pts) == 2:
            segs.append(pts)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors=[col], linewidths=1.2))
    ax.plot([], [], color=col, label=f"$\\alpha$ = {a:.2f}, R = {R_of_alpha(a) / 1e3:.1f} k$\\Omega$")
ax.set_xlim(0, VMAX); ax.set_ylim(0, 5); ax.legend(fontsize=8)
ax.set_xlabel("$V_{drv}$ (V)"); ax.set_ylabel("$I_{drv}$ (mA)")
ax.set_title("Slices of the codim2 surface at fixed $\\alpha$")
fig.tight_layout(); fig.savefig(fig_path("circuit_surface.png"), dpi=150)
