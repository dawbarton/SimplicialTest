"""A simulated version of the negative-resistance circuit of Fig. 1 (Chua & Zhong 1985).

Reconstruction from the schematic, not the authors' simulator:

* q1 is taken as an NPN (collector P1, base P2, emitter ground). The figure draws it with a
  PNP symbol, which could not conduct with its emitter grounded; with an NPN the pair
  q1/q2 forms the thyristor-like structure whose S-shaped I-V curve Fig. 3 shows.
* q2 is a PNP (emitter through 2k87 to Vdrv, base P1, collector P2).
* R(alpha) (43k for alpha = 1) joins P1 and P2; 499 joins Vdrv and P1; 4k75 joins P2 and ground.
* Transistors use the Ebers--Moll transport model with 2N3904/2N3906-like parameters
  (typical SPICE values, not checked against the devices in the paper).

With V1 and V2 fixing the voltages of P1 and P2 (Fig. 1b), the currents I1 and I2 that the
sources must supply are explicit functions of (Vdrv, V1, V2), apart from the internal emitter
node of q2, which is solved for by a safeguarded Newton iteration (the "legacy simulator").
"""

import numpy as np

VT = 0.025852
NPN = dict(IS=6.734e-15, BF=416.4, BR=0.7371)
PNP = dict(IS=1.41e-15, BF=180.7, BR=4.977)
R499, R2K87, R4K75, R43K = 499.0, 2870.0, 4750.0, 43e3


def limexp(x):
    """exp with linear continuation above 40 (avoids overflow far from the manifold)."""
    return np.exp(x) if x < 40.0 else np.exp(40.0) * (1.0 + x - 40.0)


def ebers_moll(vbe, vbc, IS, BF, BR):
    """(collector current in, base current in) of an NPN; for a PNP pass (veb, vcb)."""
    ef, er = limexp(vbe / VT), limexp(vbc / VT)
    ic = IS * ((ef - er) - (er - 1.0) / BR)
    ib = IS * ((ef - 1.0) / BF + (er - 1.0) / BR)
    return ic, ib


def pnp_emitter_current(ve, vb, vc):
    ic, ib = ebers_moll(ve - vb, vc - vb, **PNP)
    return ic + ib


def emitter_node(vdrv, p1, p2):
    """Solve (Vdrv - E)/2k87 = I_E(q2) for the q2 emitter voltage E."""
    g = lambda e: (vdrv - e) / R2K87 - pnp_emitter_current(e, p1, p2)
    lo, hi = min(vdrv, p1) - 1.0, max(vdrv, p1) + 1.0
    while g(lo) < 0:
        lo -= 1.0
    while g(hi) > 0:
        hi += 1.0
    e = min(max(vdrv, lo), hi)
    for _ in range(100):
        ge = g(e)
        if ge > 0:
            lo = e
        else:
            hi = e
        h = 1e-7
        dg = (g(e + h) - ge) / h
        en = e - ge / dg if dg < 0 else 0.5 * (lo + hi)
        if not (lo < en < hi):
            en = 0.5 * (lo + hi)
        if abs(en - e) < 1e-13:
            return en
        e = en
    return e


def currents(vdrv, v1, v2, R=R43K):
    """(I1, I2, Idrv) in amperes for source voltages Vdrv, V1 (at P1), V2 (at P2)."""
    e2 = emitter_node(vdrv, v1, v2)
    ic1, ib1 = ebers_moll(v2, v2 - v1, **NPN)           # q1: base P2, collector P1
    ic2, ib2 = ebers_moll(e2 - v1, v2 - v1, **PNP)      # q2: emitter E2, base P1, collector P2
    # KCL at P1: I1 + (Vdrv-P1)/499 + Ib2 + (P2-P1)/R = Ic1
    i1 = ic1 - ib2 - (vdrv - v1) / R499 - (v2 - v1) / R
    # KCL at P2: I2 + Ic2 + (P1-P2)/R = Ib1 + P2/4k75
    i2 = ib1 + v2 / R4K75 - ic2 - (v1 - v2) / R
    idrv = (vdrv - v1) / R499 + (vdrv - e2) / R2K87
    return i1, i2, idrv


# R(alpha): the paper gives only R(1) = 43k and "alpha in [0, 1]"; a geometric sweep from
# 1k to 43k is assumed here.
R0 = 1e3


def R_of_alpha(alpha):
    return R0 * (R43K / R0) ** alpha
