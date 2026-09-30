"""Coxeter--Freudenthal--Kuhn triangulation of R^n (Henderson & Melville, Sections 5, 7 and 12).

A path simplex is identified by an integer cube index ``I`` (the lower-left corner of a
unit cube of the lattice) and a permutation ``perm`` of ``0..n-1``. Its vertices are

    v_0 = I,    v_{i+1} = v_i + e_{perm[i]}        (equation 7)

so the simplex is a monotone walk from the lower-left to the upper-right corner of the cube.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True, slots=True)
class Simplex:
    """A path simplex (I, Pi) of the Kuhn triangulation, with integer lattice vertices."""

    I: tuple[int, ...]
    perm: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.I)

    def vertices(self) -> list[tuple[int, ...]]:
        """The n + 1 integer vertices v_0, ..., v_n (equation 7)."""
        v = list(self.I)
        out = [tuple(v)]
        for p in self.perm:
            v[p] += 1
            out.append(tuple(v))
        return out

    def pivot(self, i: int) -> "Simplex":
        """The neighbouring simplex across the (n-1)-face opposite vertex i (equations 9--11)."""
        n = self.n
        p = self.perm
        if 0 < i < n:
            q = list(p)
            q[i - 1], q[i] = q[i], q[i - 1]
            return Simplex(self.I, tuple(q))
        if i == 0:
            I = list(self.I)
            I[p[0]] += 1
            return Simplex(tuple(I), p[1:] + p[:1])
        if i == n:
            I = list(self.I)
            I[p[-1]] -= 1
            return Simplex(tuple(I), p[-1:] + p[:-1])
        raise ValueError(f"vertex index {i} out of range for n = {n}")


def locate(r: np.ndarray) -> Simplex:
    """The path simplex containing the point ``r`` given in lattice coordinates (Section 12).

    The integer part gives the cube; sorting the fractional parts in decreasing order gives
    the permutation (the largest remainder is the first step of the walk).
    """
    r = np.asarray(r, dtype=float)
    I = np.floor(r)
    frac = r - I
    perm = tuple(int(j) for j in np.argsort(-frac, kind="stable"))
    return Simplex(tuple(int(i) for i in I), perm)


def barycentre_offset(n: int) -> np.ndarray:
    """Lattice coordinates of the barycentre of the path simplex (0, identity)."""
    return np.array([(n - j) / (n + 1) for j in range(n)])


def face_combinations(n: int, k: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """All (n-k)-faces of an n-simplex as pairs (F_i, V_i) (Section 9).

    F_i lists the n-k+1 vertex indices on the face and V_i the k indices not on it; the
    neighbours sharing the face are the pivots across the facets opposite the vertices in V_i.
    """
    allv = range(n + 1)
    out = []
    for Fi in combinations(allv, n - k + 1):
        Vi = tuple(j for j in allv if j not in Fi)
        out.append((Fi, Vi))
    return out


class Lattice:
    """Affine map between integer lattice coordinates and R^n: x = origin + basis @ i (eq. 29)."""

    def __init__(self, origin, basis):
        self.origin = np.asarray(origin, dtype=float)
        self.basis = np.atleast_2d(np.asarray(basis, dtype=float))
        self._inv = np.linalg.inv(self.basis)

    @classmethod
    def around(cls, x0, grain) -> "Lattice":
        """Lattice with the point x0 at the barycentre of simplex (0, identity) (Section 12).

        ``grain`` is a scalar (cubic lattice) or a vector of per-coordinate spacings.
        """
        x0 = np.asarray(x0, dtype=float)
        h = np.broadcast_to(np.asarray(grain, dtype=float), x0.shape)
        basis = np.diag(h)
        return cls(x0 - basis @ barycentre_offset(len(x0)), basis)

    def to_x(self, i) -> np.ndarray:
        return self.origin + self.basis @ np.asarray(i, dtype=float)

    def to_lattice(self, x) -> np.ndarray:
        return self._inv @ (np.asarray(x, dtype=float) - self.origin)

    def locate(self, x) -> Simplex:
        return locate(self.to_lattice(x))
