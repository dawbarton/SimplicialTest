"""Piecewise linear (simplicial) continuation of implicitly defined k-manifolds.

Implements the algorithm of Henderson & Melville, "Piecewise Linear Continuation:
Derivative-free Manifold Generation" (Sections 4 and 8--12): a queue of candidate path
simplices and a set of tested ones; each candidate has its (n-k)-faces tested for a zero of
the piecewise linear interpolant of F, and the neighbours sharing a transverse face are added
to the candidates.
"""

from __future__ import annotations

import warnings
import heapq
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .kuhn import Lattice, Simplex, face_combinations


@dataclass(slots=True)
class Zero:
    """A zero of the PL interpolant on an (n-k)-face: a vertex of the polytope in a simplex."""

    face: tuple                 # sorted integer vertices of the (n-k)-face (global key)
    omitted: frozenset          # local indices of the simplex vertices not on the face (V_i)
    t: np.ndarray               # barycentric coordinates w.r.t. ``face`` (sum to one)
    x: np.ndarray               # the point in R^n
    u: np.ndarray | None        # interpolated user functions


@dataclass(slots=True)
class Cell:
    """A transverse n-simplex and the vertices of the convex polytope (zero set) inside it."""

    simplex: Simplex
    vertices: list              # integer vertices v_0..v_n
    zeros: list = field(default_factory=list)


class PLContinuation:
    """Enumerate the simplices of a Kuhn triangulation of R^n that meet {F = 0}.

    Parameters
    ----------
    F : callable
        ``F(x)`` for ``x`` in R^n returns the residual in R^(n-k), or a pair
        ``(residual, user_values)`` where ``user_values`` are extra functions recorded at the
        same time (the paper's ``nu`` user functions) and interpolated onto the manifold.
    x0 : array_like
        A point on (or near) the manifold.
    grain : float or array_like
        Lattice spacing (scalar, or one value per coordinate). Ignored if ``lattice`` given.
    k : int
        Dimension of the manifold.
    inside : callable, optional
        ``inside(x) -> bool`` bounds a non-compact manifold (the paper's BoundaryFunction):
        only zeros ``x`` inside the region generate pivots, so the neighbours sharing a face
        whose zero is outside are not added to the candidate list. Testing zeros rather than
        simplex barycentres keeps the truncated surface free of spurious holes.
    order : {"fifo", "lifo", "sorted"}
        Queue discipline for the candidate list (Section 18). "sorted" always takes the
        lexicographically smallest simplex (I, Pi), as an ordered set would.
    zero_eps : float
        Exact zeros of F at a vertex are replaced by this small positive value (Section 11).
    cond_max : float
        Face systems with a larger condition number are treated as singular (no crossing).
    """

    def __init__(self, F, x0, grain=None, k=1, *, lattice=None, inside=None, order="fifo",
                 zero_eps=1e-13, cond_max=1e12):
        self.F = F
        self.x0 = np.asarray(x0, dtype=float)
        self.n = len(self.x0)
        self.k = k
        self.m = self.n - k
        if not (self.n > self.m >= 1):
            raise ValueError("need n > n - k >= 1")
        self.lattice = lattice if lattice is not None else Lattice.around(self.x0, grain)
        self.inside = inside
        if order not in ("fifo", "lifo", "sorted"):
            raise ValueError("order must be 'fifo', 'lifo' or 'sorted'")
        self.order = order
        self.zero_eps = zero_eps
        self.cond_max = cond_max
        self.faces_of_simplex = face_combinations(self.n, k)

        self.values: dict = {}          # integer vertex -> (F, user)
        self.face_cache: dict = {}      # sorted face vertices -> (t, x, u) or None
        self.candidates = [] if order == "sorted" else deque()
        self.seen: set = set()          # candidates and processed simplices
        self.n_processed = 0
        self.n_solves = 0
        self.history: list = []         # (step, number of candidates) after each step

    # -- function values and face tests -------------------------------------------------

    @property
    def n_evals(self) -> int:
        return len(self.values)

    def value(self, key):
        """F (and user functions) at an integer lattice vertex; evaluated once (Section 11)."""
        val = self.values.get(key)
        if val is None:
            out = self.F(self.lattice.to_x(key))
            if isinstance(out, tuple):
                r, u = out
                u = np.atleast_1d(np.asarray(u, dtype=float))
            else:
                r, u = out, None
            r = np.atleast_1d(np.array(r, dtype=float))
            if r.shape != (self.m,):
                raise ValueError(f"F must return {self.m} residuals, got shape {r.shape}")
            r[r == 0.0] = self.zero_eps
            val = (r, u)
            self.values[key] = val
        return val

    def test_face(self, fkey):
        """Zero of the PL interpolant on the face with (sorted) vertices ``fkey`` (eq. 16).

        Solves sum_{j>=1} t_j (F(w_j) - F(w_0)) = -F(w_0), sets t_0 = 1 - sum t_j and accepts
        the solution if all 0 <= t_j <= 1. Returns ``(t, x, u)`` or ``None``.
        """
        if fkey in self.face_cache:
            return self.face_cache[fkey]
        vals = [self.value(w) for w in fkey]
        f = np.array([v[0] for v in vals])                  # (m+1) x m
        A = (f[1:] - f[0]).T                                # m x m
        b = -f[0]
        self.n_solves += 1
        res = None
        if self.m == 1:
            a = A[0, 0]
            s = np.array([b[0] / a]) if a != 0.0 else None
        else:
            s = None
            if np.linalg.cond(A) < self.cond_max:
                s = np.linalg.solve(A, b)
        if s is not None:
            t = np.concatenate(([1.0 - s.sum()], s))
            if np.all(t >= 0.0) and np.all(t <= 1.0):
                W = np.array([self.lattice.to_x(w) for w in fkey])
                x = t @ W
                u = None
                if vals[0][1] is not None:
                    u = t @ np.array([v[1] for v in vals])
                res = (t, x, u)
        self.face_cache[fkey] = res
        return res

    def cell(self, s: Simplex) -> Cell:
        """Test every (n-k)-face of ``s`` and return the polytope vertices found."""
        verts = s.vertices()
        c = Cell(s, verts)
        for Fi, Vi in self.faces_of_simplex:
            fkey = tuple(sorted(verts[j] for j in Fi))
            res = self.test_face(fkey)
            if res is not None:
                t, x, u = res
                c.zeros.append(Zero(fkey, frozenset(Vi), t, x, u))
        return c

    def is_inside(self, x) -> bool:
        return self.inside is None or bool(self.inside(x))

    # -- the continuation loop (Section 4) --------------------------------------------

    def find_start(self, max_depth=6) -> Simplex:
        """A transverse simplex near x0: the one containing x0, else breadth-first search."""
        s0 = self.lattice.locate(self.x0)
        frontier, visited = [s0], {s0}
        for _ in range(max_depth + 1):
            for s in frontier:
                if self.cell(s).zeros:
                    return s
            nxt = []
            for s in frontier:
                for i in range(self.n + 1):
                    p = s.pivot(i)
                    if p not in visited:
                        visited.add(p)
                        nxt.append(p)
            frontier = nxt
        raise RuntimeError("no transverse simplex found near the starting point")

    def start(self, simplex: Simplex | None = None) -> Simplex:
        s = self.find_start() if simplex is None else simplex
        self._push(s)
        self.seen.add(s)
        return s

    def _push(self, s: Simplex):
        if self.order == "sorted":
            heapq.heappush(self.candidates, (s.I, s.perm))
        else:
            self.candidates.append(s)

    def _pop(self) -> Simplex:
        if self.order == "sorted":
            return Simplex(*heapq.heappop(self.candidates))
        return self.candidates.popleft() if self.order == "fifo" else self.candidates.pop()

    def step(self) -> Cell | None:
        """Process one candidate simplex; returns its cell, or None when the list is empty."""
        if not self.candidates:
            return None
        s = self._pop()
        c = self.cell(s)
        pivots = set()
        for z in c.zeros:
            if self.is_inside(z.x):
                pivots |= z.omitted
        for i in sorted(pivots):
            nb = s.pivot(i)
            if nb not in self.seen:
                self.seen.add(nb)
                self._push(nb)
        self.n_processed += 1
        self.history.append((self.n_processed, len(self.candidates)))
        return c

    def run(self, max_steps=None, progress=0) -> list[Cell]:
        """Run until the candidate list is empty (or ``max_steps``); returns all cells."""
        if not self.seen:
            self.start()
        cells = []
        while self.candidates and (max_steps is None or len(cells) < max_steps):
            c = self.step()
            cells.append(c)
            if progress and len(cells) % progress == 0:
                print(f"  {len(cells)} simplices, frontier {len(self.candidates)}", flush=True)
        return cells


@dataclass
class Path:
    """A polyline traced by :func:`follow_path` (k = 1)."""

    x: np.ndarray               # points on the transverse (n-1)-faces, in order
    u: np.ndarray | None        # interpolated user functions at those points
    arclength: np.ndarray       # cumulative arclength at each point
    simplices: list             # simplices traversed
    closed: bool


def follow_path(pl: PLContinuation, *, direction=None, max_arclength=np.inf, max_steps=10**7,
                start: Simplex | None = None) -> Path:
    """Door-in/door-out path following for k = 1 (Allgower--Georg; Sections 4 and 17.1).

    Starting from a transverse simplex the curve leaves through one transverse facet; the
    pivot across that facet gives the next simplex, which the curve leaves through its other
    transverse facet. ``direction`` (a vector in R^n) selects the initial direction.
    """
    if pl.k != 1:
        raise ValueError("follow_path needs k = 1")
    s = pl.find_start() if start is None else start
    c = pl.cell(s)
    if len(c.zeros) != 2:
        warnings.warn(f"start simplex has {len(c.zeros)} transverse facets")
    a, b = c.zeros[0], c.zeros[1]
    if direction is not None and np.dot(b.x - a.x, direction) < 0:
        a, b = b, a
    pts, us, sims = [a.x, b.x], [a.u, b.u], [s]
    first_face, exit_zero = a.face, b
    closed = False
    arc = [0.0, float(np.linalg.norm(b.x - a.x))]
    while arc[-1] < max_arclength and len(sims) < max_steps:
        if not pl.is_inside(exit_zero.x):
            break
        (i,) = exit_zero.omitted
        s = s.pivot(i)
        c = pl.cell(s)
        others = [z for z in c.zeros if z.face != exit_zero.face]
        if not others:
            warnings.warn("path terminated: no exit facet")
            break
        if len(others) > 1:
            warnings.warn(f"{len(others) + 1} transverse facets in {s}; taking the first exit")
        exit_zero = others[0]
        sims.append(s)
        pts.append(exit_zero.x)
        us.append(exit_zero.u)
        arc.append(arc[-1] + float(np.linalg.norm(pts[-1] - pts[-2])))
        if exit_zero.face == first_face:
            closed = True
            break
    u = None if us[0] is None else np.array(us)
    return Path(np.array(pts), u, np.array(arc), sims, closed)
