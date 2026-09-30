"""Assembling the polytopes in each simplex into a polyhedral manifold (Section 17)."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .continuation import Cell


def polytope_edges(cell: Cell, k: int) -> list[tuple[int, int]]:
    """Pairs of polytope vertices joined by an edge.

    Two zeros lie on a common edge of the polytope when they lie on a common
    (n-k+1)-face of the simplex, i.e. their sets of omitted vertices share k-1 entries
    (the intersection rule of Section 17.2).
    """
    return [(a, b) for a, b in combinations(range(len(cell.zeros)), 2)
            if len(cell.zeros[a].omitted & cell.zeros[b].omitted) == k - 1]


def polygon(cell: Cell) -> list[int]:
    """Vertices of the (k = 2) polygon in a simplex, in cyclic order."""
    nz = len(cell.zeros)
    adj = {i: [] for i in range(nz)}
    for a, b in polytope_edges(cell, 2):
        adj[a].append(b)
        adj[b].append(a)
    if nz < 3 or any(len(v) != 2 for v in adj.values()):
        raise ValueError(f"degenerate polygon in {cell.simplex}: degrees "
                         f"{[len(v) for v in adj.values()]}")
    loop, prev = [0], None
    while True:
        cur = loop[-1]
        nxt = adj[cur][0] if adj[cur][0] != prev else adj[cur][1]
        if nxt == 0:
            break
        loop.append(nxt)
        prev = cur
    if len(loop) != nz:
        raise ValueError(f"polygon in {cell.simplex} is not a single cycle")
    return loop


def pl_gradient(pl, cell: Cell) -> np.ndarray:
    """Gradient of the (scalar, m = 1) PL interpolant of F inside a simplex."""
    X = np.array([pl.lattice.to_x(v) for v in cell.vertices])
    f = np.array([pl.value(v)[0][0] for v in cell.vertices])
    return np.linalg.solve(X[1:] - X[0], f[1:] - f[0])


class SurfaceMesh:
    """Triangulated k = 2 PL surface with vertices shared between simplices.

    Global vertices are the transverse (n-k)-faces, so a zero shared by adjacent simplices
    is a single mesh vertex. For n = 3, triangles are oriented along the gradient of the PL
    interpolant, which is continuous across simplices and so orients the surface globally.
    """

    def __init__(self, pl, cells):
        self.pl = pl
        self.index: dict = {}
        pts, users = [], []
        self.polygons: list[list[int]] = []
        self.cell_edges: list[list[tuple]] = []    # per cell, edge keys ((n-k+1)-faces)
        self.degenerate = 0
        for c in cells:
            try:
                loop = polygon(c)
            except ValueError:
                self.degenerate += 1
                continue
            ids = []
            for j in loop:
                z = c.zeros[j]
                if z.face not in self.index:
                    self.index[z.face] = len(pts)
                    pts.append(z.x)
                    users.append(z.u)
                ids.append(self.index[z.face])
            if pl.n == 3:
                a, b, d = (c.zeros[loop[j]].x for j in range(3))
                if np.dot(np.cross(b - a, d - a), pl_gradient(pl, c)) < 0:
                    ids = ids[::-1]
            self.polygons.append(ids)
            self.cell_edges.append([
                tuple(sorted(set(c.zeros[loop[j]].face) | set(c.zeros[loop[j - 1]].face)))
                for j in range(len(loop))])
        self.points = np.array(pts)
        self.user = None if not users or users[0] is None else np.array(users)

    @property
    def triangles(self) -> np.ndarray:
        """Fan triangulation of each polygon."""
        return np.array([(p[0], p[j], p[j + 1]) for p in self.polygons
                         for j in range(1, len(p) - 1)])

    def euler_characteristic(self) -> int:
        edges = {e for es in self.cell_edges for e in es}
        return len(self.points) - len(edges) + len(self.polygons)

    def boundary_edges(self) -> int:
        """Number of edges used by only one polygon (0 for a closed surface)."""
        count: dict = {}
        for es in self.cell_edges:
            for e in es:
                count[e] = count.get(e, 0) + 1
        return sum(1 for v in count.values() if v == 1)

    def boundary_loops(self) -> list[list[int]]:
        """Vertex sets of the connected pieces of the boundary (edges in one polygon only)."""
        count: dict = {}
        for es in self.cell_edges:
            for e in es:
                count[e] = count.get(e, 0) + 1
        adj: dict = {}
        for p, es in zip(self.polygons, self.cell_edges):
            for j, e in enumerate(es):
                if count[e] == 1:
                    adj.setdefault(p[j], []).append(p[j - 1])
                    adj.setdefault(p[j - 1], []).append(p[j])
        loops, seen = [], set()
        for s in adj:
            if s in seen:
                continue
            comp, stack = [], [s]
            while stack:
                v = stack.pop()
                if v not in seen:
                    seen.add(v)
                    comp.append(v)
                    stack.extend(adj[v])
            loops.append(comp)
        return loops

    def components(self) -> int:
        """Number of connected components (polygons joined through shared edges)."""
        parent = list(range(len(self.points)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for p in self.polygons:
            for j in range(1, len(p)):
                parent[find(p[j])] = find(p[0])
        return len({find(i) for i in range(len(self.points))})

    def write_stl(self, path, axes=(0, 1, 2)):
        P = self.points[:, list(axes)]
        with open(path, "w") as fp:
            fp.write("solid plcont\n")
            for a, b, c in self.triangles:
                nrm = np.cross(P[b] - P[a], P[c] - P[a])
                nn = np.linalg.norm(nrm)
                nrm = nrm / nn if nn > 0 else nrm
                fp.write(f"facet normal {nrm[0]:e} {nrm[1]:e} {nrm[2]:e}\n outer loop\n")
                for v in (a, b, c):
                    fp.write(f"  vertex {P[v, 0]:e} {P[v, 1]:e} {P[v, 2]:e}\n")
                fp.write(" endloop\nendfacet\n")
            fp.write("endsolid plcont\n")
