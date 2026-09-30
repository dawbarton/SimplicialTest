import itertools

import numpy as np
import pytest

from plcont import (Lattice, PLContinuation, Simplex, SurfaceMesh, face_combinations,
                    follow_path, locate)


# --- Kuhn triangulation -----------------------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_pivot_shares_facet_and_is_involution(n):
    rng = np.random.default_rng(n)
    for _ in range(50):
        s = Simplex(tuple(rng.integers(-5, 5, n).tolist()), tuple(rng.permutation(n).tolist()))
        V = s.vertices()
        for i in range(n + 1):
            p = s.pivot(i)
            assert p != s
            assert set(V) - {V[i]} <= set(p.vertices())
            assert len(set(V) & set(p.vertices())) == n
            # the vertex of p opposite the shared facet pivots back to s
            j = next(j for j, w in enumerate(p.vertices()) if w not in V)
            assert p.pivot(j) == s


@pytest.mark.parametrize("n", [2, 3, 4])
def test_path_simplices_tile_cube(n):
    # n! path simplices of volume 1/n! each, disjoint interiors: check by point location
    rng = np.random.default_rng(0)
    for r in rng.random((200, n)):
        s = locate(r)
        X = np.array(s.vertices(), dtype=float)
        t = np.linalg.solve(np.vstack([X.T, np.ones(n + 1)]), np.append(r, 1.0))
        assert np.all(t >= -1e-12)


def test_paper_example_9_1():
    """Section 9.1: simplex (1,1),(1,0). Figure 6 labels agree with eqs (9)-(11); the text
    of equation (22) misprints the pivots opposite vertices 0 and 1."""
    s = Simplex((1, 1), (1, 0))
    assert s.vertices() == [(1, 1), (1, 2), (2, 2)]
    assert s.pivot(2) == Simplex((0, 1), (0, 1))            # eq (21), as printed
    assert s.pivot(1) == Simplex((1, 1), (0, 1))            # Fig. 6 label; eq (22) says (1,1),(1,0)
    assert s.pivot(0) == Simplex((1, 2), (0, 1))            # eq (22) says (2,1),(0,1)
    assert (2, 1) not in [v for v in Simplex((1, 2), (0, 1)).vertices()]


def test_paper_example_9_2():
    """Section 9.2: simplex (1,1,0),(1,0,2), equation (26)."""
    s = Simplex((1, 1, 0), (1, 0, 2))
    assert s.vertices() == [(1, 1, 0), (1, 2, 0), (2, 2, 0), (2, 2, 1)]
    assert s.pivot(0) == Simplex((1, 2, 0), (0, 2, 1))       # as printed
    assert s.pivot(1) == Simplex((1, 1, 0), (0, 1, 2))       # as printed
    assert s.pivot(2) == Simplex((1, 1, 0), (1, 2, 0))       # as printed
    assert s.pivot(3) == Simplex((1, 1, -1), (2, 1, 0))      # eq (26) prints perm (1,2,0)


def test_face_combinations_match_paper():
    # Section 9.1 (eq 17) and 9.3 (eq 27; the paper swaps V_3 and V_5)
    assert face_combinations(2, 1) == [((0, 1), (2,)), ((0, 2), (1,)), ((1, 2), (0,))]
    fc = dict(face_combinations(3, 2))
    assert fc[(1, 2)] == (0, 3) and fc[(2, 3)] == (0, 1)
    assert len(face_combinations(4, 2)) == 10


def test_locate_barycentre():
    lat = Lattice.around([0.3, -1.2, 2.0], 0.1)
    s = lat.locate([0.3, -1.2, 2.0])
    assert s == Simplex((0, 0, 0), (0, 1, 2))


# --- continuation -----------------------------------------------------------------------

def test_linear_problem_is_exact():
    # For affine F the PL interpolant is exact: every zero lies on the plane
    a = np.array([0.3, -0.7, 0.2, 0.9])
    F = lambda x: [a @ x - 0.1, x[0] - 2 * x[3]]
    pl = PLContinuation(F, [0.1, 0.0, 0.0, 0.05], 0.2, k=2,
                        inside=lambda x: np.all(np.abs(x) < 1.0))
    cells = pl.run()
    Z = np.array([z.x for c in cells for z in c.zeros])
    assert len(cells) > 100
    assert np.max(np.abs(Z @ a - 0.1)) < 1e-12
    assert np.max(np.abs(Z[:, 0] - 2 * Z[:, 3])) < 1e-12


def test_circle_closes():
    F = lambda x: [x[0] ** 2 + x[1] ** 2 - 1.0]
    pl = PLContinuation(F, [1.0, 0.0], 0.05, k=1)
    path = follow_path(pl, direction=[0.0, 1.0])
    assert path.closed
    assert abs(path.arclength[-1] - 2 * np.pi) < 1e-2
    assert np.max(np.abs(np.hypot(*path.x.T) - 1)) < 2e-3
    # the general algorithm finds the same simplices
    cells = PLContinuation(F, [1.0, 0.0], 0.05, k=1).run()
    assert {c.simplex for c in cells} == set(path.simplices)


@pytest.mark.parametrize("order", ["fifo", "lifo", "sorted"])
def test_sphere_surface(order):
    F = lambda x: [x @ x - 1.0]
    pl = PLContinuation(F, [0.0, 0.0, 1.0], 0.15, k=2, order=order)
    mesh = SurfaceMesh(pl, pl.run())
    assert mesh.degenerate == 0
    assert mesh.boundary_edges() == 0
    assert mesh.euler_characteristic() == 2
    assert mesh.components() == 1
    assert np.max(np.abs(np.linalg.norm(mesh.points, axis=1) - 1)) < 0.02


def test_bounded_surface_is_a_disc():
    # a curved 2-manifold in R^4, cut by a box: the PL surface must be a disc (chi = 1)
    F = lambda x: [x[2] - np.sin(2 * x[0]) * x[1], x[3] - x[0] ** 2 + 0.3 * x[1] ** 3]
    pl = PLContinuation(F, [0.1, 0.2, np.sin(0.2) * 0.2, 0.01 - 0.3 * 0.008], (0.1, 0.1, 0.05, 0.05),
                        k=2, inside=lambda x: abs(x[0]) <= 1 and abs(x[1]) <= 1)
    mesh = SurfaceMesh(pl, pl.run())
    assert mesh.degenerate == 0
    assert mesh.components() == 1
    assert len(mesh.boundary_loops()) == 1
    assert mesh.euler_characteristic() == 1
