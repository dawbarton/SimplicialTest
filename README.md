# Simplicial (piecewise linear) continuation

A from-scratch Python implementation of the algorithm described in

> M. E. Henderson and R. Melville, *Piecewise Linear Continuation: Derivative-free Manifold
> Generation*, Research Square preprint (2023), doi:10.21203/rs.3.rs-3612152/v1
> (submitted to ACM TOMS).

The paper presents the Allgower–Schmidt PL continuation method for general k with an emphasis
on implementation, together with the reference C/C++ code DFMGEN. **DFMGEN and other existing
codes were not consulted or used here**; everything is written from the paper's text.

## Package `plcont`

| module | paper | content |
|---|---|---|
| `kuhn.py` | §5, §7, §12 | Coxeter–Freudenthal–Kuhn path simplices `(I, Π)`, the pivot rules (eqs 9–11), point location, lattices (eq 29) |
| `continuation.py` | §4, §8–11 | candidate queue and tested set, per-vertex cache of F (and user functions), per-face cache of face tests (eq 16), zero perturbation, bounded regions; `follow_path` for codim1-style (k = 1) door-in/door-out path following with arclength and interpolated user functions |
| `polytope.py` | §13, §17 | polytope edges via the omitted-vertex intersection rule, polygon ordering, global surface mesh (shared vertices), orientation, Euler characteristic, boundary loops, STL output |
| `refine.py` | §14–16 | secant Jacobian from the simplex (eq 41), face-restricted approximate Newton (eq 43), orthogonal (min-norm, "differenced pseudo-arclength") approximate Newton, conditioning measures |

```python
from plcont import PLContinuation, SurfaceMesh, follow_path
pl = PLContinuation(F, x0, grain=0.05, k=2, inside=None, order="fifo")
cells = pl.run()                     # all simplices meeting F = 0 (connected component of x0)
mesh = SurfaceMesh(pl, cells)        # k = 2: triangulated surface
path = follow_path(PLContinuation(G, y0, 0.01, k=1), max_arclength=25)   # k = 1
```

`F` may return `(residual, user_values)`; user values are recorded at the same lattice
vertices and interpolated onto the manifold (the paper's `nu` user functions).

Run the tests with `python -m pytest`, and the examples with `cd examples; python <name>.py`
(numpy, scipy, and matplotlib; figures and logs go to `figures/`).

## Reproduced examples

| paper | script | paper result | this implementation |
|---|---|---|---|
| §9.1–9.3 worked pivots, face tables | `tests/test_plcont.py` | eqs 17–27 | agree except the misprints listed below |
| Fig. A1 helix, grain 0.01, arclength 25 | `helix.py` | curve plot | 9009 simplices, 9011 F-evaluations (1 per step), max distance to the exact helix 5.0e-5 |
| Figs. A2/A3 torus, grain 0.05 | `torus.py` | 61,808 simplices; frontier plot | 107,048 simplices (see below); closed, one component, χ = 0; vertices within 1.9e-3 of the torus; triangles and quadrilaterals as stated |
| Figs. A4/A5 Klein bottle (eq A1) | `klein.py` | 19,576 simplices; crossings broken into folds | 19,268 simplices at grain 0.22; crossings resolved into non-intersecting sheets, with the pairing varying along the double curve |
| Fig. 3 circuit I–V (codim1) | `circuit.py` | folds near (6.5–6.9 V, 1.0 mA) and (1.25 V, 2.3 mA) | simulated circuit: folds (7.25 V, 1.08 mA) and (1.31 V, 2.48 mA); PL agrees with predictor–corrector to 3 digits |
| Fig. 5 response surface (codim2, n = 4) | `circuit.py` | hysteresis for R ≳ 5 kΩ | 99,619 simplices, a disc (χ = 1); hysteresis for R > 3.1 kΩ |
| §15–16 Newton refinement | `refinement.py` | claims only | PL zeros O(h²); one refinement step O(h³); the orthogonal step is robust to badly aligned faces |

### Notes on the comparisons

* **Simplex counts depend on what `grain` means.** For both surfaces N·grain² is constant to
  under 1% (torus ≈ 267, Klein bottle ≈ 943), so a count fixes the effective lattice spacing.
  With `grain` as the edge length of the Kuhn cubes, the paper's 61,808 torus simplices
  correspond to a cube edge of ≈ 0.0657 = 1.31 × 0.05. The paper's own Fig. A3 ends near
  76,000 simplices (edge ≈ 0.059), which disagrees with the Fig. A2 caption. The paper does not
  define how `grain` sets the lattice, so the counts are consistent up to that scale factor. The
  Klein bottle grain is not given; 0.22 was chosen to match the count.
* **Frontier (Fig. A3).** The paper's frontier curve rises to a peak of about 15% of the total
  halfway through, then falls linearly. That is the shape of LIFO here, though LIFO peaks
  higher (39%). FIFO keeps the frontier tiny (≤ 326), as do random and lexicographically
  sorted selection (`order="sorted"`). The reference implementation's queue discipline is not
  stated.
* **Helix direction.** The paper does not say which way `path_start_a` goes. Fig. A1 descends
  in z, so the path is started in the −z direction.
* **The PL "Klein bottle" is not a Klein bottle.** As the paper notes, the PL zero set is
  embedded. Being a level set in R³ it is also orientable, so it cannot be a Klein bottle. It
  is one closed orientable surface whose genus grows under refinement: 11, 25, and 53 at
  grains 0.22, 0.1, and 0.05. Each crossing along the double curve (the circle
  x² + (y−1)² = 2, z = 0, where F = ∇F = 0) resolves one of two ways, and the switches add
  handles. `klein_crossing.png` shows cross-sections normal to the double curve.
* **Circuit (a reconstruction, not the authors' simulator).** Topology from Fig. 1, Ebers–Moll
  transistors with typical 2N3904/2N3906 SPICE parameters (quoted from memory, not checked),
  and the internal emitter node of q2 solved inside the "simulator". The schematic draws q1
  with a PNP arrow, but a PNP with a grounded emitter cannot conduct, so q1 is taken as an
  NPN, which gives the thyristor-like S-curve of Fig. 3. With no fitting, the folds land
  within about 10% of the paper's. The paper does not define R(α) beyond R(1) = 43 kΩ;
  R(α) = 1 kΩ·43^α is assumed, so the α at which hysteresis appears (0.30 here vs 0.2 in the
  paper) is not a meaningful comparison, but the resistance (3.1 kΩ vs "about 5 kΩ") is. The
  paper's Fig. 5 shows currents up to 0.1 A, which this circuit cannot reach
  (7 V / 499 Ω ≈ 14 mA), so their simulator setup evidently differed. Fig. 3 is swept to 8 V
  because this model's upper fold is at 7.25 V.
* **Hardware-continuation emulation.** Adding 2 µA Gaussian noise to I1, I2, and Idrv (each
  vertex measured once, as §11 requires) still traces the full S-curve, with folds shifted by
  under 0.1 mA.

### Implementation choices not fixed by the paper

* Lattice: axis-aligned, with spacing `grain` (a scalar or per coordinate), translated so the
  starting point is the barycentre of simplex `(0, identity)` (§12). If that simplex is not
  transverse, a breadth-first search over neighbours finds one.
* Face test: eq 16, rejected as singular if cond > 1e12; exact zeros of F at a vertex are
  replaced by +1e-13 (§11).
* Bounded regions: the boundary function is applied to the zero that triggers each pivot, not
  to the simplex. Testing simplex barycentres (the first version) left small spurious holes
  along the cut: the codim2 circuit surface had χ = −9, with ten holes that moved when the
  bound moved. With the zero-based test it is a clean disc (χ = 1), and a regression test
  checks this.
* Refinement: the §16 orthogonal iteration is implemented as the minimum-norm step
  dx = −J⁺F(x) with the simplex secant Jacobian. That is the same as constraining the step to
  be orthogonal to the null space of J, which the paper builds by Gram–Schmidt on the polytope
  vertices.

### What the refinement experiment shows (`refinement.png`)

On the torus, with a sample of 1500 zeros per grain:

* The PL zeros converge as O(h²).
* One approximate-Newton step gives O(h³) in the median, with linear convergence thereafter
  (contraction factor O(h)). That is what an O(h)-accurate secant Jacobian predicts.
* The face-restricted step (§15) has a bad worst case. Faces nearly parallel to the surface
  (amplification 1/|cos θ| up to about 60) give errors comparable to, or at grain 0.2 larger
  than, the unrefined PL error.
* The orthogonal step (§16) removes that: its worst case also converges as O(h³), and is
  9–40× smaller than the face-restricted worst case. This supports the paper's §16 claim
  quantitatively.

## Errata found in the paper

1. Eq (22): the pivot of (1,1),(1,0) across the face opposite vertex 0 is (1,2),(0,1), not
   (2,1),(0,1); opposite vertex 1 it is (1,1),(0,1) (as Fig. 6 labels it), not (1,1),(1,0).
2. Eq (26): the pivot across the face opposite vertex 3 is (1,1,−1),(2,1,0) (rotate right,
   eq 11), not (1,1,−1),(1,2,0).
3. Eq (27): V₃ and V₅ are swapped (F₃ = [1,2] ⇒ V₃ = [0,3]; F₅ = [2,3] ⇒ V₅ = [0,1]).
4. §9.3 starts from "(1,1,0),(0,1,2)" but means the simplex (1,1,0),(1,0,2) of §9.2.
5. Fig. 7 caption says n = 3, k = 2; the map R³ → R² has k = 1.
6. §6: "k of the n − 1 vertices which do not lie on the face" should be n + 1.
7. §13: "0 ≤ θ[j] ≤ 0" should be ≤ 1. Eq 44 mixes the indices i and j.
8. §17.2 is titled "k = 1" but treats k = 2. §17.3's rule "n − k − 1 entries in the
   intersection" contradicts the tables in §17.2, which (correctly) use k − 1 shared omitted
   vertices.
9. Fig. 1 draws q1 with a PNP symbol (see the circuit notes above).
10. The torus simplex count differs between Fig. A2 (61,808) and Fig. A3 (≈ 76,000).
