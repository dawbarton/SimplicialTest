"""Piecewise linear (simplicial) continuation after Henderson & Melville."""

from .continuation import Cell, Path, PLContinuation, Zero, follow_path
from .kuhn import Lattice, Simplex, face_combinations, locate
from .polytope import SurfaceMesh, polygon, polytope_edges
from .refine import face_conditioning, refine_on_face, refine_orthogonal, secant_jacobian

__all__ = [
    "Cell", "Lattice", "Path", "PLContinuation", "Simplex", "SurfaceMesh", "Zero",
    "face_combinations", "face_conditioning", "follow_path", "locate", "polygon",
    "polytope_edges", "refine_on_face", "refine_orthogonal", "secant_jacobian",
]
