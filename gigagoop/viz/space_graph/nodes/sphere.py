from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from gigagoop.coord import Transform, Scale

from ..primitives.builder import build_sphere_mesh
from .mesh import Mesh

if TYPE_CHECKING:
    from ..engine import BaseEngine


class Sphere(Mesh):
    def __init__(self, engine: BaseEngine, M_OBJ_WCS: Transform, rgba: np.ndarray, scale: float = 1.0):
        vertices, faces, rgba = build_sphere_mesh(M_OBJ_WCS @ Scale(scale, scale, scale), rgba)
        super().__init__(engine, vertices.astype(np.float32), faces.astype(np.int32), rgba.astype(np.float32))
