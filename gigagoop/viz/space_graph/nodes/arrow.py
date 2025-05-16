from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from gigagoop.coord import Transform, Scale

from .mesh import Mesh
from ..primitives.builder import build_arrow_mesh

if TYPE_CHECKING:
    from ..engine import BaseEngine


class Arrow(Mesh):
    def __init__(self, engine: BaseEngine, origin: NDArray, direction: NDArray, width: float, rgba: NDArray):

        origin = np.array(origin)
        direction = np.array(direction)
        rgba = np.array(rgba)
        assert origin.shape == direction.shape
        n_arrows, n_tuple = origin.shape
        assert n_tuple == 3

        # Get an arrow of unit length in the X direction with a default color
        base_vertices, base_faces, base_rgba = build_arrow_mesh()

        n_vertices = len(base_vertices)
        n_faces = len(base_faces)

        # This is not the most optimal way of doing this... we should be using instancing. The idea here is essentially
        # to create one large mesh which will at least put this into a single node, so only one draw is issued for it.
        vertices = []
        faces = []
        colors = []

        for i, (p0, d, color_i) in enumerate(zip(origin, direction, rgba)):

            p1 = p0 + d

            X = p1 - p0
            length = np.linalg.norm(X)
            X = X / length

            A = np.array([1, 0, 0]) if np.abs(X[0]) < 0.9 else np.array([0, 1, 0])
            Y = np.cross(A, X)
            Y = Y / np.linalg.norm(Y)

            Z = np.cross(X, Y)

            S = Scale(length, width, width)
            M_OBJ_WCS = Transform.from_rotation_and_origin(rotation=np.c_[X, Y, Z], origin=p0) @ S

            vertices.append(M_OBJ_WCS(base_vertices))
            faces.append(base_faces + i*n_vertices)
            colors.append(np.tile(color_i, (n_faces, 1)))

        vertices = np.vstack(vertices)
        faces = np.vstack(faces)
        colors = np.vstack(colors)

        super().__init__(engine, vertices.astype(np.float32), faces.astype(np.int32), colors.astype(np.float32))
