from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import moderngl as mgl

from gigagoop.coord import Transform

from .node import Node

if TYPE_CHECKING:
    from ..engine import BaseEngine


class DepthMesh(Node):
    def __init__(self, engine: BaseEngine, vertices: np.ndarray, faces: np.ndarray):
        M_OBJ_WCS = Transform()

        # Verify the inputs
        assert vertices.dtype == np.float32
        assert faces.dtype == np.int32

        num_verts, k = vertices.shape
        assert k == 3

        num_tris, k = faces.shape
        assert k == 3

        assert np.min(faces) >= 0
        assert np.max(faces) <= num_verts - 1

        tri_vertices = vertices[faces.flatten()]
        assert len(tri_vertices) % 3 == 0

        self._vertices = tri_vertices

        super().__init__(engine, M_OBJ_WCS, shader='depth_mesh')

    def get_vao(self):
        vao = self._engine.ctx.simple_vertex_array(self._shader_program, self._vbo, 'in_vert')
        return vao

    def get_vbo(self):
        data = self._vertices.astype(np.float32)

        vbo = self._engine.ctx.buffer(data)

        return vbo

    def _render(self):
        self._vao.render(mgl.TRIANGLES)
