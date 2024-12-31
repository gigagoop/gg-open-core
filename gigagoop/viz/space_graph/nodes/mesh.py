from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import moderngl as mgl

from gigagoop.coord import Transform

from .node import Node

if TYPE_CHECKING:
    from ..engine import BaseEngine


class Mesh(Node):
    def __init__(self, engine: BaseEngine, vertices: np.ndarray, faces: np.ndarray, rgba: np.ndarray):
        M_OBJ_WCS = Transform()

        # Verify the inputs
        assert vertices.dtype == np.float32
        assert faces.dtype == np.int32
        assert rgba.dtype == np.float32

        num_verts, k = vertices.shape
        assert k == 3

        num_tris, k = faces.shape
        assert k == 3

        m, k = rgba.shape
        assert m == num_tris and k == 4

        assert np.min(faces) >= 0
        assert np.max(faces) <= num_verts - 1

        # Duplicate the face colors to each vertex
        tri_vertices = vertices[faces.flatten()]
        tri_rgba = np.repeat(rgba, 3, axis=0)

        assert len(tri_vertices) % 3 == 0
        assert len(tri_rgba) % 3 == 0
        assert tri_vertices.shape[0] == tri_rgba.shape[0]

        self._vertices = tri_vertices
        self._rgba = tri_rgba

        super().__init__(engine, M_OBJ_WCS, shader='mesh')

    def select(self):
        if not self._is_selected:
            Node.select(self)
            self._rgba += np.array([.3, 0., 0., -.5], dtype=np.float32)
            self._vbo.write(np.hstack([self._vertices, np.clip(self._rgba, a_min=0., a_max=1.)]).astype(np.float32))

    def deselect(self):
        if self._is_selected:
            Node.deselect(self)
            self._rgba -= np.array([.3, 0., 0., -.5], dtype=np.float32)
            self._vbo.write(np.hstack([self._vertices, np.clip(self._rgba, a_min=0., a_max=1.)]).astype(np.float32))

    def get_vao(self):
        vao = self._engine.ctx.simple_vertex_array(self._shader_program, self._vbo, 'in_vert', 'in_color')
        return vao

    def get_vbo(self):
        vbo = self._engine.ctx.buffer(np.hstack([self._vertices, self._rgba]).astype(np.float32))
        return vbo

    def _render(self):
        self._vao.render(mgl.TRIANGLES)
