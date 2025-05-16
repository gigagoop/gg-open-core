from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray
import moderngl as mgl

from gigagoop.coord import Transform

from .node import Node

if TYPE_CHECKING:
    from ..engine import BaseEngine


class LitMesh(Node):
    def __init__(self,
                 engine: BaseEngine,
                 vertices: NDArray,
                 faces: NDArray,
                 normals: NDArray,
                 rgba: NDArray,
                 light_pos: NDArray = np.array([0, 0, 10]),
                 light_color: NDArray = np.array([1, 1, 1]),
                 light_intensity: float = 1.0,
                 ambient_light: NDArray = np.array([0.1, 0.1, 0.1])):

        M_OBJ_WCS = Transform()

        # Verify the inputs
        light_pos = np.array(light_pos).astype(np.float32)
        light_color = np.array(light_color).astype(np.float32)
        light_intensity = float(light_intensity)
        ambient_light = np.array(ambient_light).astype(np.float32)

        assert len(light_pos) == 3
        assert len(light_color) == 3
        assert len(ambient_light) == 3

        self.light_pos = light_pos
        self.light_color = light_color
        self.light_intensity = light_intensity
        self.ambient_light = ambient_light

        assert vertices.dtype == np.float32
        assert faces.dtype == np.int32
        assert normals.dtype == np.float32
        assert rgba.dtype == np.float32

        num_verts, k = vertices.shape
        assert k == 3

        num_tris, k = faces.shape
        assert k == 3

        m, k = normals.shape
        assert m == num_tris and k == 3

        m, k = rgba.shape
        assert m == num_tris and k == 4

        assert np.min(faces) >= 0
        assert np.max(faces) <= num_verts - 1

        # Duplicate the face colors to each vertex
        tri_vertices = vertices[faces.flatten()]
        tri_normals = np.repeat(normals, 3, axis=0)
        tri_rgba = np.repeat(rgba, 3, axis=0)

        assert len(tri_vertices) % 3 == 0
        assert len(tri_normals) % 3 == 0
        assert len(tri_rgba) % 3 == 0
        assert tri_vertices.shape[0] == tri_normals.shape[0] == tri_rgba.shape[0]

        self._vertices = tri_vertices
        self._normals = tri_normals
        self._rgba = tri_rgba

        super().__init__(engine, M_OBJ_WCS, shader='lit_mesh')

    def get_vao(self):
        vao = self._engine.ctx.simple_vertex_array(self._shader_program,
                                                   self._vbo,
                                                   'in_vert', 'in_normal', 'in_color')
        return vao

    def get_vbo(self):
        vbo = self._engine.ctx.buffer(np.hstack([self._vertices, self._normals, self._rgba]).astype(np.float32))
        return vbo

    def _render(self):
        self._shader_program['light_pos'].write(self.light_pos)
        self._shader_program['light_color'].write(self.light_color)
        self._shader_program['light_intensity'].value = self.light_intensity
        self._shader_program['ambient_light'].write(self.ambient_light)
        self._vao.render(mgl.TRIANGLES)
