from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np
import moderngl as mgl

from gigagoop.coord import Transform

from .node import Node

if TYPE_CHECKING:
    from ..engine import BaseEngine


TEXTURE_INDEX = 5


class MaterialMesh(Node):
    def __init__(self,
                 engine: BaseEngine,
                 vertices: np.ndarray,
                 texture: np.ndarray,
                 M_OBJ_WCS: Optional[Transform] = None):

        if M_OBJ_WCS is None:
            M_OBJ_WCS = Transform()

        # Verify the inputs
        assert vertices.dtype == np.float64
        assert texture.dtype == np.uint8

        num_verts, k = vertices.shape
        assert k == 8
        assert num_verts % 3 == 0    # each three points form a triangle

        height, width, num_channels = texture.shape
        assert num_channels == 3

        # Pack up the vertices and uv coordinates
        u, v, nx, ny, nz, vx, vy, vz = vertices.T

        self._vertices = np.column_stack([vx, vy, vz])
        self._uv_coords = np.column_stack([u, v])

        # Each texture is bound to a texture unit. As a total simplification, we will just increment the texture index
        # to make sure each material mesh is bound to a unique texture unit. If you are spawning lots of nodes that
        # share a texture, this is stupid; however, we don't currently have a use case where we are spawning lots of
        # objects that share a texture... When we do, the assert below force us to do it a smarter way.
        global TEXTURE_INDEX
        k = TEXTURE_INDEX

        self._texture = engine.ctx.texture((width, height), 3, texture.tobytes())
        self._texture.use(location=TEXTURE_INDEX)

        TEXTURE_INDEX += 1
        assert TEXTURE_INDEX < 16, 'see note above on why it was implemented like this'

        super().__init__(engine, M_OBJ_WCS, shader='material_mesh')
        self._shader_program['texture0'].value = k

    def get_vao(self):
        vao = self._engine.ctx.vertex_array(self._shader_program, [(self._vbo, '3f 2f', 'in_vert', 'in_text')])

        return vao

    def get_vbo(self):
        #data = self._vertices.astype(np.float32)
        data = np.hstack([self._vertices, self._uv_coords]).astype(np.float32)
        vbo = self._engine.ctx.buffer(data)

        return vbo

    def _render(self):
        self._vao.render(mgl.TRIANGLES)
