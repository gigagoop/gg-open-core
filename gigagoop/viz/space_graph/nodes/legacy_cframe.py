from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import moderngl as mgl

from gigagoop.coord import Transform

from .node import Node

if TYPE_CHECKING:
    from ..engine import Engine


class CFrame(Node):
    def __init__(self, engine: Engine, M_OBJ_WCS: Transform):
        super().__init__(engine, M_OBJ_WCS, shader='lines')

    def get_vao(self):
        vao = self._engine.ctx.simple_vertex_array(self._shader_program, self._vbo, 'in_vert', 'in_color')
        return vao

    def get_vbo(self):
        vertices = [(0, 0, 0), (1, 0, 0),
                    (0, 0, 0), (0, 1, 0),
                    (0, 0, 0), (0, 0, 1)]

        colors = [(1, 0, 0, 1), (1, 0, 0, 1),
                  (0, 1, 0, 1), (0, 1, 0, 1),
                  (0, 0, 1, 1), (0, 0, 1, 1)]

        data = np.hstack([vertices, colors]).astype(np.float32)

        vbo = self._engine.ctx.buffer(data)

        return vbo

    def _render(self):
        self._vao.render(mgl.LINES)
