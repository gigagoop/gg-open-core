from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import moderngl as mgl

from gigagoop.coord import Transform

from .node import Node

if TYPE_CHECKING:
    from ..engine import BaseEngine


class Grid(Node):
    def __init__(self, engine: BaseEngine, M_OBJ_WCS: Transform, rgba: np.ndarray, n_lines: int):
        assert len(rgba) == 4
        assert rgba.dtype == np.float32

        assert isinstance(n_lines, int)
        self.n_lines = n_lines

        super().__init__(engine, M_OBJ_WCS, shader='grid')

        self._shader_program['GRID_RGBA'].write(rgba)

    def get_vao(self):
        vao = self._engine.ctx.simple_vertex_array(self._shader_program, self._vbo, 'in_vert')
        return vao

    def get_vbo(self):
        vertices = []

        n_lines = self.n_lines

        for x in range(-n_lines, n_lines + 1):
            vertices.extend([(x, -n_lines, 0), (x, n_lines, 0)])

        for y in range(-n_lines, n_lines + 1):
            vertices.extend([(-n_lines, y, 0), (n_lines, y, 0)])

        vertices = np.array(vertices, np.float32)

        vbo = self._engine.ctx.buffer(vertices)

        return vbo

    def _render(self):
        self._vao.render(mgl.LINES)
