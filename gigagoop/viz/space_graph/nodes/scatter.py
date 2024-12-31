from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import moderngl as mgl

from gigagoop.coord import Transform

from .node import Node

if TYPE_CHECKING:
    from ..engine import BaseEngine


class Scatter(Node):
    def __init__(self, engine: BaseEngine, position: np.ndarray, rgba: np.ndarray, size: float):
        M_OBJ_WCS = Transform()

        assert position.dtype == np.float32
        self._position = position

        assert rgba.dtype == np.float32
        self._rgba = rgba

        super().__init__(engine, M_OBJ_WCS, shader='points')

        size = np.array([size]).astype(np.float32)
        self._shader_program['POINT_SIZE'].write(size)

    def select(self):
        if not self._is_selected:
            Node.select(self)
            self._vbo.write(np.hstack([self._position,
                                       np.clip(self._rgba + np.array([.3, 0., 0., -.5], dtype=np.float32), \
                                               a_min=0., a_max=1.)]).astype(np.float32))

    def deselect(self):
        if self._is_selected:
            Node.deselect(self)
            self._vbo.write(np.hstack([self._position, self._rgba]).astype(np.float32))

    def get_vao(self):
        vao = self._engine.ctx.simple_vertex_array(self._shader_program, self._vbo, 'in_vert', 'in_color')
        return vao

    def get_vbo(self):
        vbo = self._engine.ctx.buffer(np.hstack([self._position, self._rgba]).astype(np.float32))
        return vbo

    def _render(self):
        self._vao.render(mgl.POINTS)
