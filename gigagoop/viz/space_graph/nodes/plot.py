from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import moderngl as mgl

from gigagoop.coord import Transform

from .node import Node

if TYPE_CHECKING:
    from ..engine import BaseEngine


class Plot(Node):
    def __init__(self,
                 engine: BaseEngine,
                 position: np.ndarray,
                 rgba: np.ndarray,
                 lines: bool,
                 linewidth: float | None):
        M_OBJ_WCS = Transform()

        assert position.dtype == np.float32
        self._position = position

        assert rgba.dtype == np.float32
        self._rgba = rgba

        if lines:
            self._render_mode = mgl.LINES
        else:
            self._render_mode = mgl.LINE_STRIP

        if linewidth is not None:
            linewidth = float(linewidth)
            assert linewidth > 0
        self._linewidth = linewidth

        super().__init__(engine, M_OBJ_WCS, shader='lines')

    def get_vao(self):
        vao = self._engine.ctx.simple_vertex_array(self._shader_program, self._vbo, 'in_vert', 'in_color')
        return vao

    def get_vbo(self):
        data = np.hstack([self._position, self._rgba]).astype(np.float32)

        vbo = self._engine.ctx.buffer(data)

        return vbo

    def _render(self):
        ctx = self._engine.ctx
        if self._linewidth is None:
            self._vao.render(self._render_mode)
            return

        previous_width = ctx.line_width
        ctx.line_width = self._linewidth
        self._vao.render(self._render_mode)
        ctx.line_width = previous_width
