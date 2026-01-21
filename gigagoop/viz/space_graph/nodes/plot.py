from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import moderngl as mgl

from gigagoop.coord import Transform

from .node import Node
from .mesh import Mesh

if TYPE_CHECKING:
    from ..engine import BaseEngine


class Plot(Node):
    def __init__(self, engine: BaseEngine, position: np.ndarray, rgba: np.ndarray, lines: bool):
        M_OBJ_WCS = Transform()

        assert position.dtype == np.float32
        self._position = position

        assert rgba.dtype == np.float32
        self._rgba = rgba

        if lines:
            self._render_mode = mgl.LINES
        else:
            self._render_mode = mgl.LINE_STRIP

        super().__init__(engine, M_OBJ_WCS, shader='lines')

    def get_vao(self):
        vao = self._engine.ctx.simple_vertex_array(self._shader_program, self._vbo, 'in_vert', 'in_color')
        return vao

    def get_vbo(self):
        data = np.hstack([self._position, self._rgba]).astype(np.float32)

        vbo = self._engine.ctx.buffer(data)

        return vbo

    def _render(self):
        self._vao.render(self._render_mode)


def _build_cylinder_vertices(p_bottom: np.ndarray,
                             p_top: np.ndarray,
                             radius: float,
                             num_sides: int) -> np.ndarray:
    up = p_top - p_bottom
    if np.allclose(up, 0):
        return np.empty((0, 3), dtype=np.float32)

    M_BOT_WCS = Transform.from_point_and_direction(point=p_bottom, direction=up)
    M_TOP_WCS = Transform.from_point_and_direction(point=p_top, direction=up)

    phi = np.linspace(0, 360, num_sides, endpoint=False)
    x = radius * np.cos(np.deg2rad(phi))
    y = radius * np.sin(np.deg2rad(phi))
    z = np.zeros_like(x)

    pos = np.column_stack([x, y, z])

    pos_top_wcs = M_TOP_WCS * pos
    pos_bot_wcs = M_BOT_WCS * pos

    vertices = []
    for i in range(num_sides):
        j = i + 1 if i < num_sides - 1 else 0

        a_bot = pos_bot_wcs[i]
        a_top = pos_top_wcs[i]
        b_bot = pos_bot_wcs[j]
        b_top = pos_top_wcs[j]

        vertices.append(a_bot)
        vertices.append(b_bot)
        vertices.append(a_top)

        vertices.append(a_top)
        vertices.append(b_top)
        vertices.append(b_bot)

    return np.array(vertices, dtype=np.float32)


class ThickPlot(Mesh):
    def __init__(self,
                 engine: BaseEngine,
                 position: np.ndarray,
                 rgba: np.ndarray,
                 lines: bool,
                 linewidth: float,
                 num_sides: int = 16):
        assert position.dtype == np.float32
        assert rgba.dtype == np.float32

        radius = linewidth / 2.0
        vertices_parts = []
        rgba_parts = []

        if lines:
            segment_pairs = [(idx, idx + 1) for idx in range(0, len(position), 2)]
        else:
            segment_pairs = [(idx, idx + 1) for idx in range(len(position) - 1)]

        for start_idx, end_idx in segment_pairs:
            p_start = position[start_idx]
            p_end = position[end_idx]
            if np.allclose(p_start, p_end):
                continue

            segment_vertices = _build_cylinder_vertices(p_start, p_end, radius, num_sides)
            if segment_vertices.size == 0:
                continue

            num_tris = segment_vertices.shape[0] // 3
            segment_rgba = 0.5 * (rgba[start_idx] + rgba[end_idx])
            vertices_parts.append(segment_vertices)
            rgba_parts.append(np.tile(segment_rgba, (num_tris, 1)))

        if not vertices_parts:
            raise ValueError('plot data did not contain any valid segments for linewidth rendering')

        vertices = np.vstack(vertices_parts).astype(np.float32)
        rgba_faces = np.vstack(rgba_parts).astype(np.float32)
        faces = np.arange(len(vertices), dtype=np.int32).reshape((-1, 3))

        super().__init__(engine=engine, vertices=vertices, faces=faces, rgba=rgba_faces)
