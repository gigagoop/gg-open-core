from __future__ import annotations

from typing import Iterable, Sequence

import imgui
import numpy as np

from gigagoop.coord import Transform
from gigagoop.viz.color import get_vertex_rgba

from .node import Node


def _check_positions(positions) -> np.ndarray:
    positions = np.array(positions, dtype=np.float32)

    if positions.ndim == 1:
        assert len(positions) == 3
        positions = np.atleast_2d(positions)

    assert positions.ndim == 2
    n, k = positions.shape
    assert k == 3

    return positions


def _normalize_texts(text: str | Sequence[str], count: int) -> list[str]:
    if isinstance(text, str):
        if count == 1:
            return [text]
        return [text for _ in range(count)]

    texts = list(text)
    assert len(texts) == count
    return [str(t) for t in texts]


class Text(Node):
    def __init__(self,
                 engine,
                 position,
                 text: str | Sequence[str],
                 color: str | Iterable = 'white',
                 alpha: float | None = 1.0,
                 size: float = 1.0,
                 offset_px: Sequence[float] = (0.0, 0.0),
                 halign: str = 'center',
                 valign: str = 'center',
                 clip: bool = True):

        self._engine = engine
        self._is_hidden = False
        self._M_OBJ_WCS = Transform()

        self._positions = _check_positions(position)
        self._texts = _normalize_texts(text, len(self._positions))
        self._rgba = get_vertex_rgba(self._positions, color, alpha=alpha)

        self._size = float(size)
        self._offset_px = np.array(offset_px, dtype=np.float32).flatten()
        assert len(self._offset_px) == 2

        self._halign = str(halign).lower()
        self._valign = str(valign).lower()
        self._clip = bool(clip)

    def get_vao(self):
        return None

    def get_vbo(self):
        return None

    def _render(self):
        pass

    def render(self):
        # Text labels are rendered in the UI pass.
        return

    def destroy(self):
        # No GL resources are allocated for text labels.
        return

    def _project_to_screen(self, screen_width: int, screen_height: int) -> np.ndarray:
        M_WCS_CAM = self._engine.camera.view_matrix.matrix
        M_CAM_IMG = self._engine.camera.projection_matrix.matrix
        M = M_CAM_IMG @ M_WCS_CAM

        ones = np.ones((len(self._positions), 1), dtype=np.float32)
        pts_h = np.hstack([self._positions, ones])

        clip = (M @ pts_h.T).T
        w = clip[:, 3]

        valid = w > np.finfo(np.float32).eps
        ndc = np.zeros_like(clip[:, :3])
        ndc[valid] = clip[valid, :3] / w[valid, None]

        if self._clip:
            valid &= (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
            valid &= (ndc[:, 0] >= -1.0) & (ndc[:, 0] <= 1.0)
            valid &= (ndc[:, 1] >= -1.0) & (ndc[:, 1] <= 1.0)

        x = (ndc[:, 0] * 0.5 + 0.5) * screen_width
        y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * screen_height

        coords = np.column_stack([x, y, valid])
        return coords

    def render_ui(self, draw_list, screen_width: int, screen_height: int):
        if self._is_hidden:
            return

        if len(self._texts) == 0:
            return

        if self._size <= 0:
            return

        coords = self._project_to_screen(screen_width, screen_height)

        imgui.set_window_font_scale(self._size)
        try:
            for idx, (x, y, valid) in enumerate(coords):
                if not valid:
                    continue

                text = self._texts[idx]
                if not text:
                    continue

                text_size = imgui.calc_text_size(text)
                if hasattr(text_size, 'x'):
                    text_w = text_size.x
                    text_h = text_size.y
                else:
                    text_w, text_h = text_size
                draw_x = float(x) + float(self._offset_px[0])
                draw_y = float(y) + float(self._offset_px[1])

                if self._halign == 'center':
                    draw_x -= text_w * 0.5
                elif self._halign == 'right':
                    draw_x -= text_w

                if self._valign == 'center':
                    draw_y -= text_h * 0.5
                elif self._valign == 'bottom':
                    draw_y -= text_h

                r, g, b, a = self._rgba[idx]
                color_u32 = imgui.get_color_u32_rgba(float(r), float(g), float(b), float(a))
                draw_list.add_text(draw_x, draw_y, color_u32, text)
        finally:
            imgui.set_window_font_scale(1.0)
