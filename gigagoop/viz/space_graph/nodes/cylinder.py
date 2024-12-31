from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from gigagoop.coord import Transform
from gigagoop.types import ArrayLike
from gigagoop.viz.color import get_vertex_rgba

from .mesh import Mesh

if TYPE_CHECKING:
    from ..engine import BaseEngine


class Cylinder(Mesh):
    def __init__(self,
                 engine: BaseEngine,
                 p_bottom: ArrayLike,
                 p_top: ArrayLike,
                 color: str,
                 radius: float,
                 num_sides: int):

        p_bottom = np.array(p_bottom)
        p_top = np.array(p_top)

        assert len(p_bottom) == 3
        assert len(p_top) == 3

        # Create local coordinate frames on the top and bottom of the cylinder
        up = p_top - p_bottom
        M_BOT_WCS = Transform.from_point_and_direction(point=p_bottom, direction=up)
        M_TOP_WCS = Transform.from_point_and_direction(point=p_top, direction=up)

        # Start by sweeping a circle on the top and bottom of the cylinder
        phi = np.linspace(0, 360, num_sides, endpoint=False)
        x = radius * np.cos(np.deg2rad(phi))
        y = radius * np.sin(np.deg2rad(phi))
        z = np.zeros_like(x)

        pos = np.column_stack([x, y, z])

        pos_top_wcs = M_TOP_WCS * pos    # points along the circle on the top
        pos_bot_wcs = M_BOT_WCS * pos    # points along the circle on the bottom

        # Now that we have a top and bottom, grab the point on the top, and then on the bottom, forming a line
        # on the edge of the cylinder. Notice that if we grab a line next to it, we have a rectangle on the side
        # of the cylinder that we can easily split into two triangles.
        vertices = []

        for i in range(num_sides):
            if i < num_sides - 1:
                j = i + 1
            else:
                j = 0

            # Grab the four vertices that represent the rectangle
            a_bot = pos_bot_wcs[i]
            a_top = pos_top_wcs[i]
            b_bot = pos_bot_wcs[j]
            b_top = pos_top_wcs[j]

            # Now carve this out into triangles
            vertices.append(a_bot)
            vertices.append(b_bot)
            vertices.append(a_top)

            vertices.append(a_top)
            vertices.append(b_top)
            vertices.append(b_bot)

        assert len(vertices) % 3 == 0
        vertices = np.array(vertices).astype(np.float32)

        faces = np.arange(len(vertices)).reshape((-1, 3)).astype(np.int32)
        rgba = get_vertex_rgba(faces, color=color).astype(np.float32)

        super().__init__(engine=engine, vertices=vertices, faces=faces, rgba=rgba)
