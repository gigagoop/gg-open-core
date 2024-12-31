from __future__ import annotations
import numpy as np
import typing
from gigagoop.coord.transform import Transform
from gigagoop.viz.space_graph.nodes import Node, CFrame, Scatter

if typing.TYPE_CHECKING:
    from gigagoop.viz.space_graph.engine.base_engine import BaseEngine


class Camera(Node):
    def __init__(self, engine: BaseEngine, points_xyz: np.ndarray, points_rgb: np.ndarray,
                 M_CAM_WCS: Transform, scale: float = 0.2, show_crs: bool = False):
        Node.__init__(self, engine, M_OBJ_WCS=Transform(), shader='lines')
        self.M_CAM_WCS = M_CAM_WCS
        self.scatter = Scatter(engine, points_xyz, points_rgb, size=3)
        self.cframe = CFrame(engine, M_CAM_WCS, scale_vertices=scale)
        self.show_crs = show_crs

    def get_vbo(self):
        pass

    def get_vao(self):
        pass

    def select(self):
        self.scatter.select()
        self.cframe.select()

    def deselect(self):
        self.scatter.deselect()
        self.cframe.deselect()

    def _render(self):
        self.scatter.render()
        if self.show_crs:
            self.cframe.render()
