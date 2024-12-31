from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import moderngl as mgl

from gigagoop.coord import Transform
from gigagoop.viz.space_graph.primitives import builder

from .node import Node

if TYPE_CHECKING:
    from ..engine import BaseEngine


class UnrealMesh(Node):
    def __init__(self, engine: BaseEngine, obj_file: str):
        M_OBJ_WCS = Transform()

        # Currently we assume materials...
        mesh = builder.UnrealMesh(obj_file, use_materials=True)

        self._mesh = mesh

        # Load all the textures to the GPU
        self._textures = {}
        for name, material in mesh.materials.items():
            rgb_image = material['texture']
            assert rgb_image.dtype == np.uint8
            assert rgb_image.ndim == 3
            H, W, N = rgb_image.shape
            assert N == 3
            texture = engine.ctx.texture(size=(W, H), components=N, data=rgb_image.tobytes(), dtype='f1')
            self._textures[name] = texture

        super().__init__(engine, M_OBJ_WCS, shader='unreal_mesh')

    def get_vao(self):
        vaos = {}

        for name, material in self._mesh.materials.items():
            vao = self._engine.ctx.simple_vertex_array(self._shader_program, self._vbo[name], 'in_vert', 'in_uv')
            vaos[name] = vao

        return vaos

    def get_vbo(self):
        vbos = {}

        for name, material in self._mesh.materials.items():
            vertices = material['vertices']    # [u, v], [nx, ny, nz], [vx, vy, vz]
            u, v, _, _, _, vx, vy, vz = vertices.T
            data = np.column_stack([vx, vy, vz, u, v]).astype(np.float32)

            vbo = self._engine.ctx.buffer(data)
            vbos[name] = vbo

        return vbos

    def _render(self):
        for name in self._mesh.materials:
            self._textures[name].use()
            self._vao[name].render(mgl.TRIANGLES)
