from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import moderngl as mgl

from gigagoop.coord import Transform
from gigagoop.typing import PathLike

if TYPE_CHECKING:
    from ..engine import BaseEngine


class Node(ABC):
    def __init__(self, engine: BaseEngine, M_OBJ_WCS: Transform, shader: str):
        self._engine = engine

        self._is_selected = False
        self._is_hidden = False
        self._M_OBJ_WCS = M_OBJ_WCS

        self._shader_program = self.get_shader_program(shader, self.get_shader_directory())
        self._vbo = self.get_vbo()
        self._vao = self.get_vao()

        self._write_model_matrix_to_uniform()
        self._write_view_matrix_to_uniform()
        self._write_projection_matrix_to_uniform()

    @abstractmethod
    def get_vao(self) -> dict | mgl.VertexArray:
        ...

    @abstractmethod
    def get_vbo(self) -> dict | mgl.Buffer:
        ...

    @abstractmethod
    def _render(self):
        ...

    @property
    def M_OBJ_WCS(self) -> Transform:
        return self._M_OBJ_WCS

    @M_OBJ_WCS.setter
    def M_OBJ_WCS(self, value: Transform):
        assert isinstance(value, Transform)
        self._M_OBJ_WCS = value
        self._write_model_matrix_to_uniform()

    def _write_model_matrix_to_uniform(self):
        model_matrix = self._M_OBJ_WCS
        buffer = _transform_to_uniform(model_matrix)
        if isinstance(self._shader_program, dict):
            for shader in self._shader_program.values():
                shader['M_OBJ_WCS'].write(buffer)
        else:
            self._shader_program['M_OBJ_WCS'].write(buffer)

    def _write_view_matrix_to_uniform(self):
        buffer = _transform_to_uniform(self._engine.camera.view_matrix)
        if isinstance(self._shader_program, dict):
            for shader in self._shader_program.values():
                shader['M_WCS_CAM'].write(buffer)
        else:
            self._shader_program['M_WCS_CAM'].write(buffer)

    def _write_projection_matrix_to_uniform(self):
        buffer = _transform_to_uniform(self._engine.camera.projection_matrix)
        if isinstance(self._shader_program, dict):
            for shader in self._shader_program.values():
                shader['M_CAM_IMG'].write(buffer)
        else:
            self._shader_program['M_CAM_IMG'].write(buffer)

    def select(self):
        self._is_selected = True

    def deselect(self):
        self._is_selected = False

    def is_hidden(self):
        return self._is_hidden

    def show(self):
        self._is_hidden = False

    def hide(self):
        self._is_hidden = True

    def render(self):
        # Before executing the render, we will update both the `view` and `projection` matrix. We do this incase the
        # camera has been updated inside the render loop. The `view` matrix represents the pose of the camera, and it is
        # very common to move the camera around while rendering - each time the camera pose changes, then we must call
        # the `self._write_view_matrix_to_uniform` to update the shader program. In the case of the `projection` matrix,
        # an update to this is less common, but does occur - this happens when the cameras intrinsics change. This could
        # occur if the focal length was being altered (as an example). It is less common to update the `projection`,
        # matrix, so we could do this outside the render loop when that occurs, however, the performance impact of
        # calling it here is negligible (as determined by noting the viewer remains at 60 FPS with or without this line
        # of code).
        if not self._is_hidden:
            self._write_view_matrix_to_uniform()
            self._write_projection_matrix_to_uniform()

            self._render()

    def destroy(self):
        # There is no garbage collector in OpenGL, so clean up manually

        if isinstance(self._vbo, dict):
            for vbo in self._vbo.values():
                vbo.release()
        else:
            self._vbo.release()

        if isinstance(self._shader_program, dict):
            for shader in self._shader_program.values():
                shader.release()
        else:
            self._shader_program.release()

        if isinstance(self._vao, dict):
            for vao in self._vao.values():
                vao.release()
        else:
            self._vao.release()

    def get_shader_directory(self) -> PathLike:
        return (Path(__file__) / '..' / '..' / 'shaders').resolve()

    def get_shader_program(self, shader: str, shader_dir: PathLike) -> dict | mgl.Program:
        shader_dir = Path(shader_dir)
        assert shader_dir.exists(), f'{shader_dir=} should exist'

        vert_file = shader_dir / f'{shader}.vert'
        assert vert_file.exists(), f'{vert_file=} should exist'

        frag_file = shader_dir / f'{shader}.frag'
        assert frag_file.exists(), f'{frag_file=} should exist'

        with open(vert_file) as fid:
            vertex_shader = fid.read()

        with open(frag_file) as fid:
            fragment_shader = fid.read()

        program = self._engine.ctx.program(vertex_shader=vertex_shader,
                                           fragment_shader=fragment_shader)

        return program


def _transform_to_uniform(transform: Transform) -> np.ndarray:
    """This is a utility function to write data to a uniform.

    At a summary level, to write a transform to a uniform:
        * Convert to a numpy array, since it obeys the buffer protocol
        * Convert to float32 since our shaders are set up for that precision
        * Transpose the matrix since opengl uses column-major and numpy uses row-major

    Notes
    -----
    In `Transform` the convention is:

        M_OBJ_WCS = [X.x Y.x Z.x T.x]
                    [X.y Y.y Z.y T.y]
                    [X.z Y.z Z.z T.z]
                    [0   0   0   1  ]

    and the internal "memory layout" is `[[X.x Y.x Z.x T.x] [X.y Y.y Z.y T.y] [X.z Y.z Z.z T.z] [0   0   0   1  ]]`,
    which represents "row major orientation".

    When applying the transform above, we use the convention of `x_wcs = M_OBJ_WCS * x_obj` to map a vector `x_obj`
    defined w.r.t. OBJ to the new vector `x_wcs` defined w.r.t. WCS. Multiplying "on the left" is referred to as
    *** pre-multiplication ***.

    With OpenGL, the memory layout of the above follows "column major", so we cannot "just" naively pass `M_OBJ_WCS`
    into a shader and perform pre-multiplication. If we did this, we would get something like:

        M_OBJ_WCS ---> shader ---> T_OBJ_WCS = [X.x X.y X.z 0]
                                               [Y.x Y.y Y.z 0]
                                               [Z.x Z.y Z.z 0]
                                               [T.x T.y T.z 1]

    where `T_OBJ_WCS` represents the matrix inside the shader, taking note that the internal "memory layout" would be
    given as `[[X.x X.y X.z 0] [Y.x Y.y Y.z 0] [Z.x Z.y Z.z 0] [T.x T.y T.z 1]]`.
    which follows "column major orientation".

    The implication of this is that `Transform` objects must be transposed before being passed to a shader to utilize
    the pre-multiplication convention of `M_OBJ_WCS * x_obj` as we do throughout the code.
    """
    arr = transform.matrix
    arr = arr.astype(np.float32)
    arr = arr.T.copy()
    return arr
