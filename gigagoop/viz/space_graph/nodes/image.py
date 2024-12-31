from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np
import moderngl as mgl

from gigagoop.coord import Transform

from .node import Node

if TYPE_CHECKING:
    from ..engine import BaseEngine


log = logging.getLogger(__name__)


class Image(Node):
    """Place an image in world space.

    Given a frame `M_OBJ_WCS` such that:
        X:  forward
        Y:  right
        Z:  up
        O:  origin

    an image is centered at the origin and spans YZ.
    """
    def __init__(self,
                 engine: BaseEngine,
                 M_OBJ_WCS: Transform,
                 image: np.ndarray,
                 width: float,
                 height: float,
                 alpha: float):

        if not image.dtype == np.uint8:
            log.warning(f'converting from {image.dtype} to `np.uint8`')

            # Assuming the image is normalized between [0, 1]
            assert np.min(image) >= 0
            assert np.max(image) <= 1
            image = (255 * image).astype(np.uint8)

        if image.ndim == 2:
            # A grayscale image was passed in
            image = np.dstack(image, image, image)
        else:
            assert image.ndim == 3

        H, W, N = image.shape
        assert N == 3 or N == 4

        if N == 4:
            log.warning('a RGBA image was passed in, dropping the alpha channel...')
            image = image[:, :, :3]

        # OpenGL assumes column major
        self._pixels = np.asfortranarray(image)

        # The `width` and `height` parameters represent actual real-world distances (in meters)
        assert width > 0
        assert height > 0

        self.width = width
        self.height = height
        super().__init__(engine, M_OBJ_WCS, shader='image')

        # Load the texture (CPU to GPU)
        H, W, N = image.shape
        self._texture = engine.ctx.texture(size=(W, H), components=N, data=self._pixels.tobytes(), dtype='f1')

        # Tell the shaders about the alpha value
        alpha = np.array([alpha]).astype(np.float32)
        self._shader_program['ALPHA'].write(alpha)

    def get_vao(self):
        vao = self._engine.ctx.simple_vertex_array(self._shader_program, self._vbo, 'in_vert', 'in_uv')
        return vao

    def get_vbo(self):
        left = -self.width/2
        right = self.width/2
        top = self.height/2
        bottom = -self.height/2

        data = np.array([
            # X  Y      Z       U    V
            0.0, left,  top,    0.0, 0.0,    # top-left
            0.0, right, top,    1.0, 0.0,    # top-right
            0.0, left,  bottom, 0.0, 1.0,    # bottom-left
            0.0, right, bottom, 1.0, 1.0,    # bottom-right
        ]).astype(np.float32)

        vbo = self._engine.ctx.buffer(data)

        return vbo

    def _render(self):
        self._texture.use()
        self._vao.render(mgl.TRIANGLE_STRIP)
