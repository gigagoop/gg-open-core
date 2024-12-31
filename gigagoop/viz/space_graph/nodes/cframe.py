from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict

import numpy as np

from gigagoop.coord import Transform

from ..primitives.builder import build_cframe_mesh
from .mesh import Mesh

if TYPE_CHECKING:
    from ..engine import BaseEngine


class CFrame(Mesh):
    def __init__(self,
                 engine: BaseEngine,
                 M_OBJ_WCS: Transform,
                 scale_vertices: float = 1.0,
                 colors: Optional[Dict[str, str]] = None):

        # Update (20231213) --------------------------------------------------------------------------------------------
        # old >>>
        #vertices, faces, rgba = build_cframe_mesh(M_OBJ_WCS)
        #super().__init__(engine, vertices.astype(np.float32), faces.astype(np.int32), rgba.astype(np.float32))
        # <<< old

        # new >>>
        # In the section above (marked `old`) the `build_cframe_mesh(M_OBJ_WCS)` would construct vertices that are
        # defined w.r.t. the world, and the internal `M_OBJ_WCS` would just be set to identity. This worked fine if the
        # coordinate system remained static; however, an update was just made where a node can have its internal
        # frame updated, which would then cause problems. The reason for this, is an updated `M_OBJ_WCS` would be not
        # updating object coordinates, but world coordinates. An easy fix is to call `build_cframe_mesh(Transform())
        # which means the vertices are defined in object coordinates. We can then call `self.M_OBJ_WCS = M_OBJ_WCS` to
        # set the proper internal frame.
        vertices, faces, rgba = build_cframe_mesh(Transform(), colors)
        vertices = vertices * scale_vertices
        super().__init__(engine, vertices.astype(np.float32), faces.astype(np.int32), rgba.astype(np.float32))
        self.M_OBJ_WCS = M_OBJ_WCS
        # <<< new

        # Update (20231213) --------------------------------------------------------------------------------------------