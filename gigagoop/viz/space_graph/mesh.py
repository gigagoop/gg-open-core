import logging
from pathlib import Path
import codecs
from typing import Optional, Dict

import numpy as np
from numpy.typing import NDArray
import pywavefront
import cv2

from gigagoop.typing import ArrayLike, PathLike
from gigagoop.coord import get_world_coordinate_system
from gigagoop.viz.color import get_vertex_rgba

log = logging.getLogger(__name__)


class Mesh:
    vertices: NDArray[float]
    faces: NDArray[int]
    rgba: NDArray[float]
    materials: Optional[Dict]

    def __init__(self,
                 vertices: ArrayLike,
                 faces: ArrayLike,
                 rgba: ArrayLike,
                 material: Optional[Dict] = None):

        self.vertices = np.array(vertices).astype(float)
        self.faces = np.array(faces).astype(int)
        self.rgba = np.array(rgba).astype(float)
        self.material = material

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def has_materials(self) -> bool:
        return self.materials is not None

    @property
    def valid_textures(self) -> bool:
        assert self.has_materials

        textures_valid = True
        try:
            for name, material in self.materials.items():
                rgb_image = material['texture']
                assert rgb_image.dtype == np.uint8
                assert rgb_image.ndim == 3
                H, W, N = rgb_image.shape
                assert N == 3
        except:
            textures_valid = False

        return textures_valid


def _strip_bom(file: Path):
    """
    Unreal writes a Byte Order Mark (BOM) on each line that causes issues with pywavefront. Use this routine to remove
    the bom each file.
    """
    with open(file, 'rb') as f:
        content = f.read()

    content = content.lstrip(codecs.BOM_UTF8)

    with open(file, 'wb') as f:
        f.write(content)


def _replace_g_with_o(file: Path):
    """
    The unreal obj file also contains `g <mesh>` which is not supported by `pywavefront`. This changes the "g" to an
    "o" to make pywavefront happy.
    """
    with open(file, 'r') as input_file:
        lines = input_file.readlines()

    modified_lines = []
    for line in lines:
        if line.startswith('g '):
            modified_line = 'o ' + line[2:]
        else:
            modified_line = line
        modified_lines.append(modified_line)

    with open(file, 'w') as output_file:
        output_file.writelines(modified_lines)


def load_ue_mesh(obj_file: PathLike, color='xkcd:pale purple', use_materials=False) -> Mesh:
    """Load a mesh from an OBJ file, where the mesh was dumped from Unreal Engine."""

    obj_file = Path(obj_file)
    assert obj_file.exists()

    assert obj_file.exists(), f'{obj_file=} should exist...'

    # Fix the unreal outputs for pywavefront
    _strip_bom(obj_file)

    # Fix the unsupported specifier
    _replace_g_with_o(obj_file)

    if use_materials:
        # If material is specified, there must be a material file
        mtl_file = obj_file.with_suffix('.mtl')
        assert mtl_file.exists()

        # Fix this file as well
        _strip_bom(mtl_file)

        # We do not want `pywavefront` to create a material
        create_materials = False

    else:
        # In this case, we do not want to use the provided material (even if it exists)
        create_materials = True

    # Now load the mesh
    logging.getLogger('pywavefront').setLevel(logging.WARNING)

    scene = pywavefront.Wavefront(
        obj_file,
        strict=False,
        create_materials=create_materials,
        parse=True,
        collect_faces=True
    )
    if use_materials:
        assert len(scene.meshes) == 1, 'when materials are used, only a single mesh per obj file is supported'

    # We use meters, while Unreal uses cm, so convert...
    cm_to_m = 1 / 100
    vertices = np.array(scene.vertices) * cm_to_m

    # For whatever reason, UE swaps the y and z axis when writing out the obj file
    x, y, z = vertices.T
    vertices = np.column_stack([x, z, y])

    # Stack all faces together -- create one mesh to rule them all
    faces = []
    for mesh in list(scene.meshes.values()):

        # Each row of faces contains indices `[i, j, k]` which reaches into `vertices` to form a triangle.
        faces.extend(mesh.faces)

    faces = np.array(faces)

    num_verts = len(vertices)
    num_faces = len(faces)

    assert np.min(faces) >= 0
    assert np.max(faces) <= num_verts - 1

    log.debug(f'{num_verts=}')
    log.debug(f'{num_faces=}')

    # Package
    rgba = get_vertex_rgba(faces, color)

    mesh = Mesh(vertices, faces, rgba)

    if not use_materials:
        return mesh

    # MATERIALS ========================================================================================================
    # In this section the materials are munged into a more convenient form - this includes loading the textures into
    # memory.
    materials = {}

    for name, material in scene.materials.items():
        assert material.vertex_format == 'T2F_N3F_V3F'  # 2+3+3 = 8 elements
        material_verts = np.array(material.vertices).reshape(-1, 8)

        # The `T2F_N3F_V3F` storage format means texture, normal, and then vertices format. Shuffle around the
        # vertices like shown above...
        u, v, nx, ny, nz, vx, vy, vz = material_verts.T
        material_verts = np.column_stack([u, v, nx, ny, nz, vx * cm_to_m, vz * cm_to_m, vy * cm_to_m])

        # In this form, the vertices must be divisible by three, since each three vertices forms a triangle, and
        # this is required (index buffers cannot be used here).
        assert len(material_verts) % 3 == 0

        # Load the texture
        texture_file = Path(material.texture.find(__file__))
        assert texture_file.exists()
        texture_rgb = cv2.imread(texture_file.as_posix())[:, :, [2, 1, 0]]  # BGR ---> RGB
        assert texture_rgb.ndim == 3

        # Package up this material
        package = {'vertices': material_verts,
                   'format': '[u, v], [nx, ny, nz], [vx, vy, vz]',
                   'texture': texture_rgb}

        materials[name] = package

    mesh.materials = materials
    return mesh


def load_arkit_mesh(obj_file: PathLike, color='xkcd:pale purple') -> Mesh:
    """Load a mesh from an OBJ file, where the mesh was dumped from ARKit."""

    logging.getLogger('pywavefront').setLevel(logging.WARNING)

    scene = pywavefront.Wavefront(
        obj_file,
        strict=True,
        create_materials=False,
        parse=True,
        collect_faces=True
    )

    # Transform the vertices from the "ARKit world coordinate system" (`ARWCS`) to ours (`WCS`)
    M_ARWCS_WCS = get_world_coordinate_system('arkit')

    vertices_arwcs = np.array(scene.vertices)
    vertices_wcs = M_ARWCS_WCS * vertices_arwcs

    # Only a single mesh should exist in the file
    meshes = list(scene.meshes.values())
    assert len(meshes) == 1
    mesh = meshes[0]

    faces = mesh.faces

    rgba = get_vertex_rgba(faces, color)

    mesh = Mesh(vertices_wcs, faces, rgba)

    return mesh
