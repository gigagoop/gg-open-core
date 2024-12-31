import logging
from pathlib import Path
import codecs
from typing import Tuple, Optional, Dict

import numpy as np
import pywavefront
import cv2

from gigagoop.types import PathLike
from gigagoop.coord import Transform, Scale, Translate, RotateY, RotateZ
from gigagoop.viz.color import get_vertex_rgba


log = logging.getLogger(__name__)
__obj_dir__ = Path(__file__).parent


def strip_bom(file: Path):
    """
    Unreal writes a Byte Order Mark (BOM) on each line that causes issues with pywavefront. Use this routine to remove
    the bom each file.
    """
    with open(file, 'rb') as f:
        content = f.read()

    content = content.lstrip(codecs.BOM_UTF8)

    with open(file, 'wb') as f:
        f.write(content)


def replace_g_with_o(file: Path):
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


class UnrealMesh:
    """
    Load an OBJ file exported from UE such that it can be loaded into SpaceGraph.
    """
    def __init__(self, obj_file: PathLike, color='xkcd:pale purple', use_materials=False):
        obj_file = Path(obj_file)
        assert obj_file.exists(), f'{obj_file=} should exist...'

        # Fix the unreal outputs for pywavefront
        strip_bom(obj_file)

        # Fix the unsupported specifier
        replace_g_with_o(obj_file)

        if use_materials:
            # If material is specified, there must be a material file
            mtl_file = obj_file.with_suffix('.mtl')
            assert mtl_file.exists()

            # Fix this file as well
            strip_bom(mtl_file)

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

        # When exporting the mesh from Unreal, we use a single mesh at a time with no material (since
        # `create_materials=True` from above, `pywavefront` creates a material)
        assert len(scene.meshes) == 1
        mesh = list(scene.meshes.values())[0]

        # We use meters, while Unreal uses cm, so convert...
        cm_to_m = 1 / 100
        vertices = np.array(scene.vertices) * cm_to_m

        # For whatever reason, UE swaps the y and z axis when writing out the obj file
        x, y, z = vertices.T
        vertices = np.column_stack([x, z, y])

        # Each row of faces contains indices `[i, j, k]` which reaches into `vertices` to form a triangle.
        faces = np.array(mesh.faces)

        num_verts = len(vertices)
        num_faces = len(faces)

        assert np.min(faces) >= 0
        assert np.max(faces) <= num_verts - 1

        log.debug(f'{num_verts=}')
        log.debug(f'{num_faces=}')

        # Package
        self.vertices = vertices
        self.faces = np.array(mesh.faces)
        self.rgba = get_vertex_rgba(self.faces, color)
        self.materials = None

        if not use_materials:
            return

        # MATERIALS ====================================================================================================
        # In this section the materials are munged into a more convenient form for space graph. This includes loading
        # the textures into memory.
        self.materials = {}

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

            self.materials[name] = package

        # Keeping this old section around for a bit, until we are confident it can be replaced...
        old_section = False
        if old_section:
            # By using a material we get a more refined (although not super accurate) color representation of the mesh.
            # This is done by sampling the texture at each vertex and then averaging the color to yield a color for the
            # entire face of the triangle. If for example, you had a 4096 x 4096 detailed texture with three vertices
            # that span the majority of this texture, you would lose a lot of detail; however, this compromise works
            # for now.
            assert len(scene.materials) == 1

            # We assume a single material (this makes life easier ;-P)
            assert len(mesh.materials) == 1
            material = mesh.materials[0]

            assert material.vertex_format == 'T2F_N3F_V3F'                 # 2+3+3 = 8 elements
            material_verts = np.array(material.vertices).reshape(-1, 8)

            # The `T2F_N3F_V3F` storage format means texture, normal, and then vertices format. Shuffle around the
            # vertices like shown above...
            u, v, nx, ny, nz, vx, vy, vz = material_verts.T
            material_verts = np.column_stack([u, v, nx, ny, nz, vx*cm_to_m, vz*cm_to_m, vy*cm_to_m])

            # The vertices are now unrolled, such that
            V = vertices[faces.flatten()]
            assert len(V) == len(material_verts)
            assert np.linalg.norm(V - material_verts[:, 5:], np.inf) == 0

            # Load the texture
            texture_file = Path(material.texture.find(__file__))
            assert texture_file.exists()
            texture_rgb = cv2.imread(texture_file.as_posix())[:, :, [2, 1, 0]]    # BGR ---> RGB
            assert texture_rgb.ndim == 3

            # Query the texture at each vertex
            height, width, _ = texture_rgb.shape

            u, v = material_verts[:, :2].T

            cols = (u * width).astype(int)
            rows = ((1 - v) * height).astype(int)

            # uv coordinates can wrap, so handle this
            cols = cols % width
            rows = rows % height

            verts_rgb = texture_rgb[cols, rows]
            verts_rgb = (verts_rgb / 255.0).astype(np.float32)

            # Now notice that `verts_rgb` contains the unrolled colors, such that the first triangle has colors
            # associated with the vertices at `[verts_rgb[0], verts_rgb[1], verts_rgb[2]]`, while the second has the
            # color at `[verts_rgb[3], verts_rgb[4], verts_rgb[5]]
            assert verts_rgb.shape == (num_faces*3, 3)

            # We can easily average these colors together with
            faces_rgb = verts_rgb.reshape(-1, 3, 3)
            assert faces_rgb.shape == (num_faces, 3, 3)
            faces_rgb = np.mean(faces_rgb, axis=1)

            # Package up this refined color
            self.rgba = np.c_[faces_rgb, np.ones(num_faces)]

    @property
    def num_triangles(self) -> int:
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


def merge_mesh(sub_vertices, sub_faces, sub_colors):
    merge_vertices = []
    merge_faces = []
    merge_rgba = []

    offset = 0
    for vertices, faces, color in zip(sub_vertices, sub_faces, sub_colors):
        merge_vertices.append(vertices)
        merge_faces.append(np.array(faces) + offset)
        merge_rgba.append(get_vertex_rgba(faces, color))

        offset += len(vertices)

    merge_vertices = np.vstack(merge_vertices)
    merge_faces = np.vstack(merge_faces)
    merge_rgba = np.vstack(merge_rgba)

    return merge_vertices, merge_faces, merge_rgba


def build_cframe_mesh(M_CFRAME_WCS: Transform,
                      colors: Optional[Dict[str, str]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a coordinate frame mesh.

    Builds a coordinate frame mesh from obj files exported from Unreal Engine.

    Returns
    -------
    vertices: shape=[n_verts, 3]
        Each row contains a spatial position `[x, y, z]` of a vertex.

    faces: shape=[n_tris, 3]
        Each row contains indices `[i, j, k]` into the vertices to form a triangle face.

    rgba: shape=[n_tris, 4]
        Each row contains a color (RGBA) for the triangle face.
    """

    sphere_mesh = UnrealMesh(__obj_dir__ / 'sphere.obj')
    cylinder_mesh = UnrealMesh(__obj_dir__ / 'cylinder.obj')
    cone_mesh = UnrealMesh(__obj_dir__ / 'cone.obj')

    # Build the x-axis
    M_XAXIS_CFRAME = Scale(sx=3, sy=1, sz=1)
    M_XARROW_XAXIS = Translate(dx=1, dy=0, dz=0) * Scale(sx=0.25, sy=1, sz=1)

    # Build the y-axis
    M_YAXIS_CFRAME = RotateZ(90) * Scale(sx=3, sy=1, sz=1)
    M_YARROW_YAXIS = Translate(dx=1, dy=0, dz=0) * Scale(sx=0.25, sy=1, sz=1)

    # Build the z-axis
    M_ZAXIS_CFRAME = RotateY(90) * Scale(sx=3, sy=1, sz=1)
    M_ZARROW_ZAXIS = Translate(dx=1, dy=0, dz=0) * Scale(sx=0.25, sy=1, sz=1)

    # Map from object coordinates to cframe
    sphere_vertices_cframe = sphere_mesh.vertices

    xaxis_vertices_cframe = M_XAXIS_CFRAME(cylinder_mesh.vertices)
    yaxis_vertices_cframe = M_YAXIS_CFRAME(cylinder_mesh.vertices)
    zaxis_vertices_cframe = M_ZAXIS_CFRAME(cylinder_mesh.vertices)

    xarrow_vertices_cframe = (M_XAXIS_CFRAME * M_XARROW_XAXIS)(cone_mesh.vertices)
    yarrow_vertices_cframe = (M_YAXIS_CFRAME * M_YARROW_YAXIS)(cone_mesh.vertices)
    zarrow_vertices_cframe = (M_ZAXIS_CFRAME * M_ZARROW_ZAXIS)(cone_mesh.vertices)

    sub_vertices = [sphere_vertices_cframe,
                    xaxis_vertices_cframe,
                    yaxis_vertices_cframe,
                    zaxis_vertices_cframe,
                    xarrow_vertices_cframe,
                    yarrow_vertices_cframe,
                    zarrow_vertices_cframe]

    sub_faces = [sphere_mesh.faces,
                 cylinder_mesh.faces,
                 cylinder_mesh.faces,
                 cylinder_mesh.faces,
                 cone_mesh.faces,
                 cone_mesh.faces,
                 cone_mesh.faces]

    origin_color = 'white'
    x_edge_color = 'red'
    y_edge_color = 'green'
    z_edge_color = 'blue'
    x_tip_color = 'red'
    y_tip_color = 'green'
    z_tip_color = 'blue'

    if colors is not None:
        try:
            origin_color = colors['origin_color']
            x_edge_color = colors['x_edge_color']
            y_edge_color = colors['y_edge_color']
            z_edge_color = colors['z_edge_color']
            x_tip_color = colors['x_tip_color']
            y_tip_color = colors['y_tip_color']
            z_tip_color = colors['z_tip_color']

        except Exception as ex:
            log.error(f'caught exception with {ex=}')

    sub_colors = [origin_color,
                  x_edge_color,
                  y_edge_color,
                  z_edge_color,
                  x_tip_color,
                  y_tip_color,
                  z_tip_color]

    merge_vertices_cframe, merge_faces, merge_rgba = merge_mesh(sub_vertices, sub_faces, sub_colors)

    # It is nice to have the default coordinate frame have arrows with unit length, we can fix up the mesh by computing
    # a scale factor to make this happen
    s = np.max(np.linalg.norm(merge_vertices_cframe, axis=1))
    merge_vertices_cframe = merge_vertices_cframe / s

    # Transform from the local frame coordinate system to the world
    merge_vertices = M_CFRAME_WCS(merge_vertices_cframe)

    return merge_vertices, merge_faces, merge_rgba


def build_sphere_mesh(M_CFRAME_WCS: Transform, rgba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sphere_mesh = UnrealMesh(__obj_dir__ / 'sphere.obj')

    # The sphere color is assumed to be uniform, which means...
    assert len(rgba) == 4

    sphere_vertices = M_CFRAME_WCS(sphere_mesh.vertices)
    sphere_faces = np.array(sphere_mesh.faces)
    sphere_rgba = get_vertex_rgba(sphere_faces, rgba)

    return sphere_vertices, sphere_faces, sphere_rgba
