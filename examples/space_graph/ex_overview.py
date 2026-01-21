"""
This example demonstrates how to use the `SpaceGraph` interface.
"""
import logging

import numpy as np
import skimage

from gigagoop.viz import SpaceGraph
from gigagoop.coord import GazeAt

log = logging.getLogger('tour')


def generate_torus(R=1.0, r=0.3, nu=64, nv=32):
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()

    x = (R + r * np.cos(v)) * np.cos(u)
    y = (R + r * np.cos(v)) * np.sin(u)
    z = r * np.sin(v)

    verts = np.stack([x, y, z], axis=1).astype(np.float32)

    # Create triangle faces from grid
    faces = []
    for i in range(nv):
        for j in range(nu):
            i0 = i * nu + j
            i1 = i * nu + (j + 1) % nu
            i2 = ((i + 1) % nv) * nu + j
            i3 = ((i + 1) % nv) * nu + (j + 1) % nu
            faces.append([i0, i2, i1])
            faces.append([i1, i2, i3])
    faces = np.array(faces, dtype=np.int32)

    # Compute face normals
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(norms == 0, 1, norms)

    normals = -normals    # adjust for LHS

    return verts, faces, normals


def main():
    logging.basicConfig(level=logging.DEBUG)

    # ---
    sg = SpaceGraph()

    # ---
    sg.scatter(np.random.randn(1000, 3) + [5, 5, 0], color='xkcd:pinkish', size=5)

    # ---
    n = 100
    x = np.cos(np.linspace(0, 10 * np.pi, n))
    y = np.sin(np.linspace(0, 10 * np.pi, n))
    z = np.linspace(0, 3, n)
    position = np.column_stack([x + 5, y - 5, z])
    sg.plot(position, color='red')

    # ---
    x = np.cos(np.linspace(0, 4 * np.pi, n))
    y = np.sin(np.linspace(0, 4 * np.pi, n))
    z = np.linspace(0, 5, n)
    position = np.column_stack([x - 5, y + 5, z])
    sg.plot(position, color='xkcd:sky blue', linewidth=0.2)

    # ---
    sg.mesh(vertices=[[10, -10, 0],
                      [10, 10, 0],
                      [10, 10, 5],
                      [10, -10, 5]],
            faces=[[0, 1, 2],
                   [0, 2, 3]],
            color=[[1, 0, 1, 0.5],
                   [0, 1, 1, 0.5]])

    # ---
    sg.add_sphere(origin=[5, 0, 2])

    # ---
    image = skimage.data.chelsea()

    height, width, _ = image.shape
    aspect_ratio = width / height
    width = 2 * aspect_ratio
    height = 2

    M_OBJ_WCS = GazeAt(eye=[-3, -1, 5],
                       target=[5, 0, 0])
    sg.add_image(image, width, height, M_OBJ_WCS, alpha=0.8)
    sg.add_cframe(M_OBJ_WCS)

    # ---
    origin = np.array([[0, -5, 0],
                       [0, -5, 0],
                       [0, -5, 0],
                       [0, -5, 0]])

    direction = np.array([[-1, 1, 0],
                          [0, 1, 0],
                          [-2, 0, 0],
                          [1, -1, 1]])

    sg.add_arrow(origin=origin,
                 direction=direction,
                 color=['xkcd:magenta', 'xkcd:teal', 'xkcd:orange', 'xkcd:cerulean'])

    # ---
    verts, faces, normals = generate_torus()
    sg.lit_mesh(verts + [5, 0, 1], faces, normals)

    # ---
    sg.show()


if __name__ == '__main__':
    main()
