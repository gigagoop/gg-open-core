"""
This example demonstrates how to use the `SpaceGraph` interface.
"""
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage

from gigagoop.viz import SpaceGraph
from gigagoop.coord import GazeAt

matplotlib.use('Qt5Agg')
plt.ion()

log = logging.getLogger('tour')


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
    sg.show()


if __name__ == '__main__':
    main()
