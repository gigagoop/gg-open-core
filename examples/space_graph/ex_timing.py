"""
Just demonstrate how long it takes to start the viewer.
"""
import logging
from time import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from gigagoop.viz import SpaceGraph

matplotlib.use('Qt5Agg')
plt.ion()

log = logging.getLogger('tour')


def main():
    logging.basicConfig(level=logging.DEBUG)

    t_start = time()

    sg = SpaceGraph()
    sg.scatter(np.random.randn(1000, 3) + [5, 5, 0], color='xkcd:pinkish', size=5)
    sg.close()

    t_elapsed = time() - t_start
    log.info(f'{t_elapsed=:0.6f} seconds')


if __name__ == '__main__':
    main()
