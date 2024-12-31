from typing import Optional, List

import numpy as np
from matplotlib.colors import to_rgba
import skimage

from gigagoop.types import ArrayLike


def false_color(input_colors: ArrayLike, ref_color: str | ArrayLike, alpha: Optional[float] = 1):
    """
    Transforms an array of RGB colors into a 'false color' representation using a specified reference color. This
    function converts the input colors to HSV, maintains their luminance (brightness), and then applies  the hue and
    saturation of the reference color. The result is an array of colors that have the same intensity as the original
    colors but are tinted with the reference color. Useful for visualizations where  color representation needs to be
    standardized or for creating artistic effects (like multi-view visualizations of two point clouds).
    """

    # The input should be [n, 3]
    input_colors = np.array(input_colors)
    m, n = input_colors.shape
    assert n == 3
    assert np.min(input_colors) >= 0
    assert np.max(input_colors) <= 1

    # Define the reference color that `input_colors` will be transformed to (such as if you want everything red)
    ref_rgb = to_rgba(ref_color)[:3]

    # Figure out the desired HSV value for this reference color
    ref_hsv = skimage.color.rgb2hsv(np.atleast_2d(ref_rgb))
    ref_h, ref_s, _ = ref_hsv[0]

    # Modulate the value based on the intensity of the RGB colors (the gray part)
    colors_gray = skimage.color.rgb2gray(input_colors)

    # Transform to HSV and mess with the value part to get our false color effect
    colors_hsv = np.zeros_like(input_colors)
    colors_hsv[:, 0] = ref_h
    colors_hsv[:, 1] = ref_s
    colors_hsv[:, 2] = colors_gray

    # Use the alpha value (for convenience)
    r, g, b = skimage.color.hsv2rgb(colors_hsv).T
    output_colors_rgba = np.column_stack([r, g, b, alpha * np.ones_like(r)])

    return output_colors_rgba


def get_vertex_rgba(vertices: ArrayLike,
                    color: str | ArrayLike = 'white',
                    alpha: Optional[float] = None) -> np.ndarray:
    """This is a flexible helper function to assign colors to the vertices, allowing a lot of different permutations of
    inputs for the assignment.
    """
    # The `vertices` should be stored as a tall matrix, where each row contains `[x, y, z`]
    vertices = np.array(vertices)
    m, n = vertices.shape
    assert n == 3

    # Check if the input is just a color string (i.e., 'red') and create a tall color matrix to match `vertices`
    if isinstance(color, str):
        if alpha is None:
            alpha = 1
        rgba = to_rgba(color, alpha=alpha)
        return np.tile(rgba, [m, 1])

    # If the input is a list of colors (i.e., `['red', 'green']`) then the number of string colors should represent the
    # number of rows in `vertices`, that is each row should have a color associated with it.
    if isinstance(color, List):
        if all([isinstance(x, str) for x in color]):
            assert len(color) == m
            rgba = np.array([to_rgba(x, alpha=alpha) for x in color])
            return rgba

    # If we get here, then `color` has either shape `[n, 3]` or `[n, 4]` depending on if the alpha value is packed
    color = np.array(color)

    if color.ndim == 1:
        color = np.atleast_2d(color)
        color = np.tile(color, [m, 1])

    assert color.ndim == 2

    cm, cn = color.shape
    assert cm == m, 'Both `color` and `vertices` should contain the same number of rows'
    assert cn in [3, 4]

    assert np.nanmin(color) >= 0.0
    assert np.nanmax(color) <= 1.0

    if cn == 4:
        rgba = color.copy()
        if alpha is not None:
            rgba[:, -1] = alpha    # override the alpha with the passed in one

        return rgba

    assert cn == 3
    rgb = color.copy()
    if alpha is None:
        alpha = 1.0
    rgba = np.c_[rgb, alpha * np.ones(m)]

    return rgba
