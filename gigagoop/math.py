from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from gigagoop.types import ArrayLike


def angle_between(v1: ArrayLike, v2: ArrayLike) -> float:
    """Returns the angle between the vectors `v1` and `v2` in degrees.
    """
    v1 = np.array(v1).astype(float)
    v2 = np.array(v2).astype(float)
    assert len(v1) == len(v2)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    phi = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return np.rad2deg(phi)


def line_plane_intersection(plane_normal: ArrayLike, plane_point: ArrayLike, ray_direction: ArrayLike,
                            ray_point: ArrayLike, epsilon: float = 0.000001, return_dist=False):

    plane_normal = np.array(plane_normal).astype(float)
    plane_point = np.array(plane_point).astype(float)
    ray_direction = np.array(ray_direction).astype(float)
    ray_point = np.array(ray_point).astype(float)

    assert len(plane_normal) == 3
    assert len(plane_point) == 3
    assert len(ray_direction) == 3
    assert len(ray_point) == 3

    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    ray_direction = ray_direction / np.linalg.norm(ray_direction)

    s = plane_normal.dot(ray_direction)
    if abs(s) < epsilon:
        return None

    d = ray_point - plane_point
    t = -plane_normal.dot(d) / s
    intersected_point = d + t*ray_direction + plane_point

    if not return_dist:
        return intersected_point
    else:
        return intersected_point, t


def ray_triangle_intersection(ray_origin: ArrayLike,
                              ray_direction: ArrayLike,
                              triangle_vertices: ArrayLike) -> Optional[NDArray]:
    """Find the intersection point of a ray and a triangle, returning `None` if no intersection exists."""

    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)
    triangle_vertices = np.array(triangle_vertices)

    # Compute the edges of the triangle
    edge1 = triangle_vertices[1] - triangle_vertices[0]
    edge2 = triangle_vertices[2] - triangle_vertices[0]

    # Compute the cross product of the ray direction and edge2
    h = np.cross(ray_direction, edge2)

    # Calculate the determinant
    a = np.dot(edge1, h)

    # Check if the ray is parallel to the triangle
    if abs(a) < 1e-8:
        return None

    # Calculate the inverse of the determinant
    inv_a = 1 / a

    # Calculate the vector from the ray origin to the first vertex of the triangle
    s = ray_origin - triangle_vertices[0]

    # Compute the first barycentric coordinate (u)
    u = np.dot(s, h) * inv_a

    # Check if the intersection is outside the triangle
    if u < 0 or u > 1:
        return None

    # Compute the cross product of s and edge1
    q = np.cross(s, edge1)

    # Compute the second barycentric coordinate (v)
    v = np.dot(ray_direction, q) * inv_a

    # Check if the intersection is outside the triangle
    if v < 0 or u + v > 1:
        return None

    # Compute the intersection distance (t)
    t = np.dot(edge2, q) * inv_a

    # Check if the intersection is behind the ray origin
    if t < 1e-8:
        return None

    # Compute the intersection point
    intersection_point = ray_origin + t * ray_direction

    return intersection_point


def plane_plane_intersection(plane1_normal: ArrayLike,
                             plane1_point: ArrayLike,
                             plane2_normal: ArrayLike,
                             plane2_point: ArrayLike) -> Optional[Tuple[NDArray, NDArray]]:
    """

    """
    plane1_normal = np.array(plane1_normal).astype(float)
    plane1_point = np.array(plane1_point).astype(float)
    plane2_normal = np.array(plane2_normal).astype(float)
    plane2_point = np.array(plane2_point).astype(float)

    assert len(plane1_normal) == 3
    assert len(plane1_point) == 3
    assert len(plane2_normal) == 3
    assert len(plane2_point) == 3

    # Find the direction vector of the intersection line
    direction = np.cross(plane1_normal, plane2_normal)

    # Check if the planes are parallel
    if np.allclose(direction, 0):
        return None

    # Form augmented matrix for the system of equations to find a point on the intersection line
    A = np.vstack((plane1_normal, plane2_normal))
    B = np.array([np.dot(plane1_normal, plane1_point), np.dot(plane2_normal, plane2_point)])

    # Solve the system of equations using least-squares (to minimize error)
    point_on_line, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)

    return point_on_line, direction


def line_line_intersection(point1: ArrayLike,
                           direction1: ArrayLike,
                           point2: ArrayLike,
                           direction2: ArrayLike,
                           return_dist: bool = False):

    point1 = np.array(point1).astype(float)
    direction1 = np.array(direction1).astype(float)
    point2 = np.array(point2).astype(float)
    direction2 = np.array(direction2).astype(float)

    assert len(point1) == 3
    assert len(direction1) == 3
    assert len(point2) == 3
    assert len(direction2) == 3

    direction1 = direction1 / np.linalg.norm(direction1)
    direction2 = direction2 / np.linalg.norm(direction2)

    A = np.vstack([direction1, -direction2]).T
    b = point2 - point1

    t_s, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    if rank < 2:
        return None    # lines are parallel

    t, s = t_s

    intersection_point1 = point1 + t * direction1
    intersection_point2 = point2 + s * direction2

    if np.allclose(intersection_point1, intersection_point2):
        if return_dist:
            return intersection_point1, t
        else:
            return intersection_point1  # lines intersect

    else:
        return None    # lines do not intersect


def unitsratio(from_unit: str, to_unit: str) -> float:
    """Conversion factor between units.

    As an example, 1 meter is equal to 1000 mm, so we can use
        m_to_mm = unitsratio('m', 'mm')
        assert m_to_mm == 1000

    A good reference for conversions is `https://www.rapidtables.com/convert/length/mm-to-inch.html` - please feel free
    to add to the `conversions` dict below if you need something ;-P.
    """
    conversions = {('m', 'mm'): 1000,
                   ('mm', 'm'): 0.001,
                   ('m', 'ft'): 1/0.3048,
                   ('ft', 'm'): 0.3048,
                   ('in', 'mm'): 25.4,
                   ('mm', 'in'): 0.0393700787,
                   ('cm', 'm'): 0.01,
                   ('m', 'cm'): 100,
                   ('m', 'in'): 39.3700787402,
                   ('in', 'm'): 0.0254,
                   ('ms', 's'): 0.001,
                   ('s', 'ms'): 1000,
                   ('ft', 'in'): 12,
                   ('in', 'ft'): 1/12}

    assert (from_unit, to_unit) in conversions, f'({from_unit=}, {to_unit=}) should exist in conversions'

    return conversions[(from_unit, to_unit)]


def check_if_rotation_matrix(R: ArrayLike) -> bool:
    """
    Return true if `R` is a rotation matrix
    """
    R = np.array(R)

    is_square = R.shape == (3, 3)  # check for square and 3x3 (which is all we really care about...)
    is_orthogonal = np.allclose(R @ R.T - np.eye(3), 0)
    has_det_one = np.isclose(np.linalg.det(R), 1)

    return is_square and is_orthogonal and has_det_one


def generate_random_rotation_matrix() -> NDArray:
    """
    Generate a random rotation matrix.
    """
    R, _ = np.linalg.qr(np.random.randn(3, 3))
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]

    assert check_if_rotation_matrix(R)

    return R
