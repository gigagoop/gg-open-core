from __future__ import annotations
import logging
from math import sin, cos
import re
from typing import overload, List
from textwrap import dedent
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from gigagoop.types import ArrayLike
from gigagoop.viz.space_graph.space_graph import check_position

log = logging.getLogger(__name__)


class Transform:
    """
    The Transform class offers a convenient way of defining coordinate frames, and mapping points, vectors, and normals
    from one frame to another. Under the hood, the transform is really just a 4x4 matrix (see [1] and [2]), but we spice
    up this matrix in an object-oriented form with some convenience methods.

    Common Notation
    ---------------
    We (typically) utilize a common notation when expressing coordinate frame transformations, that you will see
    throughout the code that looks like `M_OBJ_WCS`. To make sense of this, we start by defining two coordinate systems:
        WCS (World Coordinate System)
            This is the parent coordinate frame that the entire scene lives in. All objects defined in this scene will
            have coordinate systems that are related somehow to this frame. As an example, Unreal Engine defines an
            LHS (left-handed system) with `Z` up, while Blender defines a (right-handed system) with `Z` up.
        OBJ (Object Coordinate System)
            This frame is local to a given object. As an example, perhaps you place a camera into an Unreal Engine
            scene (a map) - this camera would have its own frame.

    The frame can be interpreted as either:
        * WCS to OBJ transform (map a point from WCS to OBJ)
        * WCS coordinate frame defined w.r.t. OBJ coordinates

    Mapping
    -------
    Given `p_obj = [1, 0, 0]` a point defined w.r.t. OBJ, we can map it to the world in any of the following equivalent
    forms:
    ```
    p_wcs = M_OBJ_WCS(p_obj)    # functional form
    p_wcs = M_OBJ_WCS @ p_obj   # matrix multiply form
    p_wcs = M_OBJ_WCS * p_obj   # matrix multiply form
    ```

    which follows the convention of "3-tuple in, 3-tuple out".

    If many points needs to be transformed, we use the convention of "tall-matrix in, tall-matrix out". As an example,
    consider `P_obj = np.random.randn(100, 3)`, or one hundred random points defined w.r.t. OBJ transformed to WCS, can
    be transformed to WCS with any of the forms listed above, e.g., `P_WCS = M_OBJ_WCS * P_OBJ`.

    Chaining
    --------
    For complicated objects, it is common for it to be comprised of numerous frames defined conveniently with respect
    to (w.r.t.) child objects. As an example, we could define `M_CAR_WCS` a coordinate frame of a car w.r.t. the world
    frame. The car could then have children frames like `M_FLT_CAR`, `M_FRT_CAR`, `M_BLT_CAR`, `M_BRT_CAR` for
    front-left tire, front-right tire, back-left tire, and back-right tire respectively. Through the use of "chaining"
    we can always get back to the original world frame. As an example, the front-left tire expressed w.r.t. the world,
    is given as `M_FLT_WCS = M_CAR_WCS * M_FLT_CAR`

    Inverse
    -------
    Given a transform `M_OBJ_WCS`, we can easily find the inverse transform as: `M_WCS_OBJ = M_OBJ_WCS.inverse`.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Transformation_matrix
    [2] https://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Transformations
    """
    name: Optional[str] = None

    @overload
    def __init__(self) -> None:
        """Create an identity Transform. """
        ...

    @overload
    def __init__(self, matrix: ArrayLike) -> None:
        """Create the Transform with a 4x4 matrix."""
        ...

    def __init__(self, matrix=None) -> None:
        if matrix is None:
            matrix = np.eye(4)

        matrix = np.array(matrix).astype(float)

        assert matrix.ndim == 2
        assert matrix.shape == (4, 4)

        self._matrix = matrix

    def __eq__(self, other: Transform) -> bool:
        """Check if two transforms are equal."""
        return np.all(np.isclose(self.matrix, other.matrix))

    def copy(self) -> Transform:
        """Return a deep copy."""
        return Transform(self.matrix.copy())

    @staticmethod
    def from_rotation(rotation: ArrayLike) -> Transform:
        rotation = np.array(rotation)

        assert rotation.ndim == 2
        assert rotation.shape == (3, 3)

        # Check if a rotation matrix was actually passed in
        det = np.linalg.det(rotation)
        if not np.all(np.isclose(rotation.T, np.linalg.inv(rotation))) and np.isclose(det, 1):
            log.warning(f'R does not appear to be a rotation matrix, {det=:0.6f}')

        origin = np.zeros(3)
        matrix = np.r_[np.c_[rotation, origin], [[0, 0, 0, 1.0]]]
        transform = Transform(matrix)

        return transform

    @staticmethod
    def from_origin(origin: ArrayLike) -> Transform:
        origin = np.array(origin).flatten()

        assert len(origin) == 3

        rotation = np.eye(3)
        matrix = np.r_[np.c_[rotation, origin], [[0, 0, 0, 1.0]]]
        transform = Transform(matrix)

        return transform

    @staticmethod
    def from_rotation_and_origin(rotation: ArrayLike, origin: ArrayLike) -> Transform:
        rotation = np.array(rotation)
        origin = np.array(origin).flatten()

        assert rotation.ndim == 2
        assert rotation.shape == (3, 3)
        assert len(origin) == 3

        matrix = np.r_[np.c_[rotation, origin], [[0, 0, 0, 1.0]]]
        transform = Transform(matrix)

        return transform

    @staticmethod
    def from_point_and_direction(point: ArrayLike, direction: ArrayLike) -> Transform:
        """Create a Transform object from a point and a direction. Say you have a point on the ground, and the up
        vector, but you want to know the basis of the ground plane, you could use this."""
        point = np.array(point)
        direction = np.array(direction)

        assert len(point) == 3
        assert len(direction) == 3

        Z = direction / np.linalg.norm(direction)

        #  Find a vector that is not parallel to Z for cross product
        if np.allclose(Z, [1, 0, 0]) or np.allclose(Z, [-1, 0, 0]):
            V = np.array([0, 1, 0])
        else:
            V = np.array([1, 0, 0])

        X = np.cross(Z, V)
        X /= np.linalg.norm(X)

        Y = np.cross(Z, X)
        Y /= np.linalg.norm(Y)

        O = point

        M_OBJ_WCS = Transform.from_rotation_and_origin(origin=O, rotation=np.c_[X, Y, Z])

        return M_OBJ_WCS

    @staticmethod
    def from_unreal(location: Optional[ArrayLike] = None,
                    rotation: Optional[ArrayLike] = None,
                    scale: Optional[ArrayLike] = None) -> Transform:
        """Create a Transform object from the provided location, rotation, and scale parameters following the Unreal
        Engine SRT order (Scale, Rotation, Translation) in the Transform palette.
        """
        if location is None:
            location = np.zeros(3)

        if rotation is None:
            rotation = np.zeros(3)

        if scale is None:
            scale = np.ones(3)

        def check_3tuple(arr):
            arr = np.array(arr).astype(float).flatten()
            assert len(arr) == 3
            return arr

        location = check_3tuple(location)
        rotation = check_3tuple(rotation)
        scale = check_3tuple(scale)

        if np.all(np.isclose(scale, [0, 0, 0])):
            log.warning(f'scale values should not be near zero, {scale=}')

        S = Scale(*scale)

        theta_x, theta_y, theta_z = rotation
        R = RotateZ(theta_z) * RotateY(theta_y) * RotateX(theta_x)

        T = Translate(*location)

        transform = T * R * S

        return transform

    def to_unreal(self) -> Tuple[NDArray, NDArray]:

        R0 = self.rotation

        X = R0[:, 0]
        Y = -R0[:, 1]
        Z = R0[:, 2]

        R1 = np.c_[X, -Y, Z]

        r = Rotation.from_matrix(R1)
        rot_x, rot_y, rot_z = r.as_euler('xyz', degrees=True)

        location = self.origin
        rotation = np.array([-rot_x, -rot_y, rot_z])

        return location, rotation

    def __repr__(self):
        m = self._matrix

        repr = dedent(f"""\
                   Transform([[ {m[0][0]:0.3f}, {m[0][1]:0.3f}, {m[0][2]:0.3f}, {m[0][3]:0.3f} ]
                              [ {m[1][0]:0.3f}, {m[1][1]:0.3f}, {m[1][2]:0.3f}, {m[1][3]:0.3f} ]
                              [ {m[2][0]:0.3f}, {m[2][1]:0.3f}, {m[2][2]:0.3f}, {m[2][3]:0.3f} ]
                              [ {m[3][0]:0.3f}, {m[3][1]:0.3f}, {m[3][2]:0.3f}, {m[3][3]:0.3f} ]])""")

        if self.name is None:
            return repr
        else:
            return f'=== {self.name} ===\n{repr}'

    def __sub__(self, other: Transform) -> Transform:
        """Compute the relative transform between two transforms.

        It is often interesting to compare "the difference" between two transforms, which is what we do here. As a
        motivating example, if we had two points `p_a` and `p_b`, we could define `v = p_a - p_b`, where `v` would be
        a vector that points from `b` to `a`.

        In our case, assume we have two poses `M_CAM1_WCS` and `M_CAM2_WCS`, if we write `V = M_CAM2_WCS - M_CAM1_WCS`
        we want `V` to represent the transform that maps from `CAM1` to `CAM2` as expressed w.r.t. the `CAM1` frame. We
        can then look at something like `V.origin` to see how far away `CAM2` is from `CAM1`.
        """
        M_CAM2_WCS = self
        M_CAM1_WCS = other

        M_WCS_CAM1 = M_CAM1_WCS.inverse
        M_CAM2_CAM1 = M_WCS_CAM1 @ M_CAM2_WCS

        return M_CAM2_CAM1

    @property
    def matrix(self) -> NDArray:
        """Return the underlying 4x4 matrix that makes up the transform."""
        return self._matrix

    @property
    def has_scale(self) -> bool:
        """Check if a transformation has scaling."""
        return not np.isclose(np.linalg.norm(self.x_axis), 1) or \
               not np.isclose(np.linalg.norm(self.y_axis), 1) or \
               not np.isclose(np.linalg.norm(self.z_axis), 1)

    @property
    def inverse(self) -> Transform:
        """Return the inverse transform."""
        T = Transform(np.linalg.inv(self._matrix))

        name = self.name
        if name is not None:
            # Check if the name has a form like `M_CAM_WCS`
            pattern = r'^([A-Za-z])_([A-Za-z]+)_([A-Za-z]+)$'
            match = re.match(pattern, name)
            if match:
                X, Y, Z = match.groups()
                inverted_name = f'{X}_{Z}_{Y}'
                T.name = inverted_name
            else:
                log.warning(f'Transform name={name} does not follow a standard convention like M_CAM_WCS, so the name'
                            'will not be inverted automatically')

        return T

    @property
    def x_axis(self) -> NDArray:
        """Return the x-axis of the frame."""
        return self._matrix[:3, 0]

    @property
    def y_axis(self) -> NDArray:
        """Return the y-axis of the frame."""
        return self._matrix[:3, 1]

    @property
    def z_axis(self) -> NDArray:
        """Return the z-axis of the frame."""
        return self._matrix[:3, 2]

    @property
    def origin(self) -> NDArray:
        """Return the origin of the frame."""
        return self._matrix[:3, 3]

    @origin.setter
    def origin(self, value: ArrayLike):
        value = np.array(value).flatten()
        assert len(value) == 3
        self._matrix[:3, 3] = value

    @property
    def rotation(self) -> NDArray:
        """Return the rotational component (a 3x3 matrix) from the transform. """
        return self._matrix[:3, :3]

    def tolist(self) -> List:
        return self._matrix.tolist()

    def __eq__(self, other: Transform) -> bool:
        """Return True when both transforms are equal."""
        return np.all(np.isclose(self.matrix, other.matrix))

    def __call__(self, points: ArrayLike) -> NDArray:
        """Map points from one frame to another."""
        return self._transform(points)

    def __mul__(self, other: ArrayLike | Transform) -> NDArray | Transform:
        """Map points from one frame to another, or apply chaining."""
        if isinstance(other, Transform):
            return self._chain(other)
        else:
            return self._transform(other)

    def __matmul__(self, other: ArrayLike | Transform) -> NDArray | Transform:
        """Map points from one frame to another, or apply chaining."""
        return self.__mul__(other)

    def _chain(self, other: Transform) -> Transform:
        # This is being called from `__mul__` from an order like `self * other`, so we need to multiply our matrices
        # the same way
        return Transform(self.matrix @ other.matrix)

    def _transform(self, points):
        points = np.array(points).astype(float)

        # In the case that points is just a 3-tuple, we handle it differently and return a 3-tuple back
        if points.ndim == 1:
            return self.rotation @ points + self.origin

        # For all other cases, we assume a tall matrix form, such that each row represents a point
        assert points.ndim == 2

        num_points, num_coords = points.shape
        assert num_coords == 3, 'input should be a tall matrix [n x 3]'

        # Take note that `points` is a tall matrix of shape `[n x 3]`. The rotation convention would mean, we need to
        # do something like:
        #
        #     Q = self.rotation @ points.T + self.origin[:, np.newaxis]
        #
        # Taking note that `Q` would now have shape `[3 x n]` we would need to return `Q.T` which would be yet another
        # transpose. In the case that `points` is large, skipping these transposes is probably worthwhile... so we will
        # do a right multiply instead.

        return points @ self.rotation.T + self.origin[np.newaxis, :]

    def apply_to_vectors(self, vectors: ArrayLike) -> NDArray:
        """
        Map vectors from one coordinate frame to another.
        """
        R = self.rotation
        vectors = check_position(vectors)

        new_vectors = vectors @ R.T
        assert new_vectors.shape == vectors.shape

        return new_vectors


def Translate(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Transform:
    """Construct a `Translate Transform`.

    A translation transformation when applied to a point `p` translates `p's` coordinates by `dx`, `dy`, and `dz`.
    """
    return Transform.from_origin([dx, dy, dz])


def Scale(sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> Transform:
    return Transform.from_rotation(np.diag([sx, sy, sz]))


def RotateX(angle: float = 0) -> Transform:
    """Construct a `RotateX Transform`

    The x rotation transformation when applied to a vector `v` rotates it about the `x` axis by `angle` degrees.

    Notes
    -----
    [1] Rotations follow the conventions of `https://en.wikipedia.org/wiki/Rotation_matrix`
    """
    theta = -np.deg2rad(angle)
    R = [[1, 0,           0],
         [0, cos(theta), -sin(theta)],
         [0, sin(theta),  cos(theta)]]

    return Transform.from_rotation(R)


def RotateY(angle: float = 0) -> Transform:
    """Construct a `RotateY Transform`

    The y rotation transformation when applied to a vector `v` rotates it about the `y` axis by `angle` degrees.

    Notes
    -----
    [1]  Rotations follow the conventions of `https://en.wikipedia.org/wiki/Rotation_matrix`
    """
    theta = -np.deg2rad(angle)
    R = [[cos(theta),  0, sin(theta)],
         [0,           1, 0],
         [-sin(theta), 0, cos(theta)]]

    return Transform.from_rotation(R)


def RotateZ(angle: float = 0) -> Transform:
    """Construct a `RotateZ Transform`

    The z rotation transformation when applied to a vector `v` rotates it about the `z` axis by `angle` degrees.

    Notes
    -----
    [1]  Rotations follow the conventions of `https://en.wikipedia.org/wiki/Rotation_matrix`
    """
    theta = np.deg2rad(angle)
    R = [[cos(theta), -sin(theta), 0],
         [sin(theta),  cos(theta), 0],
         [0,           0,          1]]

    return Transform.from_rotation(R)


def GazeAt(eye: ArrayLike, target: ArrayLike, up: Optional[ArrayLike] = None) -> Transform:
    """Returns a camera pose that looks from `eye`, at `target`, with a roll oriented by `up`.

    The standard "look at" transform (e.g. see the GLM definition) places the camera at `eye`, points it at
    `target`, and orients the roll with help from the `up` vector, creates a transform, and then *inverts it*,
    returning a transform that maps from `WCS` to `CAM`. In other words, the standard look at transform returns a
    mapping from the world (WCS) to the camera (CAM), *not* the camera pose defined w.r.t. WCS. This function
    however, actually returns the camera pose from `eye`, "gazing at" ;-) `eye`, oriented by `up`.

    When returning a camera pose, we now need a convention on the basis. The convention that we follow here is
    Unreal Engine's left-handed coordinate system with:
        UE (LHS)
            X: forward
            Y: right
            Z: up

    Warning
    -------
    [1] This is similar to a look at transform, but not quite the same (read the docstring ;-P)
    [2] This returns a LHS (read the docstring ;-P)
    """
    eye = np.array(eye).flatten()
    target = np.array(target).flatten()

    if up is None:
        up = np.array([0, 0, 1])

    up = np.array(up).flatten()

    assert len(eye) == 3
    assert len(target) == 3
    assert len(up) == 3

    X = target - eye
    X = X / np.linalg.norm(X)

    Z = up / np.linalg.norm(up)

    Y = np.cross(X, -Z)  # assumes LHS
    Y = Y / np.linalg.norm(Y)

    Z = np.cross(X, Y)  # assumes LHS
    Z = Z / np.linalg.norm(Z)

    M_CAM_WCS = Transform.from_rotation_and_origin(rotation=np.c_[X, Y, Z],
                                                   origin=eye)

    return M_CAM_WCS


def LookAt(eye: ArrayLike, target: ArrayLike, up: ArrayLike) -> Transform:
    """The standard look at transform, following the UE LHS convention (see `from_gaze_at`)."""
    M_CAM_WCS = GazeAt(eye, target, up)
    M_WCS_CAM = M_CAM_WCS.inverse

    return M_WCS_CAM

#
# def _verify_uecam_and_glcam(M_LCAM_WCS, M_RCAM_WCS):
#     # To show that this works, recall that each convention is given by
#     #         UE (LHS)
#     #             X: forward
#     #             Y: right
#     #             Z: up
#     #
#     #         OpenGL (RHS)
#     #             X: right
#     #             Y: up
#     #             Z: -forward
#
#     # It is now straightforward to verify those conventions as
#     forward = M_LCAM_WCS.matrix[:, 0]
#     right = M_LCAM_WCS.matrix[:, 1]
#     up = M_LCAM_WCS.matrix[:, 2]
#     origin = M_LCAM_WCS.matrix[:, 3]
#
#     assert np.all(np.isclose(right, M_RCAM_WCS.matrix[:, 0]))
#     assert np.all(np.isclose(up, M_RCAM_WCS.matrix[:, 1]))
#     assert np.all(np.isclose(-forward, M_RCAM_WCS.matrix[:, 2]))
#     assert np.all(np.isclose(origin, M_RCAM_WCS.matrix[:, 3]))


# def uecam_to_glcam(M_CAM_WCS: Transform) -> Transform:
#     """Transform a camera defined in the UE LHS convention to the OpenGL convention.
#     """
#     # The input camera is defined in the UE LHS convention
#     M_LCAM_WCS = M_CAM_WCS
#
#     # The transformation to the OpenGL convention is given by
#     M_LHS_RHS = Transform.get_lhs_to_rhs()
#     M_RCAM_WCS = M_LCAM_WCS * M_LHS_RHS.inverse
#     _verify_uecam_and_glcam(M_LCAM_WCS, M_RCAM_WCS)
#
#     return M_RCAM_WCS
#
#
# def glcam_to_uecam(M_CAM_WCS: Transform) -> Transform:
#     M_RCAM_WCS = M_CAM_WCS
#
#     M_LHS_RHS = Transform.get_lhs_to_rhs()
#     M_LCAM_WCS = M_RCAM_WCS * M_LHS_RHS
#     _verify_uecam_and_glcam(M_LCAM_WCS, M_RCAM_WCS)
#
#     return M_LCAM_WCS


def get_world_coordinate_system(kind: str):
    """Get a world coordinate system w.r.t. WCS.

    GigaGoop utilizes a world coordinate system convention of:
        Z Up
        X Forward
        Left-handed

    which we refer to as `WCS`

    If your coordinates are defined w.r.t. `WCS`, great, there is nothing to do, just plot them ;-P - however... it is
    very common that this is not the case, and you will be presented with coordinates in a different frame. To transform
    to and from the different conventions, you can use this function to get a `Transform` to do this job.

    Supported Coordinate Systems
    ----------------------------
    The supported WCS conventions are given by:
        Name            Kind        Symbol
        -----------     ----        ------
        GigaGoop        gg          WCS
        Unreal Engine   ue          UEWCS
        Blender         blender     BLWCS
        OpenGL          opengl      GLWCS
        ARKit           arkit       ARWCS
        OpenCV          opencv      CVWCS
    """
    kind = str(kind).lower()

    if kind in ['gg', 'ue']:
        M_WCS_WCS = Transform()    # equivalent to `M_UEWCS_WCS`

        return M_WCS_WCS

    if kind in ['blender']:
        M_BLWCS_WCS = Transform([[0, 1, 0, 0],
                                 [1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])

        return M_BLWCS_WCS

    if kind in ['opengl', 'arkit']:
        M_GLWCS_WCS = Transform([[0, 0, -1, 0],
                                 [1, 0,  0, 0],
                                 [0, 1,  0, 0],
                                 [0, 0,  0, 1]])    # equivalent to `M_ARWCS_WCS`

        return M_GLWCS_WCS

    if kind in ['opencv']:
        M_CVWCS_WCS = Transform([[0, 0,  1, 0],
                                 [1, 0,  0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0,  0, 1]])

        return M_CVWCS_WCS

    raise NotImplementedError(f'{kind=} not supported')


def get_camera_coordinate_system(kind: str):
    """Get a camera coordinate system w.r.t. CAM.

    GigaGoop utilizes a camera coordinate system convention of:
        Z Up
        X Forward
        Left-handed

    which we refer to as `CAM`

    Just like `get_world_coordinate_system` we will often need to transform between different camera conventions - use
    this function to do that.

    Supported Coordinate Systems
    ----------------------------
    The supported WCS conventions are given by:
        Name            Kind        Symbol
        -----------     ----        ------
        GigaGoop        gg          CAM
        Unreal Engine   ue          UECAM
        Blender         blender     BLCAM
        OpenGL          opengl      GLCAM
        OpenCV          opencv      CVCAM
        ARKit           arkit       ARCAM
    """
    kind = str(kind).lower()

    if kind in ['gg', 'ue']:
        M_CAM_CAM = Transform()    # equivalent to `M_UECAM_CAM`
        return M_CAM_CAM

    if kind in ['opencv']:
        M_CVCAM_CAM = Transform([[0,  0, 1, 0],
                                 [1,  0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0,  0, 0, 1]])

        return M_CVCAM_CAM

    if kind in ['opengl', 'blender', 'arkit']:
        M_GLCAM_CAM = Transform([[0, 0, -1, 0],
                                 [1, 0,  0, 0],
                                 [0, 1,  0, 0],
                                 [0, 0,  0, 1]])    # equivalent to `M_BLCAM_CAM` and `M_ARCAM_CAM`

        return M_GLCAM_CAM

    raise NotImplementedError(f'{kind=} not supported')
