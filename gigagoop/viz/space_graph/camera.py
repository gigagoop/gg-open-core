import time
from typing import Optional

import numpy as np
from moderngl_window.context.base import BaseKeys
from moderngl_window.context.base.keys import KeyModifiers

from gigagoop.coord import Transform, get_world_coordinate_system
from gigagoop.camera import PinholeCamera


class Camera:
    NEAR_PLANE = 0.01
    FAR_PLANE = 100

    def __init__(self, cam: PinholeCamera, keys: Optional[BaseKeys] = None, lock_height: bool = True):

        self._keys = keys

        # View Matrix
        # -----------
        # The view matrix is the mapping from WCS to CAM. For many applications it is more convenient to talk about the
        # camera pose which is the mapping from CAM to WCS, or put another way, the camera frame defined w.r.t. WCS. We
        # will store the camera pose and just get the view matrix by inverting the pose.
        self._M_CAM_WCS = Transform()
        self._rotation = np.zeros(3)

        # Projection Matrix
        # -----------------
        # The model, view, projection matrix does the following:
        #
        #     x_obj ---> [model] ---> x_wcs --> [view] ---> x_cam ---> [projection] ---> x_img
        #
        # This means we can interpret
        #     model:        M_OBJ_WCS
        #     view          M_WCS_CAM
        #     projection    M_CAM_IMG
        #
        # In our case, we define WCS to be the UE LHS, so we cannot "just" use things like `glm.perspectiveFov` to get
        # our perspective matrix, as this follows the OpenGL conventions. To fix up things to "just work" ;-) for our
        # case we take note that the `x_cam` are points defined w.r.t. LHS, so we transform them to RHS, and then feed
        # them into the standard OpenGL perspective matrix (http://www.songho.ca/opengl/gl_projectionmatrix.html).
        #
        # All this mumbo jumbo allows us to set our world frame as the UE LHS one...

        # The definition of the projection matrix takes a bit of work and his evolved over time. It is helpful for
        # future reading to see this progression, so we will keep note of by documenting two methods. The two methods
        # (`symmetric` and `non-symmetric`) are documented nicely in [1], so read that for an explanation. Moving
        # forward, we can support a general OpenCV calibration matrix (without skew though) with the non-symmetric
        # form that is listed below.

        # References
        # ----------
        # [1] https://jsantell.com/3d-projection
        # [2] http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
        # [3] https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL

        method = 'non-symmetric'
        self.lock_height = lock_height

        fx = cam.focal_length_x
        fy = cam.focal_length_y
        cx = cam.optical_center_x + 0.5  # center-origin to corner-origin
        cy = cam.optical_center_y + 0.5  # center-origin to corner-origin
        f = self.FAR_PLANE
        n = self.NEAR_PLANE
        width = cam.width
        height = cam.height

        assert method in ['symmetric', 'non-symmetric']

        if method == 'symmetric':
            import glm

            # The original implementation utilized the following implementation
            A = Transform(glm.perspectiveFov(glm.radians(cam.fov_vertical),
                                             width,
                                             height,
                                             self.NEAR_PLANE,
                                             self.FAR_PLANE))

            # From [2], an implementation is provided that directly utilizes the intrinsics...

            # As pointed out (and tested below), this implementation assumes the optical axis is at the center of the
            # image. For our intrinsics matrix, we follow the OpenCV convention which utilizes a `center-origin` - this
            # means we need to convert to the OpenGL convention of `corner-origin` with a `+0.5` shift.
            B = Transform([[fx / cx, 0,       0,                  0],
                           [0,       fy / cy, 0,                  0],
                           [0,       0,       -(f + n) / (f - n), -2 * f * n / (f - n)],
                           [0,       0,       -1,                 0]])

            # Take note that this derivation assumes symmetry, such that
            assert np.isclose(cx, width / 2)
            assert np.isclose(cy, height / 2)

            # Verify that both implementations match
            assert np.linalg.norm(A.matrix - B.matrix, np.inf) < 10 ** -6

            M_RCAM_IMG = B

        elif method == 'non-symmetric':
            # This implementation follows [3]
            C = Transform([[2*fx/width, 0,           (width - 2*cx)/width,    0],
                           [0,          2*fy/height, (-height + 2*cy)/height, 0],
                           [0,          0,           -(f + n)/(f - n),        -2 * f * n / (f - n)],
                           [0,          0,           -1,                      0]])

            M_RCAM_IMG = C

        else:
            raise NotImplementedError()

        M_GLWCS_WCS = get_world_coordinate_system('opengl')
        M_WCS_GLWCS = M_GLWCS_WCS.inverse

        self._M_CAM_IMG = M_RCAM_IMG * M_WCS_GLWCS

        # Movement Controls
        # -----------------
        self._last_time = 0.0
        self._last_rot_time = 0.0
        self._velocity = 10.0
        self._mouse_sensitivity = 0.5
        self._move_forward_backward = None
        self._move_left_right = None
        self._move_up_down = None

    @property
    def M_CAM_WCS(self) -> Transform:
        """Get the camera pose."""
        return self._M_CAM_WCS

    @M_CAM_WCS.setter
    def M_CAM_WCS(self, transform: Transform) -> None:
        """Set the camera pose."""
        self._M_CAM_WCS = transform

        # The location is already stored in `M_CAM_WCS` as `M_CAM_WCS.origin`; however, we need to unpack the rotation
        # parameterization and store that as well.
        _, rotation = self._M_CAM_WCS.to_unreal()
        self._rotation = rotation

    @property
    def projection_matrix(self) -> Transform:
        return self._M_CAM_IMG

    @property
    def view_matrix(self) -> Transform:
        now = time.time()
        t = max(now - self._last_time, 0)
        self._last_time = now

        M_CAM_WCS = self._M_CAM_WCS

        if self._move_forward_backward:
            move = self._move_forward_backward * self._velocity * t
            M_CAM_WCS.origin[0] += M_CAM_WCS.x_axis[0] * move
            M_CAM_WCS.origin[1] += M_CAM_WCS.x_axis[1] * move
            if not self.lock_height:
                M_CAM_WCS.origin[2] += M_CAM_WCS.x_axis[2] * move

        if self._move_left_right:
            move = self._move_left_right * self._velocity * t
            M_CAM_WCS.origin[0] += M_CAM_WCS.y_axis[0] * move
            M_CAM_WCS.origin[1] += M_CAM_WCS.y_axis[1] * move
            if not self.lock_height:
                M_CAM_WCS.origin[2] += M_CAM_WCS.y_axis[2] * move

        if self._move_up_down:
            M_CAM_WCS.origin[2] += self._move_up_down * self._velocity * t

        M_WCS_CAM = M_CAM_WCS.inverse
        return M_WCS_CAM

    @property
    def loc_x(self) -> float:
        return self._M_CAM_WCS.origin[0]

    @loc_x.setter
    def loc_x(self, value: float) -> None:
        self._M_CAM_WCS.origin[0] = value

    @property
    def loc_y(self) -> float:
        return self._M_CAM_WCS.origin[1]

    @loc_y.setter
    def loc_y(self, value: float) -> None:
        self._M_CAM_WCS.origin[1] = value

    @property
    def loc_z(self) -> float:
        return self._M_CAM_WCS.origin[2]

    @loc_z.setter
    def loc_z(self, value: float) -> None:
        self._M_CAM_WCS.origin[2] = value

    @property
    def rot_x(self) -> float:
        return self._rotation[0]

    @rot_x.setter
    def rot_x(self, value) -> None:
        location, rotation = self._M_CAM_WCS.to_unreal()
        rotation[0] = value

        # Set the camera pose with the setter (not by setting `self._M_CAM_WCS` so the rotation gets updated too
        self.M_CAM_WCS = Transform.from_unreal(location, rotation)

    @property
    def rot_y(self) -> float:
        return self._rotation[1]

    @rot_y.setter
    def rot_y(self, value) -> None:
        location, rotation = self._M_CAM_WCS.to_unreal()
        rotation[1] = value
        self.M_CAM_WCS = Transform.from_unreal(location, rotation)

    @property
    def rot_z(self) -> float:
        return self._rotation[2]

    @rot_z.setter
    def rot_z(self, value) -> None:
        location, rotation = self._M_CAM_WCS.to_unreal()
        rotation[2] = value
        self.M_CAM_WCS = Transform.from_unreal(location, rotation)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, value: float):
        self._velocity = value

    # KEYBOARD STUFF ===================================================================================================
    def disable_movement(self):
        self._move_forward_backward = None
        self._move_left_right = None
        self._move_up_down = None

    def key_input(self, key: int, action: str, modifiers: KeyModifiers) -> None:
        """Process key inputs and move camera

        Args:
            key: The key
            action: key action release/press
            modifiers: key modifier states such as ctrl or shit
        """
        keys = self._keys

        # Forward
        if key == keys.W:
            if action == keys.ACTION_PRESS:
                self._move_forward_backward = 1
            if action == keys.ACTION_RELEASE:
                self._move_forward_backward = None

        # Backward
        if key == keys.S:
            if action == keys.ACTION_PRESS:
                self._move_forward_backward = -1
            if action == keys.ACTION_RELEASE:
                self._move_forward_backward = None

        # Right
        if key == keys.D:
            if action == keys.ACTION_PRESS:
                self._move_left_right = 1
            if action == keys.ACTION_RELEASE:
                self._move_left_right = None

        # Left
        if key == keys.A:
            if action == keys.ACTION_PRESS:
                self._move_left_right = -1
            if action == keys.ACTION_RELEASE:
                self._move_left_right = None

        # Down
        if key == keys.Q:
            if action == keys.ACTION_PRESS:
                self._move_up_down = -1
            if action == keys.ACTION_RELEASE:
                self._move_up_down = None

        # Up
        if key == keys.E:
            if action == keys.ACTION_PRESS:
                self._move_up_down = 1
            if action == keys.ACTION_RELEASE:
                self._move_up_down = None

    def mouse_input(self, dx: int, dy: int) -> None:
        now = time.time()
        delta = now - self._last_rot_time
        self._last_rot_time = now

        # Greatly decrease the chance of camera popping. This can happen when the mouse enters and leaves the window
        # or when getting focus again.
        if delta > 0.1 and max(abs(dx), abs(dy)) > 2:
            return

        dx *= self._mouse_sensitivity
        dy *= self._mouse_sensitivity

        location, rotation = self._M_CAM_WCS.to_unreal()
        rotation = [rotation[0], min(max(rotation[1] - dy, -89), 89), rotation[2] + dx]
        self.M_CAM_WCS = Transform.from_unreal(location, rotation)
