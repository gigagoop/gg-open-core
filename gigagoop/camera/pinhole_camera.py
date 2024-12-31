from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Dict

import numpy as np
from numpy.typing import NDArray

import cv2

from gigagoop.coord import Transform
from gigagoop.types import ArrayLike

log = logging.getLogger(__file__)


class PinholeCamera:
    """
    A class representing a Pinhole Camera model.

    Parameters
    ----------
    focal_length_x
        The camera's focal length along the x dimension (pixels).

    focal_length_y
        The camera's focal length along the y dimension (pixels).

    width
        Horizontal resolution of the image (pixels).

    height
        Vertical resolution of the image (pixels).

    optical_center_x
        Coordinate in x dimension of the optical center (pixels). If it is not passed in, then it is inferred (see
        `self_set_optical_center` for details).

    optical_center_y
        Coordinate in y dimension of the optical center (pixels). If it is not passed in, then it is inferred (see
        `self_set_optical_center` for details).

    scale_factor
        A scaling factor applied to the camera's intrinsic parameters, such that:
            - `scale_factor=1.0` ---> no scaling
            - `scale_factor=0.5` ---> downsample by a factor of two
            - `scale_factor=2.0` ---> upsample by a factor of two
    """
    def __init__(self,
                 focal_length_x: float,
                 focal_length_y: float,
                 width: int,
                 height: int,
                 optical_center_x: Optional[float] = None,
                 optical_center_y: Optional[float] = None,
                 scale_factor: float = 1.0,
                 camera_pose: Optional[Transform] = None,
                 image_file: Optional[Path] = None):

        # The `width` and `height` specify the size of the source sensor data, which does not change. We generalize the
        # image size by allowing the width and height to change based on a scale factor (which are retrieved through
        # properties)
        self._source_width = width
        self._source_height = height

        # If a scale factor is passed in, the output images (e.g. from `create_rgb_image` will have a different size
        # than the source data
        self._configure_scaling(scale_factor)

        # When resolving the pixel to world ambiguity, we need to know the direction of the camera forward vector. Our
        # convention follows the UE LHS convention with X=forward, Y=right, and Z=up.
        self._camera_forward_direction_cam = np.array([1.0, 0.0, 0.0])

        # Store fundamental camera parameters
        self._source_focal_length_x = focal_length_x
        self._source_focal_length_y = focal_length_y
        self._set_optical_center(optical_center_x, optical_center_y, width, height)

        # It is useful to store the field of view
        self._fov_horizontal, self._fov_vertical = self._compute_fovs()

        # If the camera pose is passed in, use it, otherwise identity will suffice
        if camera_pose is None:
            camera_pose = Transform()

        # Store the reference to the pixel data if it exists, take note that `image_file=None` is valid for many
        # situations, where you just want access to things like pixel to world.
        self.image_file = image_file

        # Setup internals so we can map from pixel to world and world to pixel
        self._camera_calibration_matrix = self._compute_camera_calibration_matrix()
        self._M_CAM_WCS = camera_pose.copy()
        self._camera_projection_matrix = self._compute_camera_projection_matrix()

    @staticmethod
    def create_from_cam(cam: PinholeCamera):
        return PinholeCamera(focal_length_x=cam._source_focal_length_x,
                             focal_length_y=cam._source_focal_length_y,
                             width=cam._source_width,
                             height=cam._source_height,
                             optical_center_x=cam._source_optical_center_x,
                             optical_center_y=cam._source_optical_center_y,
                             scale_factor=cam.scale_factor,
                             camera_pose=cam.M_CAM_WCS,
                             image_file=cam.image_file)

    @staticmethod
    def create_from_hfov(fov: float,
                         width: int,
                         height: int,
                         scale_factor: float = 1.0):

        fov_h = np.deg2rad(fov)

        aspect_ratio = width / height
        fov_v = 2 * np.arctan(np.tan(fov_h / 2) / aspect_ratio)

        fx = width / (2 * np.tan(fov_h / 2))
        fy = height / (2 * np.tan(fov_v / 2))

        return PinholeCamera(focal_length_x=fx,
                             focal_length_y=fy,
                             width=width,
                             height=height,
                             scale_factor=scale_factor)

    @staticmethod
    def create_from_vfov(fov: float,
                         width: int,
                         height: int,
                         scale_factor: float = 1.0):

        fov_v = np.deg2rad(fov)

        aspect_ratio = width / height
        fov_h = 2 * np.arctan(np.tan(fov_v / 2) * aspect_ratio)

        fx = width / (2 * np.tan(fov_h / 2))
        fy = height / (2 * np.tan(fov_v / 2))

        return PinholeCamera(focal_length_x=fx,
                             focal_length_y=fy,
                             width=width,
                             height=height,
                             scale_factor=scale_factor)

    def _set_optical_center(self, u0: Optional[float], v0: Optional[float], Nx: int, Ny: int):
        """
        The image center (u0, v0) is set with the assumption that the top-left pixel is centered at `(0, 0)`. To see
        this consider the 3x1 image.

            0.0  0.5  1.0  1.5  2.0  2.5  3.0
             |    x    |    x    |    x    |     ---> image center: 3/2 = 1.5

           -0.5  0.0  0.5  1.0  1.5  2.0  2.5
             |    x    |    x    |    x    |     ---> image center: (3-1)/2 = 1

        where the first example shows an image with the pixel centers at 0.5, while the second is at 0. If we assume
        the pixels are centered at 0, then we want to use "image_center = (num_pixels - 1)/2"

        Note 1
        ------
        This function exists to handle a common situation where the image center is not defined explicitly, but instead
        inferred by the image width and height.

        Note 2
        ------
        This definition of center follows OpenCV and can be verified (see `test_ue_camera.py`) using OpenCV camera
        calibration routines.

        Scaling Notes
        -------------
        In `self.optical_center_x` and `self.optical_center_y` the scaling is accounted for with the following
        derivation. To understand the derivation, first note that the center as defined by `(3 - 1)/2`. If we were to
        scale the image, we would write `(s*3 - 1)/2` where `s` denotes the scale factor. In the case where we are not
        given the width, we need to solve for given optical center. This can be achieved by:
            - Define `cx_0 = (W - 1)/2`
            - Define `cx_s = (s*W - 1)/2`
            - Solve `W` in `cx_0` ---> `W = 2*cx_0 + 1`
            - Plug in `W` into `cx_s` ---> `cx_s = (s*(2*cx_0 + 1) - 1)/2
                                                 = (s*2*cx_0 + s - 1)/2
                                                 = s*2*cx_0/2 + s/2 - 1/2
                                                 = s(cx_0 + 1/2) - 1/2`

        which can be seen in `optical_center_x` and `optical_center_y`.
        """
        if u0 is None:
            u0 = (Nx - 1) / 2

        if v0 is None:
            v0 = (Ny - 1) / 2

        self._source_optical_center_x = u0
        self._source_optical_center_y = v0

    @property
    def scale_factor(self) -> float:
        """Return the scale factor."""
        assert self._scale_factor is not None
        return self._scale_factor

    @property
    def focal_length_x(self) -> float:
        return self._scale_factor * self._source_focal_length_x

    @property
    def focal_length_y(self) -> float:
        return self._scale_factor * self._source_focal_length_y

    @property
    def optical_center_x(self) -> float:
        return self._scale_factor * (self._source_optical_center_x + 0.5) - 0.5

    @property
    def optical_center_y(self) -> float:
        return self._scale_factor * (self._source_optical_center_y + 0.5) - 0.5

    @property
    def has_scaling(self) -> bool:
        """Return True if the scaling."""
        return not np.isclose(self._scale_factor, 1.0)

    @property
    def M_CAM_WCS(self) -> Transform:
        """Get the camera pose."""
        return self._M_CAM_WCS

    @M_CAM_WCS.setter
    def M_CAM_WCS(self, transform: Transform) -> None:
        """Set the camera pose."""
        self._M_CAM_WCS = transform
        self._camera_projection_matrix = self._compute_camera_projection_matrix()

    @property
    def width(self) -> int:
        """Return the width of the output image (in pixels).

        Note: This image width accounts for a scale factor (if it was used); so the image width returned here may not
              match the image width from the source data.
        """
        return self._scaled_width

    @property
    def height(self) -> int:
        """Return the height of the output image (in pixels).

        Note: This image height accounts for a scale factor (if it was used); so the image height returned here may not
              match the image height from the source data.
        """
        return self._scaled_height

    @property
    def fov_horizontal(self) -> float:
        """Horizontal field of view (deg)."""
        return self._fov_horizontal

    @property
    def fov_vertical(self) -> float:
        """Vertical field of view (deg)."""
        return self._fov_vertical

    @property
    def camera_calibration_matrix(self) -> NDArray[float]:
        return self._camera_calibration_matrix

    @property
    def camera_projection_matrix(self) -> NDArray[float]:
        return self._camera_projection_matrix

    def world_to_pixel(self, pos_wcs: ArrayLike) -> NDArray:
        """Map world points `pos_wcs` to pixels `pos_pix`.

        We follow the standard mapping from world to pixel of `x = PX`; see the compact form of eq. (6.2) in [4].

        Parameters
        ----------
        pos_wcs
            Tall matrix where each row contains `(x, y, z)` w.r.t. WCS.

        Returns
        -------
        pos_pix
            Tall matrix where each row contains `(col, row)` w.r.t. pixel coordinates with `(0.5, 0.5)` the top-left.
        """

        # The positions defined w.r.t. WCS in homogenous space in a fat matrix form are given by
        X = np.c_[pos_wcs, np.ones(len(pos_wcs))].T

        # Mapping to pixels in homogeneous space is simply
        P = self._camera_projection_matrix
        x = P @ X

        # The pixels `x` are homogenous coordinates as a fat matrix form with shape [3 x n] - transformation to an
        # Euclidian form is simply
        pos_pix = x[:-1, :] / x[-1, :]
        pos_pix = pos_pix.T  # tall matrix form

        return pos_pix

    def pixel_to_world(self, pos_pix: Optional[ArrayLike] = None) -> NDArray[float]:
        """Map pixel locations to world rays.

        The pixel to world maps a pixel location to "a ray", as shown in eq. 6.13 in [4] where the center of the
        top-left pixel of the image is assumed to be at `(0, 0)`, and the bottom-right center at
        `(image_width - 1, image_height - 1`).

        Now take note that we do *not* actually return "a ray", instead we return the direction of the ray (a vector)
        with the convention that it has unit length, and points forward (the same direction as the cameras forward
        axis). For a bit more detail, understand that when we perform a back-projection (the pixel to world mapping), a
        pixel gets mapped into *a line*. The line is typically modeled as a ray, so in other words, the mapping looks
        something like this `(col, row) ---> r(t) = o + t*d`.

        We could return `r(t)` as a function, but instead we choose to return `d`, the ray direction. If you want to
        construct the ray in a functional form, you could do the following:
        ```
        o_i = self.M_CAM_WCS.origin
        v_i = pos_wcs[i]
        r_i = lambda t: o_i + t*v_i
        ```

        Parameters
        ----------
        pos_pix
            Tall matrix where each row contains `(col, row)` w.r.t. pixel coordinates with `(0.5, 0.5)` the top-left.

        Returns
        -------
        ray_direction_wcs
            Tall matrix where each row contains `(vx, vy, vz)`, a vector (of unit length) w.r.t. WCS, that points
            in the direction of the ray along from the camera origin.
        """
        if pos_pix is None:
            pos_pix = self._get_pixel_supp()

        pos_pix = np.array(pos_pix)

        # The input `pos_pix` is either a vector or a tall matrix in cartesian coordinates. We start by transforming
        # this into homogenous coordinates as a fat matrix, where every column represents [col, row, 1.0].
        if pos_pix.ndim == 1:  # vector form
            col, row = pos_pix
            x = np.c_[[col, row, 1.0]]

        elif pos_pix.ndim == 2:
            x = np.c_[pos_pix, np.ones(len(pos_pix))].T

        else:
            raise NotImplementedError

        # Following eq. 6.13 we can transform to homogenous world space using the pseudo-inverse as:
        #     P = self.camera_projection_matrix
        #     X = np.linalg.pinv(P) @ x    # [4, n]
        #     pos_wcs = X[:-1, :] / X[-1, :]    # transform to Euclidian coordinates
        #     pos_wcs = pos_wcs.T               # tall matrix form

        # The pseudo-inverse can be numerically unstable, so a different form, given by eq. 6.14 is presented
        P = self._camera_projection_matrix  # decompose into [M|p4]
        M = P[:, :-1]
        p4 = P[:, -1]
        pos_wcs = np.linalg.inv(M) @ (x - p4[:, np.newaxis])  # fat matrix
        pos_wcs = pos_wcs.T  # tall matrix

        # As noted `pos_wcs` is a point along the ray, so we can easily compute the ray direction as
        ray_origin_wcs = self._M_CAM_WCS.origin
        ray_direction_wcs = pos_wcs - ray_origin_wcs

        # As a convention, we will normalize the direction to unit length
        ray_direction_wcs = ray_direction_wcs / np.linalg.norm(ray_direction_wcs, axis=1)[:, np.newaxis]

        # The ray could technically point backwards, so it is useful to make sure all the rays are pointing forward. To
        # do this, recall that `X_cam` points forward.
        camera_forward_direction_wcs = self._M_CAM_WCS.rotation @ self._camera_forward_direction_cam

        # The sign of the dot product tells us if two vectors are pointing in the same direction, e.g.
        #     np.dot([1, 0], [1, 0]) = 1    # pointing in the same direction
        #     np.dot([1, 0], [0, 1]) = 0    # orthogonal (this should never happen)
        #     np.dot([1, 0], [-1, 0]) = -1  # pointing in opposite direction
        s = np.dot(ray_direction_wcs, camera_forward_direction_wcs)
        assert not np.any(np.isclose(s, 0.0))

        ray_direction_wcs = np.sign(s)[:, np.newaxis] * ray_direction_wcs

        # In the case a vector was passed in, return that instead of a tall matrix
        if pos_pix.ndim == 1:
            ray_direction_wcs = ray_direction_wcs[0]

        return ray_direction_wcs

    def pixel_to_cam(self, pos_pix: Optional[ArrayLike] = None) -> NDArray:
        """Map pixel locations to cameras rays.

        See `pixel_to_world` for full documentation - this routine is basically the same, but the ray direction is
        defined w.r.t. CAM instead of WCS.

        Parameters
        ----------
        pos_pix
            Tall matrix where each row contains `(col, row)` w.r.t. pixel coordinates with `(0.5, 0.5)` the top-left.

        Returns
        -------
        ray_direction_cam
            Tall matrix where each row contains `(vx, vy, vz)`, a vector (of unit length) w.r.t. CAM, that points
            in the direction of the ray along from the camera origin.
        """
        if pos_pix is None:
            pos_pix = self._get_pixel_supp()

        pos_pix = np.array(pos_pix)

        # Map the pixels to rays w.r.t. WCS
        ray_direction_wcs = self.pixel_to_world(pos_pix)

        # If the input is a vector `pixel_to_world` will also return a vector. We standardize our data, so we have it
        # all packed into a fat matrix, where each column contains `[vx, vy, vz]` a vector that points in the direction
        # of the ray.
        D_wcs = np.atleast_2d(ray_direction_wcs).T

        # To convert to `CAM` we need to realize that we have vectors, not points, so it does not make sense to run
        # these values through a `M_WCS_CAM` transform (this is one of the reasons many rendering libraries make a
        # distinction between points and vectors - in our case, we "just" need to be smart about it ;-P).

        # So... the transformation to `CAM` is given by
        M_WCS_CAM = self._M_CAM_WCS.inverse
        D_cam = M_WCS_CAM.rotation @ D_wcs

        # Now pack into our standard tall matrix form
        ray_direction_cam = D_cam.T

        # In the case a vector was passed in, return that instead of a tall matrix
        if pos_pix.ndim == 1:
            ray_direction_cam = ray_direction_cam[0]

        return ray_direction_cam

    def _get_pixel_supp(self) -> NDArray[float]:
        """Get the pixel support (as a tall matrix) for the entire image."""
        cols = np.arange(self.width)
        rows = np.arange(self.height)
        cols, rows = np.meshgrid(cols, rows)
        pos_pix = np.array([cols.flatten(), rows.flatten()]).T

        return pos_pix

    def _compute_fovs(self):
        """Compute horizontal and vertical field of view."""
        fx = self.focal_length_x
        fy = self.focal_length_y

        width = self.width
        height = self.height

        fov_h = np.rad2deg(2 * np.arctan(width / (2 * fx)))
        fov_v = np.rad2deg(2 * np.arctan(height / (2 * fy)))

        return fov_h, fov_v

    def _configure_scaling(self, scale_factor):
        # Say we have an image that is `3 x 3`. If the user wants it to be scaled to `1.5 x 1.5` that will not work ;-P,
        # so we would actually return an image of size `1 x 1`, meaning the actual scale factor will be different from
        # the requested one.
        desired_scale_factor = float(scale_factor)
        assert desired_scale_factor > 0

        scaled_width = int(self._source_width * desired_scale_factor)
        scaled_height = int(self._source_height * desired_scale_factor)
        actual_scale_factor_x = scaled_width / self._source_width
        actual_scale_factor_y = scaled_height / self._source_height

        assert np.isclose(actual_scale_factor_x, actual_scale_factor_y), \
            'we could relax this, but leaving it for now...'

        actual_scale_factor = actual_scale_factor_y

        if not np.isclose(desired_scale_factor, actual_scale_factor):
            log.warning(f'scale factor changed from {desired_scale_factor=:0.6f} to {actual_scale_factor=:0.6f}')

        scaled_width = int(self._source_width * actual_scale_factor)
        scaled_height = int(self._source_height * actual_scale_factor)

        log.debug(f'image scaled from {self._source_width}x{self._source_height} to {scaled_width}x{scaled_height}')

        # Package
        self._scaled_width = scaled_width
        self._scaled_height = scaled_height
        self._scale_factor = actual_scale_factor

    def _compute_camera_calibration_matrix(self):
        """Compute the camera calibration matrix.

        The camera calibration matrix `K` is explained by eq. 6.4 in [4]. Take note that we define it w.r.t. a different
        set of camera parameters that are exposed to us through Unreal - but the results are fundamentally the same, as
        they are both pinhole camera models.
        """

        fx = self.focal_length_x
        fy = self.focal_length_y
        cx = self.optical_center_x
        cy = self.optical_center_y

        K = np.array([[fx, 0,  cx],
                      [0,  fy, cy],
                      [0,  0,  1]])

        return K

    def _compute_camera_calibration_matrix__OLD(self):
        """Compute the camera calibration matrix.

        The camera calibration matrix `K` is explained by eq. 6.4 in [4]. Take note that we define it w.r.t. a different
        set of camera parameters that are exposed to us through Unreal - but the results are fundamentally the same, as
        they are both pinhole camera models.
        """
        mm_to_m = 0.001

        W = mm_to_m * self._sensor_width
        H = mm_to_m * self._sensor_height
        f = mm_to_m * self._focal_length

        Nx = self.width
        Ny = self.height

        assert np.isclose(Nx/Ny, W/H), 'aspect ratio of simulated sensor size should match image size'

        Delta_x = W / Nx
        Delta_y = H / Ny

        # The image center (u0, v0) are set with the assumption that the top-left pixel is centered at `(0, 0)`. To see
        # this consider the 3x1 image.
        #
        #     0.0  0.5  1.0  1.5  2.0  2.5  3.0
        #      |    x    |    x    |    x    |     ---> image center: 3/2 = 1.5
        #
        #    -0.5  0.0  0.5  1.0  1.5  2.0  2.5
        #      |    x    |    x    |    x    |     ---> image center: (3-1)/2 = 1
        #
        # where the first example shows an image with the pixel centers at 0.5, while the second is at 0. If we assume
        # the pixels are centered at 0, then we want to use "image_center = (num_pixels - 1)/2"

        u0 = (Nx - 1) / 2
        v0 = (Ny - 1) / 2

        K = np.array([[f/Delta_x, 0,         u0],
                      [0,         f/Delta_y, v0],
                      [0,         0,         1]])

        return K

    def _compute_camera_projection_matrix(self):
        """Compute the camera projection matrix.

        The camera projection matrix maps a homogenous point defined w.r.t. the world to homogenous pixel space. This is
        a fundamental term used in (linear) camera models  (see eq. 6.8 in [4]).
        """
        # The camera frame w.r.t. the world is given by
        M_CAM_WCS = self._M_CAM_WCS

        # The `CAM` frame follows the Unreal Engine convention - it is left-handed with:
        #     X: forward
        #     Y: right
        #     Z: up

        # Our camera matrix `K` assumes the OpenCV standard of a right-handed frame with:
        #     X: right
        #     Y: down (-up)
        #     Z: forward

        # Define `OCV` as the OpenCV camera frame, and define the trivial mapping between `CAM` and `OCV`
        M_OCV_CAM = Transform([[0, 0, 1, 0],
                               [1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, 0, 1]])

        # Chaining these together we can go from the world to the OpenCV camera with
        M_WCS_CAM = M_CAM_WCS.inverse
        M_CAM_OCV = M_OCV_CAM.inverse
        M_WCS_OCV = M_CAM_OCV * M_WCS_CAM

        # The projection matrix is then
        R = M_WCS_OCV.rotation
        t = M_WCS_OCV.origin

        K = self._camera_calibration_matrix

        P = K @ np.c_[R, t]

        return P

    def resize_image(self, img: ArrayLike) -> NDArray:
        img = np.array(img)

        if self.has_scaling:
            resized_img = cv2.resize(img, (self._scaled_width, self._scaled_height), interpolation=cv2.INTER_AREA)
            assert resized_img.shape[:2] == (self._scaled_height, self._scaled_width)
            return resized_img

        else:
            return img

    def create_hdr_image(self) -> NDArray[np.float32]:
        """Get the HDR image, in the range [0, max]."""
        raise NotImplementedError('method should be defined in the subclass')

    def create_ldr_image(self, img_hdr=None, tone_mapping_method: Optional[str] = None) -> NDArray[np.uint8]:
        """Create the RGB image, in the range [0, 255] (for viewing purposes)."""
        raise NotImplementedError('method should be defined in the subclass')

    def get_pixel_corners(self, col: int | float, row: int | float):
        """Compute the rays at the corner of a given pixel.

        A pixel occupies an area, this function will fill compute the rays that shoot through each of the corners. If
        you want the corners of the first pixel (the top-left pixel), you would use `get_pixel_corners(cam, c=0, r=0)`.
        """
        assert 0 <= col <= self.width - 1
        assert 0 <= row <= self.height - 1

        tl_pix = [col - 0.5, row - 0.5]
        tr_pix = [col + 0.5, row - 0.5]
        bl_pix = [col - 0.5, row + 0.5]
        br_pix = [col + 0.5, row + 0.5]

        tl_ray_wcs = self.pixel_to_world(tl_pix)
        tr_ray_wcs = self.pixel_to_world(tr_pix)
        bl_ray_wcs = self.pixel_to_world(bl_pix)
        br_ray_wcs = self.pixel_to_world(br_pix)

        return tl_ray_wcs, tr_ray_wcs, bl_ray_wcs, br_ray_wcs

    def get_frustum_main_edges(self):
        """Compute the main edges of the frustum.

        The main edges of the frustum are the ones that project outward from the camera into the scene. If we consider
        the near and far clipping planes, the frustum would consist of 12 edges. In this case, we only consider 4 edges,
        which again, are the ones that shoot out.
        """
        W = self.width
        H = self.height

        tl_ray_wcs, _, _, _ = self.get_pixel_corners(col=0, row=0)
        _, tr_ray_wcs, _, _ = self. get_pixel_corners(col=W-1, row=0)
        _, _, _, br_ray_wcs = self.get_pixel_corners(col=W-1, row=H-1)
        _, _, bl_ray_wcs, _ = self.get_pixel_corners(col=0, row=H-1)

        return tl_ray_wcs, tr_ray_wcs, bl_ray_wcs, br_ray_wcs

    def copy(self) -> PinholeCamera:
        return PinholeCamera.create_from_cam(self)

    @staticmethod
    def _compute_ortho_depth_scaling(ray_direction_cam: ArrayLike) -> NDArray[float]:
        ray_direction_cam = np.array(ray_direction_cam)  # tall matrix
        ray_direction_cam = ray_direction_cam / np.linalg.norm(ray_direction_cam, axis=1)[:, np.newaxis]

        # The cartesian coordinate frame used by our spherical coordinates is:
        #     X: right
        #     Y: forward
        #     Z: up

        # Our camera frame is:
        #     X: forward
        #     Y: right
        #     Z: up

        # We need to map rays from camera coordinates to the spherical coordinate, to utilize the expressions in [5]
        M_SPH_CAM = Transform([[0, 1, 0, 0],
                               [1, 0, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

        M_CAM_SPH = M_SPH_CAM.inverse

        ray_direction_sph = M_CAM_SPH.rotation @ ray_direction_cam.T  # fat matrix

        # Now compute the spherical angles, noting that `theta = 90 and phi = 90` means the ray is aligned with the
        # optical axis.
        x, y, z = ray_direction_sph
        theta = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))
        phi = np.sign(y) * np.arccos(x / np.sqrt(x ** 2 + y ** 2))

        scale = (np.sin(theta) * np.sin(phi))
        assert np.all(scale > 0)

        return scale

    @staticmethod
    def ortho_to_persp(ray_direction_cam: ArrayLike, depth_ortho: ArrayLike) -> NDArray[float]:
        """Transform from orthographic depth to perspective depth (see `20230126_sdg_depth_channel.pptx - Baran`)."""
        depth_ortho = np.array(depth_ortho)
        scale = PinholeCamera._compute_ortho_depth_scaling(ray_direction_cam)
        d_persp = depth_ortho.flatten() / scale
        d_persp = np.reshape(d_persp, depth_ortho.shape)

        return d_persp

    @staticmethod
    def persp_to_ortho(ray_direction_cam: ArrayLike, depth_persp: ArrayLike):
        """Transform from perspective depth to orthographic depth."""
        depth_persp = np.array(depth_persp)
        scale = PinholeCamera._compute_ortho_depth_scaling(ray_direction_cam)
        d_ortho = scale * depth_persp.flatten()
        d_ortho = np.reshape(d_ortho, depth_persp.shape)

        return d_ortho

    def params_to_dict(self) -> Dict:

        # Take note that 1.0 is used for the scale factor, as all the camera parameters (e.g. `focal_length`) have
        # already been scaled appropriately. If the scale factor is not set to unity, we would scale the parameters
        # multiple times, which is bad... ummm k.
        camera_parameters = {'focal_length_x': self.focal_length_x,
                             'focal_length_y': self.focal_length_y,
                             'width': self.width,
                             'height': self.height,
                             'optical_center_x': self.optical_center_x,
                             'optical_center_y': self.optical_center_y,
                             'scale_factor': 1.0,
                             'camera_pose': self.M_CAM_WCS.matrix.tolist()}

        return camera_parameters


