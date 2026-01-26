from __future__ import annotations
import logging
import socket
import pickle
import time
from multiprocessing import Process
from pathlib import Path
from typing import Optional, overload, Dict, Sequence

import numpy as np
from matplotlib.colors import to_rgba
import zmq

from gigagoop.coord import Transform, Scale, Translate
from gigagoop.typing import ArrayLike, PathLike
from gigagoop.viz.color import get_vertex_rgba
from gigagoop.camera import PinholeCamera

from .params import WindowParams
from .primitives import builder
from .mesh import Mesh

log = logging.getLogger(__name__)


def _start_engine(port: int,
                  window_params: Optional[WindowParams] = None,
                  cam: Optional[PinholeCamera] = None):
    from .engine import ServerEngine

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('moderngl_window').setLevel(logging.WARNING)

    engine = ServerEngine(port, window_params, cam)
    engine.run()


def check_position(position: ArrayLike) -> np.ndarray:
    """Validates the input position array has shape `[n, 3]` and converts it to a numpy array.

    This function checks if the input position array has the correct shape `[n, 3]`, where `n` is the number
    of points and 3 represents the x, y, and z coordinates. If the input position array has the correct
    shape, it is converted to a numpy array and returned.
    """
    position = np.array(position)

    # Allow a 3-tuple to be passed in, but convert it to the "tall matrix form"
    if position.ndim == 1:
        assert len(position) == 3
        position = np.atleast_2d(position)

    # Verify "tall matrix" form
    assert position.ndim == 2
    n, k = position.shape
    assert k == 3

    return position


def _normalize_direction(direction: ArrayLike) -> np.ndarray:
    direction = np.array(direction, dtype=float).flatten()
    assert len(direction) == 3
    norm = np.linalg.norm(direction)
    assert norm > np.finfo(float).eps
    return direction / norm


def _frame_from_forward_up(origin: ArrayLike,
                           forward: ArrayLike,
                           up: Optional[ArrayLike] = None) -> Transform:
    origin = np.array(origin, dtype=float).flatten()
    assert len(origin) == 3

    forward = _normalize_direction(forward)

    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up = np.array(up, dtype=float).flatten()
        assert len(up) == 3

    if abs(np.dot(forward, up) / (np.linalg.norm(up) + 1e-12)) > 0.98:
        if abs(forward[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            up = np.array([0.0, 1.0, 0.0], dtype=float)

    up = up / (np.linalg.norm(up) + 1e-12)

    # LHS basis: X forward, Y right, Z up
    Y = np.cross(forward, -up)
    if np.linalg.norm(Y) < np.finfo(float).eps:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        if abs(np.dot(forward, fallback)) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        Y = np.cross(forward, fallback)
    Y = Y / (np.linalg.norm(Y) + 1e-12)
    Z = np.cross(forward, Y)
    Z = Z / (np.linalg.norm(Z) + 1e-12)

    return Transform.from_rotation_and_origin(rotation=np.c_[forward, Y, Z], origin=origin)


def _normalize_texts(text: str | Sequence[str], count: int) -> list[str]:
    if isinstance(text, str):
        if count == 1:
            return [text]
        return [text for _ in range(count)]

    texts = list(text)
    assert len(texts) == count
    return [str(t) for t in texts]


def _normalize_vectors(value: ArrayLike | Sequence[ArrayLike],
                       count: int) -> np.ndarray:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, np.ndarray)):
        if len(value) == 3 and np.array(value).ndim == 1:
            vecs = np.tile(np.array(value, dtype=float).reshape(1, 3), (count, 1))
        else:
            vecs = np.array(value, dtype=float)
    else:
        vecs = np.array(value, dtype=float)

    if vecs.ndim == 1:
        assert len(vecs) == 3
        vecs = np.tile(vecs.reshape(1, 3), (count, 1))

    assert vecs.ndim == 2
    assert vecs.shape[0] == count and vecs.shape[1] == 3
    return vecs


def _render_text_rgba(text: str,
                      rgb: np.ndarray,
                      font_scale: float,
                      thickness: int,
                      padding_px: int,
                      line_spacing: float) -> np.ndarray:
    try:
        import cv2
    except Exception as ex:
        raise ImportError('SpaceGraph.text3d requires opencv-python (cv2).') from ex

    lines = text.splitlines() or ['']
    font = cv2.FONT_HERSHEY_SIMPLEX

    metrics = []
    max_width = 0
    total_height = 0

    for line in lines:
        (w, h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        metrics.append((line, w, h, baseline))
        max_width = max(max_width, w)
        total_height += h + baseline

    if len(lines) > 1:
        for _, _, h, baseline in metrics[:-1]:
            total_height += int((line_spacing - 1.0) * (h + baseline))

    width = max_width + 2 * padding_px
    height = total_height + 2 * padding_px

    width = max(width, 1)
    height = max(height, 1)

    mask = np.zeros((height, width), dtype=np.uint8)

    y = padding_px
    for idx, (line, w, h, baseline) in enumerate(metrics):
        baseline_y = y + h
        cv2.putText(mask,
                    line,
                    (padding_px, baseline_y),
                    font,
                    font_scale,
                    255,
                    thickness,
                    cv2.LINE_AA)
        y += h + baseline
        if idx < len(metrics) - 1:
            y += int((line_spacing - 1.0) * (h + baseline))

    mask_f = (mask.astype(np.float32) / 255.0).reshape(height, width, 1)
    rgb_f = np.clip(rgb.reshape(1, 1, 3), 0.0, 1.0)
    rgb_img = np.broadcast_to((rgb_f * 255.0).astype(np.uint8), (height, width, 3)).copy()
    alpha_img = (mask_f * 255.0).astype(np.uint8)

    rgba = np.concatenate([rgb_img, alpha_img], axis=2)
    return rgba


def get_open_port():
    # Bind to port 0 so the OS picks a free ephemeral port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        _, port = sock.getsockname()

    log.debug(f'using port: {port}')
    return port


class SpaceGraph:
    """
    The whole idea of `SpaceGraph` is to have a tool at our fingertips to do something like:
    ```
        sg = SpaceGraph()
        sg.plot(np.random.randn(100, 3))

        # Do other stuff
    ```

    where `ax.plot` is *non-blocking*, which means you can run those commands from within the PyCharm debugger, or from
    a Jupyter notebook.

    PyCharm Slowdown (Windows)
    --------------------------
    A common use-case is to use `SpaceGraph` from within the PyCharm debugger. By default, using `SpaceGraph` in this
    way is a bit slow; this can be fixed by unchecking `Attach to subprocess automatically while debugging` in
    `Settings > Build, Execution, Deployment > Python Debugger`. Take note that you will now lose access to debugging
    subprocesses, so use this only if you know what you are doing. The speedup is ~2-3x, so it's up to you if this is
    worth it or not (on this machine it takes ~3 seconds to start `SpaceGraph` in a PyCharm debugger with default
    settings, and ~1 second to start when subprocess debugging is disabled). Furthermore, take note that this slowdown
    is specific to Windows due to its inability to fork a process.

    Design
    ------
    To achieve this non-blocking behavior, we utilize a request-reply pattern, such that a server starts up waiting for
    requests (like `plot`) from the client. In our case,
        `SpaceGraph`:   the client
        `Engine`:       the server
    """
    def __init__(self,
                 show_wcs=True,
                 show_grid=True,
                 window_params: Optional[WindowParams] = None,
                 cam: Optional[PinholeCamera] = None,
                 startup_timeout_s: Optional[float] = 5.0,
                 scale_wcs=1.0,
                 grid_size=10):

        self._closed = False
        self._startup_timeout_s = None if startup_timeout_s is None else float(startup_timeout_s)
        self._port = get_open_port()
        self._start_server_non_blocking(window_params, cam)
        try:
            self._setup_client()
        except Exception:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            raise

        if show_wcs:
            self.add_cframe(scale=scale_wcs)

        if show_grid:
            self.add_xy_grid(size=grid_size)

    # ==================================================================================================================
    # UTILITY
    # -----------
    @property
    def M_CAM_WCS(self) -> Transform:
        """Get the camera pose."""
        send_message = {'MessageType': 'get_camera_pose'}
        recv_msg = self._send_message(send_message)
        M_CAM_WCS = Transform(recv_msg['Optional']['M_CAM_WCS'])
        return M_CAM_WCS

    @M_CAM_WCS.setter
    def M_CAM_WCS(self, transform: Transform) -> None:
        """Set the camera pose (also setting the default).

        Set the camera pose and the default camera pose, such that when the `Reset` button is pressed, the camera will
        move back to the pose given here. This API gives you nice things like:
            sg = SpaceGraph()
            sg.M_CAM_WCS = Transform.from_unreal(...)
            sg.M_CAM_WCS = GazeAt(...)
        """
        message = {'MessageType': 'set_camera_pose',
                   'M_CAM_WCS': transform.matrix}

        self._send_message(message)

    def get_image_snapshot(self) -> np.ndarray:
        """Get a snapshot of the current (rgba) image displayed."""
        message = {'MessageType': 'get_image_snapshot'}

        recv_msg = self._send_message(message)
        rgba = recv_msg['Optional']['rgba']

        return rgba

    def show(self):
        """Spin forever to keep SpaceGraph open.

        When using SpaceGraph in a non-debug environment, it is convenient to have a blocking function to keep it up and
        running. As an example, you might want to run an application like
            sg = SpaceGraph()
            ...
            sg.show()    # keeps SpaceGraph open
        """
        while True:
            time.sleep(1)

            if not self._process.is_alive():
                break

    def close(self):
        # TODO - trigger a clean termination, this is a hack...
        self._closed = True
        self._shutdown_client()
        if self._process.is_alive():
            self._process.terminate()

    # ==================================================================================================================
    # ADD [GEOMETRY]
    # --------------
    def add_cframe(self,
                   M_OBJ_WCS: Optional[Transform] = None,
                   scale=1.0,
                   colors: Optional[Dict[str, str]] = None,
                   use_variant: bool = False) -> None:
        """Adds a coordinate frame to space graph.

        The coordinate frame is defined by a transformation matrix `M_OBJ_WCS`, which represents the object's frame
        w.r.t. the world coordinate system (WCS). If no transformation is provided, an identity matrix is used,
        meaning the object's local and world coordinate systems are the same.

        If the dictionary `colors` is passed in, then you can specify specific colors for the cframe. The colors should
        be specified as strings that can be interpreted by `matplotlib.colors.to_rgba`. See the `use_variant` below to
        understand the fields of the dictionary.

        If the `use_variant` flag is specified, then instead of coloring the cframe with RGB, a CMY color scheme is
        used. This coloring scheme can be helpful when comparing different cframes with each other.
        """

        if M_OBJ_WCS is None:
            M_OBJ_WCS = Transform()    # identity

        M_OBJ_WCS = M_OBJ_WCS * Scale(scale, scale, scale)

        if use_variant and colors is None:
            colors = {'origin_color': 'xkcd:white',
                      'x_edge_color': 'xkcd:bright cyan',
                      'y_edge_color': 'xkcd:electric purple',
                      'z_edge_color': 'xkcd:lemon yellow',
                      'x_tip_color': 'xkcd:bright cyan',
                      'y_tip_color': 'xkcd:electric purple',
                      'z_tip_color': 'xkcd:lemon yellow'}

        message = {'MessageType': 'add_cframe',
                   'M_OBJ_WCS': M_OBJ_WCS.matrix,
                   'colors': colors}

        self._send_message(message)

    def add_xy_grid(self,
                    M_OBJ_WCS: Optional[Transform] = None,
                    color: str = 'xkcd:medium gray',
                    alpha: float = 1.0,
                    size: int = 10) -> None:

        if M_OBJ_WCS is None:
            M_OBJ_WCS = Transform()    # identity

        rgba = to_rgba(color, alpha)

        size = int(size)
        size = max(size, 0)

        message = {'MessageType': 'add_xy_grid',
                   'M_OBJ_WCS': M_OBJ_WCS.matrix,
                   'rgba': np.array(rgba),
                   'size': size}

        self._send_message(message)

    def add_sphere(self,
                   origin: Optional[ArrayLike] = None,
                   color: str = 'xkcd:lime',
                   size: float = 1.0,
                   alpha: float = 1.0) -> None:

        if origin is None:
            origin = np.zeros(3)

        rgba = to_rgba(color, alpha)

        origin = np.array(origin)
        if origin.ndim == 1:
            origin = origin.flatten()
            assert len(origin) == 3
            x, y, z = origin

            M_OBJ_WCS = Translate(x, y, z)

            message = {'MessageType': 'add_sphere',
                       'M_OBJ_WCS': M_OBJ_WCS.matrix,
                       'rgba': np.array(rgba),
                       'size': size}
            self._send_message(message)
            return

        assert origin.ndim == 2 and origin.shape[1] == 3
        if origin.shape[0] == 0:
            return

        message = {'MessageType': 'add_spheres',
                   'origin': origin.astype(np.float32),
                   'rgba': np.array(rgba),
                   'size': size}

        self._send_message(message)

    def add_arrow(self,
                  origin: ArrayLike,
                  direction: ArrayLike,
                  width: float = 0.5,
                  color: str | ArrayLike = 'xkcd:sky blue',
                  alpha: float = 1.0):

        origin = check_position(origin)
        direction = check_position(direction)
        width = float(width)
        rgba = get_vertex_rgba(origin, color, alpha)

        assert origin.shape == direction.shape
        assert width > np.finfo(np.float32).eps

        lens = np.linalg.norm(direction, axis=1)
        assert np.max(lens) > np.finfo(np.float32).eps, 'direction vectors should have a direction ;-P'

        message = {'MessageType': 'add_arrow',
                   'origin': origin,
                   'direction': direction,
                   'width': width,
                   'rgba': rgba}

        self._send_message(message)

    def add_image(self,
                  image: np.ndarray,
                  width: float,
                  height: float,
                  M_OBJ_WCS: Optional[Transform] = None,
                  alpha: float = 1) -> None:

        if M_OBJ_WCS is None:
            M_OBJ_WCS = Transform()    # identity

        assert image.ndim >= 2
        if image.dtype != np.uint8:
            assert np.min(image) >= 0
            assert np.max(image) <= 1

        alpha = float(alpha)
        assert 0 <= alpha <= 1

        message = {'MessageType': 'add_image',
                   'image': image,
                   'width': width,
                   'height': height,
                   'M_OBJ_WCS': M_OBJ_WCS.matrix,
                   'alpha': alpha}

        self._send_message(message)

    def add_unreal_mesh(self, obj_file: PathLike) -> None:

        # Before sending a message to the server, make sure the mesh is valid - this is helpful to catch the error on
        # the client side instead of in a spawned process
        mesh = builder.UnrealMesh(obj_file, use_materials=True)
        assert mesh.has_materials
        assert mesh.valid_textures

        # Should be good to go...
        obj_file = Path(obj_file).as_posix()

        message = {'MessageType': 'add_unreal_mesh',
                   'obj_file': obj_file}

        self._send_message(message)

    # ==================================================================================================================
    # PLOTTING (SCATTER, PLOT, MESH, ...)
    # -----------------------------------
    def scatter(self,
                position: ArrayLike,
                color: str | ArrayLike = 'white',
                alpha: float = 1.0,
                size: float = 1.0):

        position = check_position(position)
        rgba = get_vertex_rgba(position, color, alpha)

        message = {'MessageType': 'scatter',
                   'position': position,
                   'rgba': rgba,
                   'size': size}

        self._send_message(message)

    def plot(self,
             position: ArrayLike,
             color: str | ArrayLike = 'white',
             alpha: float = 1.0,
             lines: bool = False,
             linewidth: Optional[float] = None):
        """Create a line plot.

        When `lines=False`, then a "line-strip" is used, such that `position[0]` connects to `position[1], which then
        connects to `position[2]`, and so on. If you want a bunch of disconnected lines, then set `lines=True` which
        means `position[0] <---> position[1]` would be a line, `position[2] <---> position[3]` would be another line,
        and so on. When `linewidth` is provided, the plot is rendered as a tube with the requested width.
        """
        position = check_position(position)
        rgba = get_vertex_rgba(position, color, alpha)
        lines = bool(lines)

        if lines:
            assert len(position) % 2 == 0

        if linewidth is not None:
            linewidth = float(linewidth)
            if linewidth <= 0:
                raise ValueError('linewidth must be positive')

        message = {'MessageType': 'plot',
                   'position': position,
                   'rgba': rgba,
                   'lines': lines,
                   'linewidth': linewidth}

        self._send_message(message)

    def text(self,
             position: ArrayLike,
             text: str | list[str],
             color: str | ArrayLike = 'white',
             alpha: float = 1.0,
             size: float = 1.0,
             offset_px: ArrayLike = (0.0, 0.0),
             halign: str = 'center',
             valign: str = 'center',
             clip: bool = True):
        """Render text labels anchored to 3D positions.

        Notes
        -----
        The `size` parameter is a scale multiplier on the default UI font.
        """
        position = check_position(position)

        if isinstance(text, str):
            texts = [text for _ in range(len(position))]
        else:
            texts = list(text)
            if len(texts) == 1 and len(position) > 1:
                texts = texts * len(position)
            else:
                assert len(texts) == len(position)

        rgba = get_vertex_rgba(position, color, alpha)

        offset_px = np.array(offset_px, dtype=np.float32).flatten()
        assert len(offset_px) == 2

        message = {'MessageType': 'text',
                   'position': position,
                   'text': texts,
                   'rgba': rgba,
                   'size': float(size),
                   'offset_px': offset_px,
                   'halign': str(halign),
                   'valign': str(valign),
                   'clip': bool(clip)}

        self._send_message(message)

    def text3d(self,
               position: ArrayLike,
               text: str | Sequence[str],
               normal: ArrayLike = (1.0, 0.0, 0.0),
               up: Optional[ArrayLike] = None,
               height: float | Sequence[float] = 0.5,
               color: str | ArrayLike = 'white',
               alpha: float = 1.0,
               font_scale: float = 1.0,
               thickness: int = 2,
               padding_px: int = 4,
               line_spacing: float = 1.0):
        """Render 3D text quads that face a fixed direction."""
        position = check_position(position)
        texts = _normalize_texts(text, len(position))
        normals = _normalize_vectors(normal, len(position))

        if up is None:
            ups = None
        else:
            ups = _normalize_vectors(up, len(position))

        rgba = get_vertex_rgba(position, color, alpha)

        heights = np.array(height, dtype=float)
        if heights.ndim == 0:
            heights = np.full(len(position), float(heights))
        elif heights.ndim == 1:
            assert len(heights) == len(position)
        else:
            raise ValueError('height must be a scalar or a 1D array')

        for idx, (pos, text_str, normal_vec, rgba_i, height_i) in enumerate(
                zip(position, texts, normals, rgba, heights)):
            if text_str is None:
                continue

            if height_i <= 0:
                raise ValueError('height must be positive')

            rgb = np.array(rgba_i[:3], dtype=float)
            alpha_i = float(rgba_i[3])
            image = _render_text_rgba(text_str,
                                      rgb,
                                      font_scale=float(font_scale),
                                      thickness=int(thickness),
                                      padding_px=int(padding_px),
                                      line_spacing=float(line_spacing))

            h_px, w_px, _ = image.shape
            if h_px <= 0 or w_px <= 0:
                continue

            width_i = float(height_i) * (float(w_px) / float(h_px))

            up_vec = None if ups is None else ups[idx]
            M_OBJ_WCS = _frame_from_forward_up(pos, normal_vec, up_vec)

            message = {'MessageType': 'text3d',
                       'M_OBJ_WCS': M_OBJ_WCS.matrix,
                       'image': image,
                       'width': width_i,
                       'height': float(height_i),
                       'alpha': alpha_i}

            self._send_message(message)

    @overload
    def mesh(self,
             vertices: ArrayLike,
             faces: Optional[ArrayLike] = None,
             color: str | ArrayLike = 'white',
             alpha: float = None):
        ...

    @overload
    def mesh(self, mesh: Mesh):
        ...

    def mesh(self,
             vertices: ArrayLike | Mesh,
             faces: Optional[ArrayLike] = None,
             color: str | ArrayLike = 'white',
             alpha: float = None):

        if isinstance(vertices, Mesh):
            mesh = vertices

            vertices = mesh.vertices
            faces = mesh.faces
            color = mesh.rgba

        vertices = check_position(vertices)

        if faces is None:
            # When no faces are passed in, then we assume the vertices are of the form `[a1, b1, c1, a2, b2, c2, ...]`
            # such that the first triangle consists of `[a1, b1, c1]`, the second `[a2, b2, c2]`, etc.
            assert len(vertices) % 3 == 0
            num_tris = len(vertices)
            faces = np.arange(num_tris).reshape([-1, 3])

        faces = check_position(faces)

        # The `vertices` input is indexed from `[0, num_verts-1]`, so make sure faces is bounded correctly such that it
        # indexes into `vertices`
        num_verts = vertices.shape[0]

        assert np.min(faces) >= 0 and np.max(faces) <= num_verts - 1, \
            'faces is out of bounds, it should index into `vertices`'

        # We can use `get_vertex_rgba` to form the color of each triangle (since we assume (at least for now) that the
        # color of each triangle is uniform)
        rgba = get_vertex_rgba(faces, color, alpha)

        # Pack up the message and let it rip
        message = {'MessageType': 'mesh',
                   'vertices': vertices,
                   'faces': faces,
                   'rgba': rgba}

        self._send_message(message)

    def lit_mesh(self,
                 vertices: ArrayLike,
                 faces: ArrayLike,
                 normals: ArrayLike,
                 color: str | ArrayLike = 'white',
                 alpha: float = None,
                 light_pos: Optional[ArrayLike] = None,
                 light_color: Optional[ArrayLike] = None,
                 light_intensity: float = 1.0,
                 ambient_light: Optional[ArrayLike] = None,
                 face_colors: Optional[ArrayLike] = None):

        if light_pos is None:
            light_pos = [0, 0, 10]

        if light_color is None:
            light_color = [1, 1, 1]

        if ambient_light is None:
            ambient_light = [0.1, 0.1, 0.1]

        vertices = check_position(vertices)
        faces = check_position(faces)
        normals = check_position(normals)

        num_faces = len(faces)
        assert len(normals) == num_faces

        # The `vertices` input is indexed from `[0, num_verts-1]`, so make sure faces is bounded correctly such that it
        # indexes into `vertices`
        num_verts = vertices.shape[0]

        assert np.min(faces) >= 0 and np.max(faces) <= num_verts - 1, \
            'faces is out of bounds, it should index into `vertices`'

        light_pos = np.array(light_pos)
        light_color = np.array(light_color)
        ambient_light = np.array(ambient_light)

        assert len(light_pos) == 3
        assert len(light_color) == 3
        assert len(ambient_light) == 3

        # We can use `get_vertex_rgba` to form the color of each triangle (since we assume (at least for now) that the
        # color of each triangle is uniform)
        if face_colors is None:
            rgba = get_vertex_rgba(faces, color, alpha)
        else:
            face_colors = np.array(face_colors).astype(np.float32)
            assert face_colors.shape == (num_faces, 4)
            rgba = face_colors

        # Pack up the message and let it rip
        message = {'MessageType': 'lit_mesh',
                   'vertices': vertices,
                   'faces': faces,
                   'normals': normals,
                   'rgba': rgba,
                   'light_pos': light_pos,
                   'light_color': light_color,
                   'light_intensity': light_intensity,
                   'ambient_light': ambient_light}

        self._send_message(message)

    def wireframe(self,
                  vertices: ArrayLike,
                  faces: Optional[ArrayLike] = None,
                  vertex_color: str = 'white',
                  vertex_size: float = 1.0,
                  mesh_color: str = 'white',
                  mesh_alpha: float = 0.5,
                  edge_color: str = 'white',
                  edge_alpha: float = 1.0,
                  linewidth: Optional[float] = None):

        vertices = check_position(vertices)

        if faces is None:
            faces = np.arange(len(vertices)).reshape(-1, 3)

        faces = check_position(faces)

        self.scatter(vertices, color=vertex_color, size=vertex_size)

        self.mesh(vertices, faces, color=mesh_color, alpha=mesh_alpha)

        # Find all the unique edges so we don't plot a bunch of lines that are duplicates
        edges = np.vstack([faces[:, [0, 1]],      # edge from vertex 0 to vertex 1
                           faces[:, [1, 2]],      # edge from vertex 1 to vertex 2
                           faces[:, [2, 0]]])     # edge from vertex 2 to vertex 0

        sorted_edges = np.sort(edges, axis=1)
        unique_edges = np.unique(sorted_edges, axis=0)

        edges_as_lines = np.array([(vertices[start], vertices[end])
                                   for start, end in unique_edges])
        edges_as_lines = edges_as_lines.reshape(-1, 3)

        self.plot(edges_as_lines, lines=True, color=edge_color, alpha=edge_alpha, linewidth=linewidth)

    def vectors(self,
                vectors: ArrayLike,
                positions: Optional[ArrayLike] = None,
                scale: float = 1.0,
                color: str = 'white',
                alpha: float = 1.0):

        vectors = check_position(vectors)
        if positions is None:
            positions = np.zeros_like(vectors)

        positions = check_position(positions)
        assert positions.shape == positions.shape

        lines = []
        for pos, vector in zip(positions, vectors):
            lines.append(pos)
            lines.append(pos + scale * vector)

        lines = np.vstack(lines)

        self.plot(lines, lines=True, color=color, alpha=alpha)

    # ------------------------------------------------------------------------------------------------------------------
    def _ensure_server_running(self):
        if self._closed or self._socket is None:
            raise RuntimeError('SpaceGraph client is closed.')

        if not self._process.is_alive():
            self._closed = True
            self._shutdown_client()
            raise RuntimeError('SpaceGraph server is not running; it may have been closed.')

    def _shutdown_client(self):
        socket = getattr(self, '_socket', None)
        if socket is not None:
            try:
                socket.close(0)
            except Exception:
                pass
            self._socket = None

        context = getattr(self, '_context', None)
        if context is not None:
            try:
                context.term()
            except Exception:
                pass
            self._context = None

    def _send_message(self, message, timeout_s: Optional[float] = None):
        assert 'MessageType' in message

        self._ensure_server_running()

        # Serialize the message
        data = pickle.dumps(message)

        # Put it on the wire
        try:
            self._socket.send(data)

        except Exception as ex:
            log.error(f'error sending message with {ex=}')
            self._shutdown_client()
            raise

        # Wait for a reply
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        start_time = time.monotonic()
        poll_interval_ms = 250
        while True:
            if timeout_s is None:
                poll_ms = poll_interval_ms
            else:
                elapsed = time.monotonic() - start_time
                remaining_s = timeout_s - elapsed
                if remaining_s <= 0:
                    raise TimeoutError(f'no response from SpaceGraph server on port {self._port}')
                poll_ms = int(min(remaining_s, poll_interval_ms / 1000.0) * 1000)

            events = dict(poller.poll(poll_ms))
            if events.get(self._socket) == zmq.POLLIN:
                data = self._socket.recv()
                break

            if not self._process.is_alive():
                self._closed = True
                self._shutdown_client()
                raise RuntimeError('SpaceGraph server closed before responding.')
        recv_msg = pickle.loads(data)
        assert 'MessageType' in recv_msg and recv_msg['MessageType'] == 'Ack' and recv_msg['Success'] == True

        return recv_msg

    def _setup_client(self):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)

        self._socket.connect(f'tcp://127.0.0.1:{self._port}')

        # Check if the server is up and running by first sending a message
        try:
            self._send_message(message={'MessageType': 'IsAlive'}, timeout_s=self._startup_timeout_s)
        except TimeoutError as ex:
            self._shutdown_client()
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            raise TimeoutError('SpaceGraph server did not respond during startup') from ex

    def _start_server_non_blocking(self,
                                   window_params: Optional[WindowParams] = None,
                                   cam: Optional[PinholeCamera] = None):

        # The `Engine` is a ZeroMQ server that accepts messages (requests) - this gets started up in a non-blocking way
        self._process = Process(target=_start_engine,
                                args=(self._port, window_params, cam))
        self._process.start()
