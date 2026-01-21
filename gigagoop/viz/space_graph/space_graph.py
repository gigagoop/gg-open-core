from __future__ import annotations
import logging
import socket
import pickle
import time
from multiprocessing import Process
from pathlib import Path
from typing import Optional, overload, Dict

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


def get_open_port():

    def is_port_in_use(port, host='localhost'):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return False
            except socket.error as ex:
                return True

    # ------------------------------------------------------------------------------------------------------------------
    # TODO - a serious issue is that pygame does not support multiple instances at the same time, so this actually works
    #        fine for ZeroMQ, but fails in weird ways if we use multiple instances. For now, we will disable this, and
    #        force a failure if multiple space graph instances are started
    num_attempts = 100
    open_port = None
    for _ in range(num_attempts):
        check_port = np.random.randint(low=49152, high=65535)
        if not is_port_in_use(check_port):
            open_port = check_port
            break

    assert open_port is not None, 'could not find an open port after 100 attempts'
    log.debug(f'using port: {open_port}')
    # ------------------------------------------------------------------------------------------------------------------
    # open_port = 5555
    # assert not is_port_in_use(open_port)
    # ------------------------------------------------------------------------------------------------------------------

    return open_port


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
                 scale_wcs=1.0,
                 grid_size=10):

        self._port = get_open_port()
        self._start_server_non_blocking(window_params, cam)
        self._setup_client()

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

        origin = np.array(origin).flatten()
        assert len(origin) == 3
        x, y, z = origin

        M_OBJ_WCS = Translate(x, y, z)

        rgba = to_rgba(color, alpha)

        message = {'MessageType': 'add_sphere',
                   'M_OBJ_WCS': M_OBJ_WCS.matrix,
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
        and so on.

        When `linewidth` is provided, line segments are rendered as thin cylinders with the requested width. This
        provides a visible thickness compared to the default line-strip rendering.
        """
        position = check_position(position)
        rgba = get_vertex_rgba(position, color, alpha)
        lines = bool(lines)

        if linewidth is not None:
            linewidth = float(linewidth)
            if linewidth <= 0:
                raise ValueError('linewidth must be a positive number')

        if lines:
            assert len(position) % 2 == 0

        message = {'MessageType': 'plot',
                   'position': position,
                   'rgba': rgba,
                   'lines': lines,
                   'linewidth': linewidth}

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
                  edge_alpha: float = 1.0):

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

        self.plot(edges_as_lines, lines=True, color=edge_color, alpha=edge_alpha)

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
    def _send_message(self, message):
        assert 'MessageType' in message

        # Serialize the message
        data = pickle.dumps(message)

        # Put it on the wire
        try:
            self._socket.send(data)

        except Exception as ex:
            log.error(f'error sending message with {ex=}')
            raise ex

        # Wait for a reply
        data = self._socket.recv()
        recv_msg = pickle.loads(data)
        assert 'MessageType' in recv_msg and recv_msg['MessageType'] == 'Ack' and recv_msg['Success'] == True

        return recv_msg

    def _setup_client(self):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)

        self._socket.connect(f'tcp://localhost:{self._port}')

        # Check if the server is up and running by first sending a message
        self._send_message(message={'MessageType': 'IsAlive'})

    def _start_server_non_blocking(self,
                                   window_params: Optional[WindowParams] = None,
                                   cam: Optional[PinholeCamera] = None):

        # The `Engine` is a ZeroMQ server that accepts messages (requests) - this gets started up in a non-blocking way
        self._process = Process(target=_start_engine,
                                args=(self._port, window_params, cam))
        self._process.start()
