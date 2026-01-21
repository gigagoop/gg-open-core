# ----------------------------------------------------------------------------------------------------------------------
import logging
import pickle
from typing import List

import numpy as np
import zmq
import imgui
import moderngl as mgl
import moderngl_window as mglw
from moderngl_window.context.base.keys import KeyModifiers
from moderngl_window.context.pyglet.window import PygletWrapper, Window
from moderngl_window.integrations.imgui import ModernglWindowRenderer

from gigagoop.coord import Transform
from .nodes import Node, Grid, CFrame, Scatter, Plot, Mesh, Sphere, Cylinder
from .camera import Camera

log = logging.getLogger(__name__)


class FpsTracker:
    length = 100

    def __init__(self):
        self._frame_index = 0
        self._history = np.nan * np.ones(self.length)

    def update(self, delta_time):
        fps = 1 / delta_time
        self._history[self._frame_index % self.length] = fps
        self._frame_index += 1

    @property
    def fps(self):
        return np.nanmean(self._history)


class Engine:
    def __init__(self, port: int, title: str, width: int = 1620, height: int = 1080):

        # Get the window class to be used
        window_cls = mglw.get_window_cls(window='moderngl_window.context.pyglet.Window')

        # Instantiate the window class, creating a `class BaseWindow`
        window = window_cls(title=title,
                            size=(width, height),
                            fullscreen=False,
                            resizable=True,
                            gl_version=(3, 3),
                            aspect_ratio=width / height,
                            vsync=True,
                            samples=8,
                            cursor=True)

        window.print_context_info()
        mglw.activate_context(window=window)

        # Construct a camera
        self.camera = Camera(window.keys, aspect_ratio=window.aspect_ratio, near=0.1, far=1000)
        self._ui_set_default_camera_pose()

        # Construct an imgui instance for all the ui elements
        imgui.create_context()
        self._imgui = ModernglWindowRenderer(window)
        self._fps_tracker = FpsTracker()

        # Register callbacks
        window.render_func = self._render
        window.resize_func = self._resize_event
        window.close_func = self._close_event
        window.key_event_func = self._key_event
        window.mouse_press_event_func = self._mouse_press_event
        window.mouse_release_event_func = self._mouse_release_event
        window.mouse_position_event_func = self._mouse_position_event
        window.mouse_drag_event_func = self._mouse_drag_event
        window.mouse_scroll_event_func = self._mouse_scroll_event
        window.unicode_char_entered_func = self._unicode_char_entered_event

        # Store references for later use and to prevent garbage collection
        self._window = window

        # Expose the ability to turn on/off the mouse cursor (this assumes pyglet is used)
        pyglet_window = window._window
        assert isinstance(pyglet_window, PygletWrapper), 'please use the pyglet backend'
        self._set_mouse_visible = pyglet_window.set_mouse_visible

        # Utilize the timer as shown in the mglw examples
        self._timer = mglw.Timer()
        self._timer.start()

        # Swap buffers once before staring the main loop. This can trigger additional resize events reporting a more
        # accurate buffer size.
        self._set_opengl_state(window)
        window.swap_buffers()
        window.set_default_viewport()

        # The scene contains all the nodes to render
        self._scene: List[Node] = []

        # To mimic Unreal Engine, we want to move the camera only if the right mouse button is pressed. To do this, we
        # need to track if the button is being pressed
        self._mouse_button_pressed = None

        # Setup ZeroMQ
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.REP)
        self._zmq_socket.bind(f'tcp://*:{port}')

    @property
    def ctx(self) -> mgl.Context:
        return self._window.ctx

    @property
    def screen(self) -> mgl.Framebuffer:
        return self.ctx.screen

    def run(self):
        """Start the rendering loop for the engine.

        This method is a blocking call, as it keeps executing until the window is closed or the application is
        terminated. It handles window clearing, frame rendering, and buffer swapping during each iteration.
        """
        window = self._window
        timer = self._timer

        try:
            while not window.is_closing:
                current_time, delta = timer.next_frame()

                # Set the OpenGL state (the reason for calling this over and over is `imgui` modifies the OpenGL state
                # from our custom settings - to keep our settings we need to make sure they are set each time)
                self._set_opengl_state(window)

                # Render the frame
                window.use()    # always bind the window frame-buffer before calling `window.render`
                window.render(timer, delta)

                # Check if any ZeroMQ messages have been received
                self._check_for_request(delta)

                # Render the user interface
                self._fps_tracker.update(delta)
                self._render_ui(current_time, delta)

                if not window.is_closing:
                    window.swap_buffers()

            window.destroy()
            self._imgui.shutdown()

        except KeyboardInterrupt:
            pass

    def _set_opengl_state(self, window: Window):
        window.ctx.enable(mgl.DEPTH_TEST | mgl.PROGRAM_POINT_SIZE)
        window.ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        window.ctx.blend_equation = mgl.FUNC_ADD
        window.ctx.depth_func = '<'

    def _render(self, time: float, frame_time: float):
        self.ctx.clear(red=0.0, green=0.0, blue=0.0, alpha=0.0, depth=1.0)

        for node in self._scene:
            assert isinstance(node, Node)
            node.render()

    def _render_ui(self, time: float, frame_time: float):
        imgui.new_frame()

        # ---
        imgui.begin('Stats', True)
        imgui.text(f'{self._fps_tracker.fps:0.2f} FPS')
        imgui.text(f'{1/self._fps_tracker.fps*1000:0.2f} ms')
        imgui.end()

        # ---

        imgui.begin('Camera', True)
        if imgui.button('Reset', width=60, height=20):
            self._ui_set_default_camera_pose()
        imgui.end()
        # ---

        imgui.render()
        self._imgui.render(imgui.get_draw_data())

    # UI CALLBACKS -----------------------------------------------------------------------------------------------------
    def _ui_set_default_camera_pose(self):
        self.camera.set_position(x=10, y=10, z=10)
        self.camera.set_rotation(pitch=-37, yaw=227)

    # MESSAGE HANDLING -------------------------------------------------------------------------------------------------
    def _check_for_request(self, delta_time: float):
        try:
            # In a non-blocking way, get the message (as a chunk of bytes)
            request_msg = self._zmq_socket.recv(flags=zmq.DONTWAIT)

            # We are "just" passing around json, as it is fast and flexible...
            request = pickle.loads(request_msg)

            # Do something with the message
            self._handle_message(request)

            # Ack to the client, calm them down, and let them know we got the message
            response = {'MessageType': 'Ack',
                        'Success': True}
            response_msg = pickle.dumps(response)
            self._zmq_socket.send(response_msg)

        except zmq.Again:
            pass    # no message received (this is not an error), continuing...

    def _add_node(self, node: Node):
        assert isinstance(node, Node)
        self._scene.append(node)

    def _handle_message(self, request_json):
        # Validate the message type
        assert 'MessageType' in request_json
        message_type = request_json['MessageType']
        engine = self

        if message_type == 'IsAlive':
            return

        if message_type == 'add_cframe':
            M_OBJ_WCS = Transform(request_json['M_OBJ_WCS'])
            node = CFrame(engine, M_OBJ_WCS)
            self._add_node(node)
            return

        if message_type == 'add_xy_grid':
            M_OBJ_WCS = Transform(request_json['M_OBJ_WCS'])
            rgba = np.array(request_json['rgba']).astype(np.float32)
            node = Grid(engine, M_OBJ_WCS, rgba)
            self._add_node(node)
            return

        if message_type == 'add_sphere':
            M_OBJ_WCS = Transform(request_json['M_OBJ_WCS'])
            rgba = np.array(request_json['rgba']).astype(np.float32)
            node = Sphere(engine, M_OBJ_WCS, rgba)
            self._add_node(node)
            return

        if message_type == 'scatter':
            position = np.array(request_json['position']).astype(np.float32)
            rgba = np.array(request_json['rgba']).astype(np.float32)
            size = float(request_json['size'])
            node = Scatter(engine, position, rgba, size)
            self._add_node(node)
            return

        if message_type == 'plot':
            position = np.array(request_json['position']).astype(np.float32)
            rgba = np.array(request_json['rgba']).astype(np.float32)
            lines = bool(request_json.get('lines', False))
            linewidth = request_json.get('linewidth')

            if linewidth is None:
                node = Plot(engine, position, rgba, lines)
                self._add_node(node)
                return

            linewidth = float(linewidth)
            radius = linewidth / 2.0
            num_sides = 12

            if lines:
                assert len(position) % 2 == 0
                indices = [(i, i + 1) for i in range(0, len(position), 2)]
            else:
                indices = [(i, i + 1) for i in range(len(position) - 1)]

            for i, j in indices:
                p0 = position[i]
                p1 = position[j]
                if np.allclose(p0, p1):
                    continue
                color = 0.5 * (rgba[i] + rgba[j])
                node = Cylinder(engine, p0, p1, color=color, radius=radius, num_sides=num_sides)
                self._add_node(node)
            return

        if message_type == 'mesh':
            vertices = np.array(request_json['vertices']).astype(np.float32)
            faces = np.array(request_json['faces']).astype(np.int32)
            rgba = np.array(request_json['rgba']).astype(np.float32)
            node = Mesh(engine, vertices, faces, rgba)
            self._add_node(node)
            return

        log.error(f'unhandled request with {request_json=}')
        raise NotImplementedError

    # EVENT HANDLING ---------------------------------------------------------------------------------------------------
    def _resize_event(self, width: int, height: int):
        self.camera.projection.update(aspect_ratio=self._window.aspect_ratio)
        self._imgui.resize(width, height)

    def _close_event(self):
        for node in self._scene:
            assert isinstance(node, Node)
            node.destroy()

    @property
    def _is_right_mouse_button_pressed(self) -> bool:
        return self._mouse_button_pressed == 2

    def _key_event(self, key: int, action: str, modifiers: KeyModifiers):
        if self._is_right_mouse_button_pressed:
            self.camera.key_input(key, action, modifiers)

        else:
            self.camera.disable_movement()

        self._imgui.key_event(key, action, modifiers)

    def _mouse_press_event(self, x: int, y: int, button: int):
        self._mouse_button_pressed = button

        if button == 2:
            self._set_mouse_visible(False)

        self._imgui.mouse_press_event(x, y, button)

    def _mouse_release_event(self, x: int, y: int, button: int):
        self._mouse_button_pressed = None
        self._set_mouse_visible(True)
        self._imgui.mouse_release_event(x, y, button)

    def _mouse_position_event(self, x: int, y: int, dx: int, dy: int):
        self._imgui.mouse_position_event(x, y, dx, dy)

    def _mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        if self._is_right_mouse_button_pressed:
            self.camera.rot_state(-dx, -dy)

        self._imgui.mouse_drag_event(x, y, dx, dy)

    def _mouse_scroll_event(self, x_offset: int, y_offset: int):
        if y_offset > 0:
            # Speed up the movement
            self.camera.velocity += 1
        else:
            # Slow down the movement
            self.camera.velocity = max(1, self.camera.velocity - 1)

        self._imgui.mouse_scroll_event(x_offset, y_offset)

    def _unicode_char_entered_event(self, char: str):
        self._imgui.unicode_char_entered(char)
