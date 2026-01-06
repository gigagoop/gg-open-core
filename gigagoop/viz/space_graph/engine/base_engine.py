import logging
from typing import List, Optional

import numpy as np
import imgui
import moderngl as mgl
import moderngl_window as mglw
from moderngl_window.context.base.keys import KeyModifiers
from moderngl_window.context.pyglet.window import PygletWrapper
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from matplotlib.colors import to_rgba

from gigagoop.coord import Transform, GazeAt
from gigagoop.camera import PinholeCamera

from ..nodes import Node
from ..camera import Camera
from ..params import WindowParams

from .utils.fps_tracker import FpsTracker
from .utils.gl_debug import GLDebug


log = logging.getLogger(__name__)


class BaseEngine:
    def __init__(self,
                 window_params: Optional[WindowParams] = None,
                 cam: Optional[PinholeCamera] = None,
                 opengl_debug: bool = False,
                 background_color: str = 'black'):

        # If the window or camera parameters are not passed in, we use the defaults. Take note that if only the window
        # parameters are passed in, then the camera is configured to match the window (it can be different if ya want)
        if window_params is None:
            window_params = WindowParams()

        if cam is None:
            cam = PinholeCamera.create_from_vfov(fov=53,
                                                 width=window_params.width,
                                                 height=window_params.height)

        # Set the background colors
        self._background_color = to_rgba(background_color)

        # To mimic Unreal Engine, we want to move the camera only if the right mouse button is pressed. To do this, we
        # need to track if the button is being pressed
        self._mouse_button_pressed = None

        # Get the window class to be used
        window_cls = mglw.get_window_cls(window='moderngl_window.context.pyglet.Window')

        # Instantiate the window class, creating a `class BaseWindow`
        window = window_cls(title=window_params.title,
                            size=(window_params.width, window_params.height),
                            fullscreen=False,
                            resizable=True,
                            gl_version=(3, 3),
                            aspect_ratio=window_params.aspect_ratio,
                            vsync=True,
                            samples=8,
                            cursor=True)

        window.ctx.clear(*self._background_color)

        window.print_context_info()
        mglw.activate_context(window=window)

        # Attach the OpenGL debug interface
        self.gl_debug = GLDebug(enabled=opengl_debug)

        # Construct a camera
        self.camera = Camera(cam, keys=window.keys)

        self._default_camera_pose_location, self._default_camera_pose_rotation = GazeAt(eye=[-10, 10, 10],
                                                                                        target=[0, 0, 0]).to_unreal()
        self._ui_set_default_camera_pose()

        # Construct an imgui instance for all the ui elements
        imgui.create_context()

        self._imgui = ModernglWindowRenderer(window)

        self._fps_tracker = FpsTracker()

        # Register callbacks
        window.resize_func = self.resize_event
        window.close_func = self.close_event
        window.key_event_func = self.key_event
        window.mouse_press_event_func = self.mouse_press_event
        window.mouse_release_event_func = self.mouse_release_event
        window.mouse_position_event_func = self.mouse_position_event
        window.mouse_drag_event_func = self.mouse_drag_event
        window.mouse_scroll_event_func = self.mouse_scroll_event
        window.unicode_char_entered_func = self.unicode_char_entered_event

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
        window.swap_buffers()
        window.set_default_viewport()

        # The scene contains all the nodes to render
        self._scene: List[Node] = []

    @property
    def ctx(self) -> mgl.Context:
        return self._window.ctx

    @property
    def screen(self) -> mgl.Framebuffer:
        return self._window.ctx.screen

    def run(self):
        """Start the rendering loop for the engine.

        This method is a blocking call, as it keeps executing until the window is closed or the application is
        terminated. It handles window clearing, frame rendering, and buffer swapping during each iteration.
        """
        window = self._window
        timer = self._timer

        try:
            while not window.is_closing:
                current_time, delta_time = timer.next_frame()

                self.render_frame(current_time, delta_time)
                self.render_ui(current_time, delta_time)

                if not window.is_closing:
                    window.swap_buffers()

            window.destroy()
            self._imgui.shutdown()

        except KeyboardInterrupt:
            pass

    def render_frame(self, current_time: float, delta_time: float):
        """Render to "the frame" for display.

        This is the main entry point for rendering to the screen. "The frame" is the rendering the frame buffer managed
        by the windowing system. If you are subclassing this engine, you may in some cases need more control over the
        rendering, so you should overload this method and do your own thang'.
        """
        self.gl_debug.push_debug_group('render frame')

        # Set up the OpenGL state to render the frame. We cannot "just" set up the state in the constructor once, since
        # `imgui` will change the OpenGL state; therefore, we need to set it each time we are performing a render.
        self.ctx.enable_only(mgl.DEPTH_TEST | mgl.PROGRAM_POINT_SIZE | mgl.BLEND)
        self.ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)
        self.ctx.blend_equation = mgl.FUNC_ADD
        self.ctx.depth_func = '<'

        # Render the scene
        self.screen.use()
        self.screen.clear(*self._background_color)

        for node in self._scene:
            node.render()

        self.gl_debug.pop_debug_group()

    def ui_render_delegate(self, current_time: float, delta_time: float):
        """This delegate is fired during ui rendering.

        The `imgui` package is used for ui rendering which follows a pattern of
            imgui.new_frame()
            ...
            imgui.render()

        If you would like to inject your own ui elements, then overload this delegate and define your `imgui` stuff
        inside it. The delegate will get fired between the `imgui.new_frame()` and `imgui.render().

        As an example,
            class Viewer(BaseEngine):
                def ui_render_delegate(self, current_time: float, delta_time: float):
                    imgui.begin('Monkeys', True)
                    imgui.text('They are funny!')
                    imgui.end()
        """
        ...

    def render_ui(self, current_time: float, delta_time: float):
        """Render the user interface.

        This is the main render function for the ui components. In general, you should not touch this but instead modify
        the ui render delegate when you are extending ui functionality. If you want to totally redo the ui, then take
        over this method and do whatever bad things you desire.
        """
        self.gl_debug.push_debug_group('render ui')

        # ==============================================================================================================
        # SETUP
        # ==============================================================================================================
        self.screen.use()    # main fbo (screen) must be active for ui rendering
        self._fps_tracker.update(delta_time)
        imgui.new_frame()

        # ==============================================================================================================
        # STATS
        # ==============================================================================================================
        imgui.begin('Stats', closable=True)
        imgui.text(f'{self._fps_tracker.fps:0.2f} FPS')
        imgui.text(f'{1/self._fps_tracker.fps*1000:0.2f} ms')
        imgui.text(f'{self.camera.velocity:0.2f} (cam vel)')
        imgui.end()

        # ==============================================================================================================
        # CAMERA
        # ==============================================================================================================
        camera = self.camera
        imgui.begin('Camera', closable=True)

        def f_fmt(val):
            if np.isclose(val, 0.0):
                val = 0.0
            return f'{val:0.3f}'

        # LOCATION
        # --------
        ENTER_RETURNS_TRUE = imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
        imgui.text('Location:')

        # LOC_X
        imgui.same_line()
        imgui.push_item_width(100)
        changed, new_text = imgui.input_text('##loc_x', f_fmt(camera.loc_x), 256, ENTER_RETURNS_TRUE)
        if changed:
            try:
                new_float = float(new_text)
                camera.loc_x = new_float
            except:
                pass
        imgui.pop_item_width()

        # LOC_Y
        imgui.same_line()
        imgui.push_item_width(100)
        changed, new_text = imgui.input_text('##loc_y', f_fmt(camera.loc_y), 256, ENTER_RETURNS_TRUE)
        if changed:
            try:
                new_float = float(new_text)
                camera.loc_y = new_float
            except:
                pass
        imgui.pop_item_width()

        # LOC_Z
        imgui.same_line()
        imgui.push_item_width(100)
        changed, new_text = imgui.input_text('##loc_z', f_fmt(camera.loc_z), 256, ENTER_RETURNS_TRUE)
        if changed:
            try:
                new_float = float(new_text)
                camera.loc_z = new_float
            except:
                pass
        imgui.pop_item_width()

        # ROTATION
        # --------
        imgui.text('Rotation:')

        # ROT_X
        imgui.same_line()
        imgui.push_item_width(100)
        changed, new_text = imgui.input_text('##rot_x', f_fmt(camera.rot_x), 256, ENTER_RETURNS_TRUE)
        if changed:
            try:
                new_float = float(new_text)
                camera.rot_x = new_float
            except:
                pass
        imgui.pop_item_width()

        # ROT_Y
        imgui.same_line()
        imgui.push_item_width(100)
        changed, new_text = imgui.input_text('##rot_y', f_fmt(camera.rot_y), 256, ENTER_RETURNS_TRUE)
        if changed:
            try:
                new_float = float(new_text)
                camera.rot_y = new_float
            except:
                pass
        imgui.pop_item_width()

        # ROT_Z
        imgui.same_line()
        imgui.push_item_width(100)
        changed, new_text = imgui.input_text('##rot_z', f_fmt(camera.rot_z), 256, ENTER_RETURNS_TRUE)
        if changed:
            try:
                new_float = float(new_text)
                camera.rot_z = new_float
            except:
                pass
        imgui.pop_item_width()

        # RESET
        if imgui.button('Reset', width=60, height=20):
            self._ui_set_default_camera_pose()
            self.camera.velocity = 10
        imgui.end()

        # ==============================================================================================================
        # FINISH
        # ==============================================================================================================
        self.ui_render_delegate(current_time, delta_time)
        imgui.render()
        self._imgui.render(imgui.get_draw_data())

        self.gl_debug.pop_debug_group()

    def add_node(self, node: Node):
        """Add a node to the scene for rendering."""
        assert isinstance(node, Node)
        self._scene.append(node)

    def _ui_set_default_camera_pose(self):
        self.camera.M_CAM_WCS = Transform.from_unreal(location=self._default_camera_pose_location,
                                                      rotation=self._default_camera_pose_rotation)

    # EVENT HANDLING ---------------------------------------------------------------------------------------------------
    @property
    def _is_right_mouse_button_pressed(self) -> bool:
        return self._mouse_button_pressed == 2

    def resize_event(self, width: int, height: int):
        self._imgui.resize(width, height)

    def close_event(self):
        for node in self._scene:
            assert isinstance(node, Node)
            node.destroy()

    def key_event(self, key: int, action: str, modifiers: KeyModifiers):
        if self._is_right_mouse_button_pressed:
            self.camera.key_input(key, action, modifiers)

        else:
            self.camera.disable_movement()

        self._imgui.key_event(key, action, modifiers)

    def mouse_press_event(self, x: int, y: int, button: int):
        self._mouse_button_pressed = button

        if button == 2:
            self._set_mouse_visible(False)

        self._imgui.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int):
        self._mouse_button_pressed = None
        self._set_mouse_visible(True)
        self._imgui.mouse_release_event(x, y, button)

    def mouse_position_event(self, x: int, y: int, dx: int, dy: int):
        self._imgui.mouse_position_event(x, y, dx, dy)

    def mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        if self._is_right_mouse_button_pressed:
            self.camera.mouse_input(dx, dy)

        self._imgui.mouse_drag_event(x, y, dx, dy)

    def mouse_scroll_event(self, x_offset: int, y_offset: int):
        """
        On scroll: choose a coarse step of 1.0 when velocity > 1.0,
        otherwise a fine step of 0.01.  Round to int only in the coarse region.
        """
        # Choose step: coarse int steps above 1.0, fine-grained below
        coarse = self.camera.velocity > 1.0
        step = 1.0 if coarse else 0.1

        # Compute new velocity, clamped at the minimum step
        if y_offset > 0:
            new_v = self.camera.velocity + step
        else:
            new_v = max(step, self.camera.velocity - step)

        self.camera.velocity = int(round(new_v)) if coarse else new_v

        self._imgui.mouse_scroll_event(x_offset, y_offset)

    def unicode_char_entered_event(self, char: str):
        self._imgui.unicode_char_entered(char)

    @property
    def frame_index(self) -> int:
        return self._fps_tracker.frame_index



