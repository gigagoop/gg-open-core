import logging
import pickle
from typing import Optional

import numpy as np
import zmq

from gigagoop.coord import Transform
from gigagoop.camera import PinholeCamera

from ..nodes import Grid, CFrame, Scatter, Plot, ThickPlot, Mesh, Sphere, Image, UnrealMesh, Arrow, LitMesh
from ..params import WindowParams

log = logging.getLogger(__name__)

from .base_engine import BaseEngine


class ServerEngine(BaseEngine):
    def __init__(self,
                 port: int,
                 window_params: Optional[WindowParams] = None,
                 cam: Optional[PinholeCamera] = None):

        super().__init__(window_params, cam)

        # Setup ZeroMQ
        self._zmq_port = port
        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.REP)
        self._zmq_socket.bind(f'tcp://*:{port}')

    def render_frame(self, current_time: float, delta_time: float):
        super().render_frame(current_time, delta_time)

        # Check if any ZeroMQ messages have been received
        try:
            # In a non-blocking way, get the message (as a chunk of bytes)
            request_msg = self._zmq_socket.recv(flags=zmq.DONTWAIT)

            # We are "just" passing around json, as it is fast and flexible...
            request = pickle.loads(request_msg)

            # Do something with the message
            optional = self._handle_message(request)

            # Ack to the client, calm them down, and let them know we got the message
            response = {'MessageType': 'Ack',
                        'Success': True,
                        'Optional': optional}
            response_msg = pickle.dumps(response)
            self._zmq_socket.send(response_msg)

        except zmq.Again:
            pass    # no message received (this is not an error), continuing...

    def _handle_message(self, request):
        # Validate the message type
        assert 'MessageType' in request
        message_type = request['MessageType']
        engine = self

        ret_message = None

        if message_type == 'IsAlive':
            pass

        elif message_type == 'add_cframe':
            M_OBJ_WCS = Transform(request['M_OBJ_WCS'])
            colors = request['colors']
            node = CFrame(engine, M_OBJ_WCS, colors=colors)
            self.add_node(node)

        elif message_type == 'get_camera_pose':
            M_CAM_WCS = self.camera.M_CAM_WCS
            ret_message = {'M_CAM_WCS': M_CAM_WCS.matrix}

        elif message_type == 'set_camera_pose':
            M_CAM_WCS = Transform(request['M_CAM_WCS'])
            location, rotation = M_CAM_WCS.to_unreal()
            self._default_camera_pose_location = location
            self._default_camera_pose_rotation = rotation
            self._ui_set_default_camera_pose()

        elif message_type == 'get_image_snapshot':
            buffer_bytes = self.screen.read(components=4)
            image = np.frombuffer(buffer_bytes, dtype=np.uint8).reshape((self.screen.height, self.screen.width, 4))
            image = np.flipud(image)
            ret_message = {'rgba': image}

        elif message_type == 'add_xy_grid':
            M_OBJ_WCS = Transform(request['M_OBJ_WCS'])
            rgba = np.array(request['rgba']).astype(np.float32)
            size = int(request['size'])
            node = Grid(engine, M_OBJ_WCS, rgba, n_lines=size)
            self.add_node(node)

        elif message_type == 'add_image':
            M_OBJ_WCS = Transform(request['M_OBJ_WCS'])
            image = request['image']
            width = request['width']
            height = request['height']
            alpha = request['alpha']
            node = Image(engine, M_OBJ_WCS, image, width, height, alpha)
            self.add_node(node)

        elif message_type == 'add_unreal_mesh':
            obj_file = request['obj_file']
            node = UnrealMesh(engine, obj_file)
            self.add_node(node)

        elif message_type == 'add_sphere':
            M_OBJ_WCS = Transform(request['M_OBJ_WCS'])
            rgba = np.array(request['rgba']).astype(np.float32)
            node = Sphere(engine, M_OBJ_WCS, rgba, scale=request['size'])
            self.add_node(node)

        elif message_type == 'add_arrow':
            origin = np.array(request['origin']).astype(np.float32)
            direction = np.array(request['direction']).astype(np.float32)
            width = np.array(request['width']).astype(np.float32)
            rgba = np.array(request['rgba']).astype(np.float32)
            node = Arrow(engine, origin, direction, width, rgba)
            self.add_node(node)

        elif message_type == 'scatter':
            position = np.array(request['position']).astype(np.float32)
            rgba = np.array(request['rgba']).astype(np.float32)
            size = float(request['size'])
            node = Scatter(engine, position, rgba, size)
            self.add_node(node)

        elif message_type == 'plot':
            position = np.array(request['position']).astype(np.float32)
            rgba = np.array(request['rgba']).astype(np.float32)
            lines = request['lines']
            linewidth = request.get('linewidth')
            if linewidth is None:
                node = Plot(engine, position, rgba, lines)
                self.add_node(node)
                return ret_message

            try:
                node = ThickPlot(engine, position, rgba, lines, float(linewidth))
            except ValueError:
                node = Plot(engine, position, rgba, lines)
            self.add_node(node)

        elif message_type == 'mesh':
            vertices = np.array(request['vertices']).astype(np.float32)
            faces = np.array(request['faces']).astype(np.int32)
            rgba = np.array(request['rgba']).astype(np.float32)
            node = Mesh(engine, vertices, faces, rgba)
            self.add_node(node)

        elif message_type == 'lit_mesh':
            node = LitMesh(engine=self,
                           vertices=np.array(request['vertices']).astype(np.float32),
                           faces=np.array(request['faces']).astype(np.int32),
                           normals=np.array(request['normals']).astype(np.float32),
                           rgba=np.array(request['rgba']).astype(np.float32),
                           light_pos=np.array(request['light_pos']).astype(np.float32),
                           light_color=np.array(request['light_color']).astype(np.float32),
                           light_intensity=float(request['light_intensity']),
                           ambient_light=np.array(request['ambient_light']).astype(np.float32))

            self.add_node(node)

        else:
            log.error(f'unhandled request with {request=}')
            raise NotImplementedError

        return ret_message
