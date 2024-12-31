from __future__ import annotations


class GLDebug:
    """
    This is a wrapper around some OpenGL debug capabilities. An attempt is made to import OpenGL only *if* it needs to
    be (which is set by enabling the capability).
    """
    _gl = None

    def lazy_import_opengl(self):
        if self._gl is None:
            from OpenGL import GL
            self._gl = GL

        return self._gl

    def __init__(self, enabled: bool):
        self._enabled = enabled

        if not enabled:
            return

        gl = self.lazy_import_opengl()

        # Make sure the debug extension is available
        debug_extension_available = False
        num_extensions = gl.glGetIntegerv(gl.GL_NUM_EXTENSIONS)
        for i in range(num_extensions):
            extension = gl.glGetStringi(gl.GL_EXTENSIONS, i)
            if extension and extension.decode() == 'GL_KHR_debug':
                debug_extension_available = True
                break

        assert debug_extension_available, '`GL_KHR_debug` is not available, and this is a problem... you are trying ' \
                                          'to use OpenGL debugging, but it is not supported on your GPU ;-P'

        gl.glEnable(gl.GL_DEBUG_OUTPUT)
        gl.glEnable(gl.GL_DEBUG_OUTPUT_SYNCHRONOUS)

    def push_debug_group(self, msg: str):
        if not self._enabled:
            return

        gl = self.lazy_import_opengl()

        gl.glPushDebugGroup(gl.GL_DEBUG_SOURCE_APPLICATION, 1, -1, msg.encode('utf-8'))

    def pop_debug_group(self):
        if not self._enabled:
            return

        gl = self.lazy_import_opengl()

        gl.glPopDebugGroup()


