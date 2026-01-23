#version 330

uniform mat4 M_CAM_IMG;     // projection matrix
uniform mat4 M_WCS_CAM;     // view matrix
uniform mat4 M_OBJ_WCS;     // model matrix

in vec3 in_vert;
in vec2 in_uv;

out vec2 uv;

void main() {
    gl_Position = M_CAM_IMG * M_WCS_CAM * M_OBJ_WCS * vec4(in_vert, 1.0);
    uv = in_uv;
}
