#version 330

uniform mat4 M_CAM_IMG;      // projection matrix
uniform mat4 M_WCS_CAM;      // view matrix
uniform mat4 M_OBJ_WCS;      // model matrix

in vec3 in_vert;

out vec4 p_cam;

void main() {
    p_cam = M_WCS_CAM * M_OBJ_WCS * vec4(in_vert, 1.0);
    gl_Position = M_CAM_IMG * p_cam;
}