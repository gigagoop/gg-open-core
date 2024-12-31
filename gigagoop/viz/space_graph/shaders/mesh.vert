#version 330 core

uniform mat4 M_CAM_IMG;      // projection matrix
uniform mat4 M_WCS_CAM;      // view matrix
uniform mat4 M_OBJ_WCS;      // model matrix

in vec3 in_vert;
in vec4 in_color;

out vec4 color;

void main() {
    gl_Position = M_CAM_IMG * M_WCS_CAM * M_OBJ_WCS * vec4(in_vert, 1.0);
    color = in_color;
}