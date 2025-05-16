#version 330 core

uniform mat4 M_CAM_IMG;      // projection matrix
uniform mat4 M_WCS_CAM;      // view matrix
uniform mat4 M_OBJ_WCS;      // model matrix

in vec3 in_vert;
in vec3 in_normal;
in vec4 in_color;

out vec4 color;
out vec3 normal;
out vec3 world_pos;

void main() {
    gl_Position = M_CAM_IMG * M_WCS_CAM * M_OBJ_WCS * vec4(in_vert, 1.0);
    color = in_color;
    normal = normalize(in_normal);  // normals already in WCS, just make sure they are normalized
    world_pos = vec3(M_OBJ_WCS * vec4(in_vert, 1.0));
}