#version 330

in vec4 p_cam;
out float perspective_depth;

void main() {
    perspective_depth = length(p_cam.xyz);
}