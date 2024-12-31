#version 330 core

uniform vec4 GRID_RGBA;    // color of the grid

out vec4 f_color;

void main() {
    f_color = GRID_RGBA;
}