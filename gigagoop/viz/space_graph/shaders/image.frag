#version 330

uniform sampler2D image_texture;
uniform float ALPHA;

in vec2 uv;

out vec4 fragColor;

void main() {
    fragColor = texture(image_texture, uv);
    fragColor.a = ALPHA;
}