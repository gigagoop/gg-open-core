#version 330

uniform sampler2D image_texture;
uniform float ALPHA;

in vec2 uv;

out vec4 fragColor;

void main() {
    vec4 tex = texture(image_texture, uv);
    tex.a *= ALPHA;
    if (tex.a <= 0.001) {
        discard;
    }
    fragColor = tex;
}
