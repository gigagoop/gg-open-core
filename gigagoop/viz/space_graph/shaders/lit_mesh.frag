#version 330 core

uniform vec3 light_pos;        // Light position in world space
uniform vec3 light_color;      // Light color (e.g., vec3(1.0) for white)
uniform float light_intensity; // Light intensity (e.g., 1.0)
uniform vec3 ambient_light;    // Ambient light color (e.g., vec3(0.1))

in vec4 color;
in vec3 normal;
in vec3 world_pos;

out vec4 frag_color;

void main() {
    // Normalize inputs
    vec3 N = normalize(normal);
    vec3 L = normalize(light_pos - world_pos); // Light direction

    // Diffuse term
    float diffuse = max(dot(N, L), 0.0);

    // Combine ambient and diffuse
    vec3 lighting = (ambient_light + diffuse * light_color) * light_intensity;
    frag_color = vec4(color.rgb * lighting, color.a);
}