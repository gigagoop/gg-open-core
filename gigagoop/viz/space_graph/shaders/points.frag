#version 330 core

in vec4 color;
out vec4 frag_color;

// The code commented below creates a nice looking circle sprite, which was thought to be preferred..., however, this
// ended up not looking great in a few experiments. For now, this will be commented out - in the future, we should use
// a branch to enable different sprite options, similar to `plt.plot(..., marker='*')`.
//void main() {
//    vec2 center = vec2(0.5);
//    float radius = 0.5;
//
//    float distance_to_center = length(gl_PointCoord.xy - center);
//    float alpha = 1.0 - smoothstep(radius - 0.05, radius, distance_to_center);
//
//    if (alpha < 0.01) {
//        discard;
//    }
//
//    frag_color = vec4(color.rgb, alpha);
//}

void main() {
    frag_color = color;
}