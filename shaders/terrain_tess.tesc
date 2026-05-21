#version 460

// Tessellation control: triangle patches (3 control points). Picks a
// distance-adaptive subdivision level — dense up close, ~none far away
// ("smart, not dense far away"). Each edge's level is derived ONLY from
// that edge's two endpoints, so two patches sharing an edge compute the
// identical level → no T-junction cracks.

layout(vertices = 3) out;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;       // .w = morph (unused for tessellated near chunks)
    vec4 emissive;
    vec4 tex_params;
} pc;

layout(location = 0) in  vec3 cpPos[];
layout(location = 1) in  vec3 cpNormal[];
layout(location = 2) in  vec2 cpUv[];

layout(location = 0) out vec3 tcPos[];
layout(location = 1) out vec3 tcNormal[];
layout(location = 2) out vec2 tcUv[];

// Tuning: full subdivision within kNear m, ramping to 1 (no extra
// geometry) by kFar m. kMaxTess caps the inner/outer level.
// Tuned for the ≤0.55 m per-type displacement: ~0.5–1 m triangles are
// plenty for that relief, so a modest max level + tighter range keeps
// open-vista views performant (kMaxTess 14 + 110 m cratered to ~25 fps
// overlooking the valley).
// Production values: dense enough near for smooth dm relief, ramps to
// ~1 by kFar so open vistas stay cheap.
// ~10x+ denser near mesh: max level 40 (≈16x the triangles of the
// old level-10 cap at the peak) held within kNear, then a concave
// (gamma) falloff so the heavy density stays in a small radius around
// the camera and the far ramp still reaches 1 by kFar — that keeps the
// total triangle budget bounded so open vistas don't crater.
// Runtime tess knobs come in via pc.emissive (set per-draw from the
// UI sliders): x = max level, y = full-detail distance (m),
// z = fade-out distance (m), w = falloff exponent. Fall back to sane
// defaults if a value is 0 (e.g. older callers).
float kMaxTessV() { return pc.emissive.x > 0.5  ? pc.emissive.x : 40.0; }
float kNearV()    { return pc.emissive.y > 0.01 ? pc.emissive.y : 7.0;  }
float kFarV()     { return pc.emissive.z > 0.01 ? pc.emissive.z : 60.0; }
float kFalloffV() { return pc.emissive.w > 0.01 ? pc.emissive.w : 0.55; }

// Clip-space w from the camera ≈ view-space distance — a good, cheap
// proxy for "how close is this control point" without needing the
// camera position in this stage.
float cpDist(int i) {
    vec4 c = pc.mvp * vec4(cpPos[i], 1.0);
    return max(c.w, 0.05);
}

float tessForDist(float d) {
    float nr = kNearV(), fr = max(kNearV() + 1.0, kFarV());
    float t = 1.0 - clamp((d - nr) / (fr - nr), 0.0, 1.0);
    t = pow(t, kFalloffV());       // concave: dense band hugs the camera
    return mix(1.0, kMaxTessV(), t);
}

// Edge level from the edge's two endpoints only (shared → crack-free).
float edgeTess(int a, int b) {
    return tessForDist(0.5 * (cpDist(a) + cpDist(b)));
}

void main() {
    tcPos[gl_InvocationID]    = cpPos[gl_InvocationID];
    tcNormal[gl_InvocationID] = cpNormal[gl_InvocationID];
    tcUv[gl_InvocationID]     = cpUv[gl_InvocationID];

    if (gl_InvocationID == 0) {
        // Vulkan triangle tess: outer[i] is the edge OPPOSITE vertex i,
        // i.e. the edge between the other two control points.
        float e0 = edgeTess(1, 2);
        float e1 = edgeTess(2, 0);
        float e2 = edgeTess(0, 1);
        gl_TessLevelOuter[0] = e0;
        gl_TessLevelOuter[1] = e1;
        gl_TessLevelOuter[2] = e2;
        gl_TessLevelInner[0] = max(e0, max(e1, e2));
    }
}
