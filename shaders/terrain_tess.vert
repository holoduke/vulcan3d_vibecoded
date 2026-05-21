#version 460

// Pass-through vertex stage for the near-terrain GPU tessellation
// pipeline. Runs once per patch control point (3 per triangle patch);
// the real work — distance-adaptive subdivision + detail displacement
// + normal recompute — happens in the .tesc / .tese stages.
//
// The terrain mesh draw uses an identity model matrix, so object space
// == world space here (same assumption terrain.vert relies on).

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(location = 0) out vec3 cpPos;
layout(location = 1) out vec3 cpNormal;
layout(location = 2) out vec2 cpUv;

void main() {
    cpPos    = inPosition;
    cpNormal = inNormal;
    cpUv     = inUv;
}
