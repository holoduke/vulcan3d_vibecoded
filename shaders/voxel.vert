#version 460

// Voxel shape bounding-box vertex shader (multi-shape, rotation-capable).
//
// Each draw passes a rigid transform (rotation R + translation T) that
// maps shape-local coords → world. Draws the world-space AABB of the
// shape as a 12-tri unit cube (36 indices derived from gl_VertexIndex).
// Fragment shader does the brick-DDA from there.
//
// PC layout is R + T instead of a single mat4 so the fragment shader can
// build the world→local inverse analytically (transpose(R), -T) for
// free instead of running glsl's full mat4 inverse() per fragment.

layout(push_constant) uniform PC {
    mat3  R;            // shape-local→world rotation (48 bytes, std430)
    vec4  T;            // .xyz = translation, .w pad
    vec4  dims_vs;      // xyz = shape extent in metres, w = voxel size
    ivec4 grid_dir;     // xyz = brick grid dims, w = directory base offset
} pc;

layout(set = 0, binding = 0) uniform CameraUbo {
    mat4 view_proj;
    mat4 prev_view_proj;
    vec4 camera_pos;
    vec4 sun_dir;
    vec4 sun_color;
    vec4 ambient;
    vec4 viewport;
    vec4 pal[16];
} cam;

layout(location = 0) out vec3 vWorldPos;

const ivec3 kCorner[8] = ivec3[8](
    ivec3(0,0,0), ivec3(1,0,0), ivec3(0,1,0), ivec3(1,1,0),
    ivec3(0,0,1), ivec3(1,0,1), ivec3(0,1,1), ivec3(1,1,1)
);

// CCW winding from outside. Pipeline uses cull = FRONT so only the far
// faces rasterise (one fragment per pixel of the volume, works inside or
// outside the box).
const int kIdx[36] = int[36](
    0, 4, 6,  0, 6, 2,
    1, 3, 7,  1, 7, 5,
    0, 1, 5,  0, 5, 4,
    2, 6, 7,  2, 7, 3,
    0, 2, 3,  0, 3, 1,
    4, 5, 7,  4, 7, 6
);

void main() {
    int  ci    = kIdx[gl_VertexIndex];
    vec3 unit  = vec3(kCorner[ci]);
    vec3 local = unit * pc.dims_vs.xyz;        // shape-local AABB corner
    vec3 world = pc.R * local + pc.T.xyz;
    vWorldPos  = world;
    gl_Position = cam.view_proj * vec4(world, 1.0);
}
