#version 460

// Voxel shape bounding-box vertex shader.
//
// Draws the world-space AABB of a single VoxelShape as a 12-tri unit
// cube (36 indices). No vertex buffer — corners are derived from
// gl_VertexIndex. The fragment shader then does brick-DDA from the
// camera through this volume.

layout(push_constant) uniform PC {
    vec4  origin_world;   // xyz = AABB min in world space
    vec4  dims_world;     // xyz = AABB size (world meters) = dim_bricks * kBrickSize
    ivec4 dims_bricks;    // xyz = brick grid dims
    vec4  voxel_size;     // x = kVoxelSize (0.1), y = kBrickSize (1.6)
    vec4  shape_idx;      // reserved for multi-shape (Session B)
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

// 8 cube corners — index i maps to (i&1, (i>>1)&1, (i>>2)&1).
const ivec3 kCorner[8] = ivec3[8](
    ivec3(0,0,0), ivec3(1,0,0), ivec3(0,1,0), ivec3(1,1,0),
    ivec3(0,0,1), ivec3(1,0,1), ivec3(0,1,1), ivec3(1,1,1)
);

// 12 triangles × 3 vertices, outward-facing winding (CCW from outside).
// Drawn with cull=FRONT so only the FAR faces rasterize, giving exactly
// one fragment shader invocation per pixel of the AABB — works from
// outside and inside the volume.
const int kIdx[36] = int[36](
    // -X face (x=0): corners 0,4,6,2
    0, 4, 6,  0, 6, 2,
    // +X face (x=1): corners 1,3,7,5
    1, 3, 7,  1, 7, 5,
    // -Y face (y=0): corners 0,1,5,4
    0, 1, 5,  0, 5, 4,
    // +Y face (y=1): corners 2,6,7,3
    2, 6, 7,  2, 7, 3,
    // -Z face (z=0): corners 0,2,3,1
    0, 2, 3,  0, 3, 1,
    // +Z face (z=1): corners 4,5,7,6
    4, 5, 7,  4, 7, 6
);

void main() {
    int  ci    = kIdx[gl_VertexIndex];
    vec3 unit  = vec3(kCorner[ci]);
    vec3 local = unit * pc.dims_world.xyz;
    vec3 world = pc.origin_world.xyz + local;
    vWorldPos  = world;
    gl_Position = cam.view_proj * vec4(world, 1.0);
}
