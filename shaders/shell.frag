#version 460
#extension GL_EXT_nonuniform_qualifier : require

// SPOM silhouette shell pass: fragment stage.
// Each fragment is on the SHELL TOP (the rasterised face is the outward-
// extruded original brush face). March the parallax height map starting
// at the shell-top fragment, going INWARD along the view direction. If
// the march hits a brick top within the shell band (depth ≤ shell_t),
// draw that brick's texture color. Else discard — that's a "gap" between
// bricks past the brush silhouette, so sky shows through.
//
// Writes scene_color (location 0) + motion_vec=0 (location 1) so the
// pipeline matches the main pass's 2-color-attachment layout. The depth
// gets written to the brick TOP's projected depth so downstream passes
// see correct geometry there.

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vWorldPos;
layout(location = 2) in vec2 vUv;
layout(location = 3) flat in vec4 vTexParams;
layout(location = 4) in vec3 vShellBase;
layout(location = 5) in float vShellT;
layout(location = 6) in float vOrigDepth;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;

const int kTextureCount = 16;
layout(set = 0, binding = 3) uniform sampler2D u_albedo[kTextureCount];
layout(set = 0, binding = 12) uniform sampler2D u_spom_height[kTextureCount];

// Depth buffer sampler. Bound in set 1 to avoid touching scene_desc_set_layout_.
layout(set = 1, binding = 0) uniform sampler2D u_depth_buffer;

layout(set = 0, binding = 0) uniform SceneUBO {
    vec4  sun_direction;
    vec4  sun_color;
    vec4  ambient;
    vec4  sky_color;
    ivec4 rt_flags;
    vec4  rt_params;
    ivec4 rt_flags2;
    vec4  rt_params2;
    vec4  camera_pos;
    vec4  rt_lod;
    vec4  viewport;
    vec4  muzzle_pos;
    vec4  muzzle_color;
    vec4  terrain_params;
    vec4  terrain_h_low;
    vec4  terrain_h_high;
    vec4  grass_extra;
    vec4  grass_extra2;
    mat4  light_vp;
    vec4  terrain_extra;
    vec4  _scene_pad[24];
    vec4  restir_params;
    vec4  spom_params;
} scene;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
} pc;

int height_idx_for_albedo(int a) {
    if (a == 1) return 0;
    if (a == 4) return 1;
    return -1;
}

void main() {
    int albedo_idx = int(vTexParams.x);
    int hidx       = height_idx_for_albedo(albedo_idx);
    if (hidx < 0 || vShellT < 0.0001) discard;

    vec3 N = normalize(vNormal);
    if (abs(N.y) > 0.5) discard;  // not a wall face -- floors handled by main pass

    // Discard if anything was drawn at this pixel in the main pass.
    //
    // Earlier attempt: compare against vOrigDepth (the un-extruded
    // face's clip-space depth). That broke for SSDM-mode walls because
    // cube.frag writes gl_FragDepth = parallax_hit_depth which is
    // slightly deeper than vOrigDepth, so the discard never fired and
    // shell overdrew the wall with simpler lighting (= "wrong shadows").
    //
    // The simpler robust check: if existing_depth is NOT sky (< 0.99),
    // there's geometry there already — don't draw shell. Shell ends up
    // drawing ONLY in pixels where sky is behind (the silhouette
    // extension band past brush corners). Vert input vOrigDepth is no
    // longer needed (kept in interface for ABI stability).
    //
    // texelFetch (nearest, no filter): linear sampling would interpolate
    // across the wall→sky boundary and give false positives in a thin
    // band at edges. With nearest, each pixel gets exactly one of "wall
    // depth" or "sky depth", no blend.
    ivec2 px = ivec2(gl_FragCoord.xy);
    float existing_depth = texelFetch(u_depth_buffer, px, 0).r;
    if (existing_depth < 0.99) discard;

    // View direction (camera-to-fragment, normalised).
    vec3 V = normalize(vWorldPos - scene.camera_pos.xyz);

    // Triplanar UV (object-space, world-space-equivalent for axis-aligned brushes):
    // pick the lateral plane based on dominant normal.
    vec3 absN = abs(N);
    vec2 uv;
    vec2 dUV_dWorldX;  // unit per metre, used to convert march steps to UV
    vec2 dUV_dWorldZ;
    if (absN.x > absN.z) {
        // wall faces ±X: lateral UV = world (Z, Y)
        uv = vec2(vWorldPos.z, vWorldPos.y) * vTexParams.z;
    } else {
        // wall faces ±Z: lateral UV = world (X, Y)
        uv = vec2(vWorldPos.x, vWorldPos.y) * vTexParams.z;
    }

    // March parallax. View ray steps INWARD (along -N component of V).
    // Maximum march distance = vShellT (the shell thickness in world).
    // If we don't find a brick top within that distance, the brick at
    // this lateral UV doesn't reach up to our shell-top fragment ->
    // silhouette gap, discard.
    // Larger relief in the shell pass than cube.frag's main parallax.
    // Reason: the shell extends ~12cm past the geometric edge (shell.vert
    // kShellThickness) so we need brick bumps tall enough to cover that
    // band — otherwise even at brick TOPS the brick height (=4cm) doesn't
    // reach up to most of the shell band and we discard everything. Set
    // bump height == shell thickness so a height=1 sample exactly fills
    // the shell. The visible bump on the wall surface won't match
    // cube.frag's subtler parallax, but the silhouette extension is
    // what the user is after.
    const int   kMaxSteps    = 24;
    const float kHeightScale = 0.12;
    float strength  = max(0.0, scene.spom_params.x);
    float max_depth = kHeightScale * strength;
    // Shell-top sits vShellT ABOVE the un-extruded face. The brick TOP
    // (height = 1) lives at the un-extruded face (depth_world = 0 from
    // un-extruded face). So from the shell-top, the brick top is at
    // vShellT distance inward. The crevice (height = 0) is at vShellT +
    // max_depth.
    //
    // March from shell-top inward, accumulating world distance.
    // At each step, sample height map at the current lateral UV. The
    // brick surface at this UV sits at world depth from un-extruded face
    // = (1 - h) * max_depth. From shell-top, that's vShellT + (1 - h)*max_depth.
    // If our march has gone farther INWARD than that, we've crossed into
    // the brick surface -> hit.

    // View ray in world space, scaled so 1 unit of "t" = 1 metre.
    vec3 ray_dir = V;
    // Project ray onto the wall's normal plane to get the lateral component.
    // March step parameter: how far we step along V each iteration.
    float t = 0.0;
    float step = (vShellT + max_depth) / float(kMaxSteps);
    bool hit = false;
    vec3 hit_world = vec3(0.0);
    float hit_depth_from_base = 0.0;

    for (int i = 0; i < kMaxSteps; ++i) {
        t += step;
        vec3 p_world = vWorldPos + ray_dir * t;
        // Distance from un-extruded face (vShellBase plane along N):
        float depth_from_base = dot(p_world - vShellBase, -N);
        if (depth_from_base < 0.0) continue;  // still above un-extruded face
        // Lateral UV at this march point.
        vec2 puv;
        if (absN.x > absN.z) puv = vec2(p_world.z, p_world.y) * vTexParams.z;
        else                 puv = vec2(p_world.x, p_world.y) * vTexParams.z;
        float h = texture(u_spom_height[hidx], puv).r;
        float surface_depth = (1.0 - h) * max_depth;  // brick surface depth from un-extruded face
        if (depth_from_base >= surface_depth) {
            hit = true;
            hit_world = p_world;
            hit_depth_from_base = surface_depth;
            uv = puv;
            break;
        }
    }
    if (!hit) discard;  // silhouette gap -- no brick reaches this shell-top fragment

    // Sample brick albedo at the lateral UV of the hit.
    int  a_idx = clamp(albedo_idx, 0, kTextureCount - 1);
    vec3 tex_albedo = texture(u_albedo[a_idx], uv).rgb;
    vec3 albedo = pc.color.rgb * tex_albedo;

    // Basic lighting: sun N·L (approximate, since the brick face normal isn't
    // recomputed here -- use geometric N as approximation) + ambient.
    vec3 sun_dir = normalize(-scene.sun_direction.xyz);
    float ndotl  = max(0.0, dot(N, sun_dir));
    vec3 lit     = albedo * (scene.ambient.rgb + scene.sun_color.rgb * ndotl);

    outColor  = vec4(lit, 1.0);
    outMotion = vec2(0.0);

    // Write depth at the BRICK TOP's projected position so downstream passes
    // see correct geometry. The brick is at hit_world. Project to clip space:
    vec4 hit_clip = pc.mvp * (inverse(pc.model) * vec4(hit_world, 1.0));
    gl_FragDepth = hit_clip.z / hit_clip.w;
}
