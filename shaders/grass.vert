#version 460

// Per-vertex blade-mesh attributes (5-vert ribbon).
layout(location = 0) in vec3 inPos;       // local-space (y in [0, blade_h])
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;        // uv.y = height ratio 0..1

// Per-instance blade attributes (one entry per blade in the
// instance buffer; matches GrassBlade in grass.h).
layout(location = 3) in vec4 inWorldPos;  // .xyz = blade base, .w = pad
layout(location = 4) in vec4 inRotHeight; // .x = rotation_y, .y = height_factor, .zw = pad
layout(location = 5) in vec4 inTint;      // .xyz = tint, .w = pad

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
    vec4  grass_extra;   // x: height_scale, y: alpha_cutoff
} scene;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    // .x = max grass distance (m). Beyond this, vertex collapses to NaN
    //      so the rasteriser drops the triangle.
    // .y = wind strength
    // .z = time (seconds, used by wind sin)
    // .w = unused
    vec4 grass_params;
} pc;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec3 vColor;
layout(location = 2) out vec2 vUv;
layout(location = 3) out float vHeightRatio;
layout(location = 4) out float vCullKill;
layout(location = 5) out vec3 vWorldPos;

mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(
        vec3( c, 0.0,  s),
        vec3(0.0, 1.0, 0.0),
        vec3(-s, 0.0,  c)
    );
}

void main() {
    float rotation     = inRotHeight.x;
    float height_scale = inRotHeight.y;
    vec3  base_world   = inWorldPos.xyz;

    // Local position scaled by per-blade height variation × the
    // global grass height slider so users can dial blade height
    // up/down in real time.
    vec3 lp = inPos;
    lp.y *= height_scale * scene.grass_extra.x;

    // ---- Wind: a moving wave (gust band) travelling across the world.
    // The whole field doesn't bow at once — only blades the gust is
    // currently passing through bend strongly, others stay still.
    // Implementation:
    //   1. Pick a wind direction (slowly rotating so gusts come from
    //      different headings over time).
    //   2. Project blade XZ onto that direction — `along` is metres
    //      along the wind axis.
    //   3. `phase = along - speed * t` slides the wave with time.
    //   4. Multiple gust crests spaced `period` apart, each a smooth
    //      pulse of `width` metres — pulse() goes from 0 outside the
    //      band to 1 at the centre.
    //   5. Modulate gust amplitude by low-frequency noise so each
    //      gust has variable thickness/strength (some weak, some
    //      strong). And add a small per-blade phase offset so
    //      neighbours don't move in lock-step.
    //   6. Bend = pulse * thickness * (height_ratio^2) so the base
    //      anchors and the tip moves most.
    float wind_strength = pc.grass_params.y;
    float t             = pc.grass_params.z;

    // Slow-rotating wind direction so the field doesn't always sway
    // the same way. 0.04 rad/s ≈ a 26° drift over 10 sec.
    float wind_yaw = 0.5 + sin(t * 0.04) * 0.3;
    vec2  wind_dir = vec2(cos(wind_yaw), sin(wind_yaw));

    // Project blade XZ onto wind axis (along) and the perpendicular
    // axis (across). The gust front isn't a perfect straight line —
    // we wobble its position based on `across` so the wave reads as
    // an organic gust boundary, not a marching ruler line.
    vec2  perp_dir = vec2(-wind_dir.y, wind_dir.x);
    float along   = dot(base_world.xz, wind_dir);
    float across  = dot(base_world.xz, perp_dir);

    // ---- Primary gust wave: wider band, lower speed ----
    // Wider pulse + larger period = each gust is a clear "wall" of
    // motion that takes a couple of seconds to pass over you, with
    // calm in between. Speed 5 m/s ≈ a brisk breeze.
    const float kPeriod = 55.0;
    const float kWidth  = 14.0;
    float       speed   = 5.0;

    // Front wobble: shift the gust phase by a low-frequency cross-axis
    // noise so the gust front is wavy/curved, not a perfectly straight
    // line. Amplitude ~6 m. The two sin terms at different freqs give
    // a non-repeating-feeling boundary.
    float front_wobble = sin(across * 0.12 + t * 0.7) * 4.0 +
                         sin(across * 0.31 + t * 1.1) * 2.0;
    float main_phase = along - speed * t + front_wobble;
    float wrapped    = mod(main_phase, kPeriod) - kPeriod * 0.5;
    float pulse = 0.0;
    if (abs(wrapped) < kWidth) {
        float u = wrapped / kWidth;
        pulse = 0.5 + 0.5 * cos(u * 3.14159265);
    }

    // Variable thickness — different gusts hit harder than others.
    float gust_id_pos = main_phase - wrapped;
    float thickness = 0.40 + 0.60 *
                      (0.5 + 0.5 * sin(gust_id_pos * 0.045 + 1.7));

    // Occasional bigger burst — slow low-frequency multiplier so once
    // every ~30 s the entire field gets a stronger gust.
    float burst = 0.85 + 0.45 * sin(t * 0.18 + along * 0.005);

    // Per-blade jitter so neighbours don't move identically.
    float jitter = sin(base_world.x * 0.41 +
                       base_world.z * 0.73 + t * 0.9) * 0.15;

    float gust = pulse * thickness * burst * (1.0 + jitter);

    // Idle breeze: a tiny per-blade sway that's always on, so blades
    // outside the gust band aren't dead-still. Amplitude ~15% of the
    // wind strength, two phases so X and Z don't oscillate in lockstep
    // (looks like a gentle drifting jitter rather than a back-and-
    // forth metronome).
    float idle_phase_a = base_world.x * 0.41 + base_world.z * 0.27;
    float idle_phase_b = base_world.x * 0.19 + base_world.z * 0.53;
    vec2  idle_sway = vec2(sin(t * 1.3 + idle_phase_a),
                           cos(t * 0.9 + idle_phase_b)) * 0.15 * wind_strength;

    float bend = inUv.y * inUv.y;
    vec2  sway_xz = wind_dir * gust * wind_strength + idle_sway;
    lp.x += sway_xz.x * bend;
    lp.z += sway_xz.y * bend;

    // Per-blade Y rotation, then translate to world.
    mat3 R = rotY(rotation);

    // Smooth distance fade — blades shrink to zero over the last 25%
    // of the max-distance slider so the cull doesn't pop visibly.
    float view_dist_base = distance(base_world, scene.camera_pos.xyz);
    float fade_start = pc.grass_params.x * 0.75;
    float fade = 1.0 - smoothstep(fade_start, pc.grass_params.x, view_dist_base);
    lp.y *= fade;

    // Slope fade — blades whose stored heightmap-normal Y is below
    // the user threshold shrink toward 0 height. Smoothstep makes
    // the slope-density transition gradual instead of cutting blades
    // off cliff-flat at one slope value.
    float slope_n  = inRotHeight.z;
    float slope_th = scene.grass_extra.z;
    float slope_fade = smoothstep(slope_th - 0.10, slope_th + 0.05, slope_n);
    lp.y *= slope_fade;

    vec3 world = R * lp + base_world;

    // Hard cull: well beyond max distance, collapse to NaN clip space
    // so the rasteriser drops the triangle entirely.
    vCullKill = step(pc.grass_params.x + 5.0, view_dist_base);
    if (vCullKill > 0.5) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        vNormal = vec3(0.0, 1.0, 0.0);
        vColor  = vec3(0.0);
        vUv     = vec2(0.0);
        vHeightRatio = 0.0;
        vWorldPos = vec3(0.0);
        return;
    }

    gl_Position = pc.mvp * vec4(world, 1.0);

    // Normal in world space — the blade billboards roughly toward
    // camera by virtue of its random rotation; for shading we use a
    // mostly-up normal mixed with a slight side bias. Good enough for
    // foliage reads.
    vNormal = normalize(R * vec3(0.0, 0.8, 0.6));
    vColor  = inTint.xyz;
    vUv     = inUv;
    vHeightRatio = inUv.y;
    vWorldPos = world;
}
