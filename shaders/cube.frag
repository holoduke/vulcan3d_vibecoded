#version 460
#extension GL_EXT_ray_query : require

layout(location = 0) in vec3 vNormal;
// flat — see cube.vert for the rationale (these are push-constant values
// that must not perspective-interpolate; the vTexParams.x precision drift
// was silently enabling texture sampling when textures were "off").
layout(location = 1) flat in vec3 vColor;
layout(location = 2) in vec3 vWorldPos;
layout(location = 3) flat in vec4 vEmissive;
layout(location = 4) in vec2 vUv;
layout(location = 5) flat in vec4 vTexParams;  // x: albedo idx, y: normal idx,
                                               // z: uv scale,
                                               // w: 0 = world-space triplanar
                                               //    1 = object-space triplanar
layout(location = 6) in vec3 vObjectPos;
layout(location = 7) in vec3 vObjectNormal;
layout(location = 8) in vec4 vPrevClip;  // prev_view_proj × prev_model × local_pos
layout(location = 0) out vec4 outColor;
// Per-pixel screen-space motion vector — current_uv minus prev_uv. Dynamic
// surfaces get correct reprojection because cube.vert applied prev_model
// (DynRender::prev_world × scale for dyn boxes; current model for static
// brushes / camera-attached viewmodel / sub-pixel particles). Future SVGF
// pass reads motion_vec_image_ + uses this delta to walk into history.
layout(location = 1) out vec2 outMotion;

layout(set = 0, binding = 0) uniform SceneUBO {
    vec4  sun_direction;
    vec4  sun_color;
    vec4  ambient;
    vec4  sky_color;
    ivec4 rt_flags;     // x:shadow_on, y:shadow_samples, z:ao_samples, w:frame
    vec4  rt_params;    // x:shadow_softness, y:ao_radius, z:ambient_strength, w:shadow_strength
    ivec4 rt_flags2;    // x:gi_samples, y:reflections_on, z:gi_bounces, w:ao_mode
                        //   ao_mode: 0=off, 1=fast (2 short rays), 2=RTAO (full)
    vec4  rt_params2;   // x:gi_strength, y:gi_radius, z:reflection_strength,
                        // w:shadow_curve (0=linear, 1=cubic)
    vec4  camera_pos;   // xyz = world-space eye
    vec4  rt_lod;       // x:lod_near, y:lod_far
    vec4  viewport;     // x:w, y:h, z:1/w, w:1/h — used by motion-vec output
    // Muzzle flash dynamic light. xyz = world-space origin, .w = intensity
    // (0 disables the contribution). Color/radius live in muzzle_color.
    vec4  muzzle_pos;
    vec4  muzzle_color; // rgb = color (linear), w = falloff radius (m)
} scene;

// Distance-based sample LOD. fragments within lod_near get full samples;
// fragments past lod_far drop to a single ray. Standard "RT becomes a luxury
// at distance" trick used in commercial engines.
int lod_samples(int requested, float dist) {
    float lod_t = clamp((dist - scene.rt_lod.x) /
                        max(0.001, scene.rt_lod.y - scene.rt_lod.x), 0.0, 1.0);
    int n = int(round(mix(float(requested), 1.0, lod_t)));
    return max(1, n);
}

layout(set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

struct Material {
    vec4 color;
    vec4 emissive;
    vec4 tex;       // x: albedo idx, y: normal idx, z: uv scale, w: spare
};
layout(set = 0, binding = 2, std430) readonly buffer Materials {
    Material materials[];
};

// Bound texture arrays (must match kTextureCount on the C++ side).
layout(set = 0, binding = 3) uniform sampler2D u_albedo[5];
layout(set = 0, binding = 4) uniform sampler2D u_normal[5];

// Triplanar projection sample. Avoids the "what UV does this cube face
// use" problem and works correctly on every brush size — the texture stays
// world-aligned regardless of brush dimensions. Cost is 3 samples per pass,
// which is fine for the coverage we have.
vec3 triplanar_sample(sampler2D tex, vec3 wp, vec3 N) {
    vec3 blend = pow(abs(N), vec3(4.0));
    blend /= max(blend.x + blend.y + blend.z, 1e-3);
    // Now that mipmaps exist (texture.cpp generates them via blit chain) and
    // vTexParams is `flat` so the sampler-array index is dynamically uniform,
    // standard `texture()` is correct — derivatives select the right mip and
    // far-distance moiré disappears.
    vec3 cx = texture(tex, wp.zy).rgb;
    vec3 cy = texture(tex, wp.xz).rgb;
    vec3 cz = texture(tex, wp.xy).rgb;
    return cx * blend.x + cy * blend.y + cz * blend.z;
}

// Triplanar tangent-space normal. For each projection axis, sample the
// tangent-space normal and transform into world space using a known basis
// for that axis. Then blend by face weight, same as the albedo. Cheap,
// works without explicit per-vertex tangents.
vec3 triplanar_normal(sampler2D ntex, vec3 wp, vec3 N) {
    vec3 blend = pow(abs(N), vec3(4.0));
    blend /= max(blend.x + blend.y + blend.z, 1e-3);

    vec3 nx = texture(ntex, wp.zy).rgb * 2.0 - 1.0;
    vec3 ny = texture(ntex, wp.xz).rgb * 2.0 - 1.0;
    vec3 nz = texture(ntex, wp.xy).rgb * 2.0 - 1.0;

    // For each axis, build a world-space normal: the tangent-space sample's
    // .xy goes into the perpendicular plane, .z reinforces the axis.
    vec3 wnx = vec3(N.x * sign(N.x) + nx.z * sign(N.x), nx.y, nx.x);
    vec3 wny = vec3(ny.x, N.y * sign(N.y) + ny.z * sign(N.y), ny.y);
    vec3 wnz = vec3(nz.x, nz.y, N.z * sign(N.z) + nz.z * sign(N.z));

    return normalize(wnx * blend.x + wny * blend.y + wnz * blend.z);
}

// Interleaved Gradient Noise (Jorge Jimenez, "Next Generation Post-Processing
// in Call of Duty: Advanced Warfare", SIGGRAPH 2014).
// Adjacent pixels see *smoothly varying* random values rather than hash-style
// uncorrelated jumps, so the per-pixel integration error has low spatial
// frequency — looks like a smooth gradient rather than a dotted dither.
float ign(vec2 p) {
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// Per-sample IGN: shift the input position by the sample index along a known
// magic vector so each sample's jitter is uncorrelated within a pixel but
// stays smooth across neighbouring pixels.
float rand(uvec3 seed) {
    float s = float(seed.x) + 5.588238  * float(seed.z);
    float t = float(seed.y) + 1.388765  * float(seed.z);
    return ign(vec2(s, t));
}

vec3 cos_hemi(float u1, float u2, vec3 n) {
    float r = sqrt(u1);
    float phi = 6.28318530718 * u2;
    vec3 d = vec3(r * cos(phi), sqrt(max(0.0, 1.0 - u1)), r * sin(phi));
    vec3 up_a = abs(n.y) < 0.999 ? vec3(0,1,0) : vec3(1,0,0);
    vec3 t = normalize(cross(up_a, n));
    vec3 b = cross(n, t);
    return d.x * t + d.y * n + d.z * b;
}

// Shadow + AO test. Cull-mask 0x01 picks up only instances marked as
// shadow-casters (bit 0). Sparks and projectiles are flagged 0xFE on the
// host — they're tiny visual effects whose hard shadow streaks looked wrong
// and whose AO darkening produced visible halos around hits.
bool any_hit(vec3 origin, vec3 dir, float t_max) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT,
                          0x01, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    return rayQueryGetIntersectionTypeEXT(rq, true) ==
           gl_RayQueryCommittedIntersectionTriangleEXT;
}

// Procedural sky: warm low horizon → cool zenith, brighter near sun.
vec3 sample_sky(vec3 dir) {
    vec3 L = normalize(scene.sun_direction.xyz);
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    vec3 sky = mix(horizon, zenith, pow(up, 0.45));
    float halo = pow(max(dot(dir, L), 0.0), 8.0);
    sky += scene.sun_color.rgb * scene.sun_color.a * 0.08 * halo;
    return sky;
}

// Closest-hit ray cast. Returns true on hit, fills out_t.
bool closest_hit(vec3 origin, vec3 dir, float t_max,
                 out float out_t) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                          0xFF, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t = rayQueryGetIntersectionTEXT(rq, true);
    return true;
}

// Closest-hit + material lookup via TLAS instance custom index.
bool closest_hit_material(vec3 origin, vec3 dir, float t_max,
                          out float out_t, out int out_idx) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                          0xFF, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t = rayQueryGetIntersectionTEXT(rq, true);
    out_idx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    return true;
}

void main() {
    // Screen-space motion vector — current_uv − prev_uv. prev_uv comes from
    // perspective-divided prev_clip (smooth-interpolated by the rasterizer).
    // Behind-camera prev pixels (prev_clip.w ≤ 0) have no valid prev_uv —
    // fall back to zero motion; the future SVGF pass treats sentinel as
    // "no history available" and rebuilds variance from current-frame
    // neighborhood instead of reprojecting.
    {
        vec2 current_uv = (gl_FragCoord.xy + 0.5) * scene.viewport.zw;
        if (vPrevClip.w > 0.0) {
            vec2 prev_ndc = vPrevClip.xy / vPrevClip.w;
            vec2 prev_uv = prev_ndc * vec2(0.5, 0.5) + vec2(0.5);
            outMotion = current_uv - prev_uv;
        } else {
            outMotion = vec2(0.0);
        }
    }

    vec3 N = normalize(vNormal);

    if (vEmissive.a > 0.5) {
        outColor = vec4(vColor + vEmissive.rgb, 1.0);
        return;
    }


    vec3 L = normalize(scene.sun_direction.xyz);

    // --- Albedo + bump mapping (triplanar projection) ---
    // Choose object-space (texture glued to the model) for dynamic objects,
    // world-space (texture aligned to the world grid) for static brushes.
    int   albedo_idx_raw = int(vTexParams.x);
    int   normal_idx_raw = int(vTexParams.y);
    float scale          = max(0.001, vTexParams.z);
    bool  obj_space      = vTexParams.w > 0.5;
    vec3  sample_pos     = (obj_space ? vObjectPos : vWorldPos) * scale;
    vec3  proj_n         = obj_space ? normalize(vObjectNormal) : N;
    bool  use_albedo     = albedo_idx_raw >= 0 && albedo_idx_raw < 5;
    bool  use_normal     = normal_idx_raw >= 0 && normal_idx_raw < 5 && !obj_space;
    // Texture sampling lives in branches gated on use_albedo / use_normal.
    // Sampler-array indexing on Vulkan requires a known-valid index; clamp
    // before indexing to keep the access well-defined regardless of any
    // hoisting the optimiser might do.
    vec3 albedo = vColor;
    if (use_albedo) {
        int  albedo_idx = clamp(albedo_idx_raw, 0, 4);
        vec3 tex_albedo = triplanar_sample(u_albedo[albedo_idx], sample_pos, proj_n);
        albedo = tex_albedo * vColor;
    }
    if (use_normal) {
        int  normal_idx = clamp(normal_idx_raw, 0, 4);
        N = triplanar_normal(u_normal[normal_idx], sample_pos, N);
    }

    float up = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    vec3 ambient = mix(scene.ambient.rgb, scene.sky_color.rgb * 0.45, up);
    ambient *= scene.rt_params.z;

    float n_dot_l = max(dot(N, L), 0.0);

    // Per-pixel seed *without* the frame counter — keeps the per-pixel sample
    // pattern stable across frames so the noise doesn't shimmer. Spatial
    // dither still exists but is now a fixed pattern, much less distracting
    // than per-frame churn.
    uvec3 seed_base = uvec3(uint(gl_FragCoord.x), uint(gl_FragCoord.y), 0u);

    // Distance-from-camera used for sample LOD across all RT effects.
    float cam_dist = distance(vWorldPos, scene.camera_pos.xyz);

    // --- Sun shadow (RT, contact-hardening / PCSS-style) ---
    // 1. BLOCKER SEARCH: a handful of cheap closest-hit rays in a wide cone
    //    around L; record the average distance to the occluder.
    // 2. PENUMBRA ESTIMATE: cone half-angle ∝ avg_blocker_distance × softness.
    //    Close-by occluders give a small cone (sharp shadow); far ones give
    //    a wide cone (soft shadow). Real-world effect: shadow under a pole
    //    is razor-sharp, the same shadow stretched across the floor is fuzzy.
    // 3. SHADOW RAYS: stratified sampling inside the size-adapted cone.
    float shadow = 0.0;
    if (n_dot_l > 0.0 && scene.rt_flags.x != 0) {
        int  N_s = lod_samples(max(1, scene.rt_flags.y), cam_dist);
        float base_softness = scene.rt_params.x;

        vec3 ref = abs(L.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 tan_u = normalize(cross(ref, L));
        vec3 tan_v = cross(L, tan_u);

        float bias = 0.005 + 0.02 * (1.0 - n_dot_l);
        vec3 origin = vWorldPos + N * bias;

        // 1. Blocker search — 4 rays in a wider cone (4× base softness) so we
        //    actually catch occluders even when the surface is close to lit.
        const int kBlockerSearch = 4;
        float sum_t = 0.0;
        int   hits = 0;
        for (int i = 0; i < kBlockerSearch; ++i) {
            float br1 = rand(seed_base + uvec3(i, 91u, 13u));
            float br2 = rand(seed_base + uvec3(i, 19u, 71u));
            float r   = sqrt(br1) * base_softness * 4.0;
            float phi = 6.28318530718 * br2;
            vec3 jitter = (cos(phi) * tan_u + sin(phi) * tan_v) * r;
            vec3 dir = normalize(L + jitter);
            float t;
            if (closest_hit(origin, dir, 200.0, t)) {
                sum_t += t;
                ++hits;
            }
        }

        if (hits == 0) {
            // No occluder anywhere in the wide cone → fully lit.
            shadow = 0.0;
        } else {
            // 2. Penumbra estimate. Distance is normalised against a 10m
            //    reference, then raised to a power between 1 (linear, the
            //    classic PCSS) and 3 (cubic — very contact-sharp close-up,
            //    very soft far away). The exponent comes from the
            //    shadow_curve slider in the menu.
            float avg_t = sum_t / float(hits);
            float t_norm = avg_t * 0.1;
            float curve_exp = mix(1.0, 3.0, clamp(scene.rt_params2.w, 0.0, 1.0));
            float scale = pow(max(t_norm, 0.0), curve_exp);
            float penumbra = clamp(base_softness * scale * 0.6,
                                   base_softness * 0.25,
                                   base_softness * 6.0);

            // 3. Stratified shadow rays in the size-adapted cone.
            int strata = int(ceil(sqrt(float(N_s))));
            float inv_strata = 1.0 / float(strata);
            float blocked = 0.0;
            int taken = 0;
            for (int sy = 0; sy < strata && taken < N_s; ++sy) {
                for (int sx = 0; sx < strata && taken < N_s; ++sx) {
                    float r1 = rand(seed_base + uvec3(taken, 11u, 47u));
                    float r2 = rand(seed_base + uvec3(taken, 53u, 23u));
                    float u1 = (float(sx) + r1) * inv_strata;
                    float u2 = (float(sy) + r2) * inv_strata;
                    float r   = sqrt(u1) * penumbra;
                    float phi = 6.28318530718 * u2;
                    vec3 jitter = (cos(phi) * tan_u + sin(phi) * tan_v) * r;
                    vec3 dir = normalize(L + jitter);
                    if (any_hit(origin, dir, 200.0)) blocked += 1.0;
                    ++taken;
                }
            }
            shadow = (blocked / float(taken)) * scene.rt_params.w;
        }
    }

    vec3 direct = albedo * scene.sun_color.rgb * scene.sun_color.a *
                  n_dot_l * (1.0 - shadow);

    // --- Muzzle flash: dynamic point light + RT shadow ---
    // Active for kMuzzleFlashDuration on each shot; CPU sets intensity > 0
    // and a position just in front of the eye. One shadow ray per pixel,
    // gated by N·L > 0 and within the falloff radius — keeps the cost
    // negligible when no shot is firing. Inverse-square attenuation with
    // a smooth cutoff at the radius so contributions don't pop.
    if (scene.muzzle_pos.w > 0.0) {
        vec3  m_pos       = scene.muzzle_pos.xyz;
        float m_intensity = scene.muzzle_pos.w;
        vec3  m_color     = scene.muzzle_color.rgb;
        float m_radius    = max(0.5, scene.muzzle_color.w);

        vec3  to_light    = m_pos - vWorldPos;
        float dist        = length(to_light);
        if (dist < m_radius) {
            vec3 L_m       = to_light / max(dist, 1e-3);
            float n_dot_lm = max(dot(N, L_m), 0.0);
            if (n_dot_lm > 0.0) {
                // Smooth falloff: 1/(1 + d²)·(1 - d/r)². The window term zeroes
                // contribution exactly at the radius, no hard edge.
                float window = 1.0 - dist / m_radius;
                float atten  = (window * window) / (1.0 + dist * dist);

                // RT shadow: one any-hit ray to the light. t_max stops short
                // of the light origin so a triangle exactly at m_pos can't
                // self-occlude.
                float bias    = 0.005 + 0.02 * (1.0 - n_dot_lm);
                vec3  origin  = vWorldPos + N * bias;
                bool  in_shadow = any_hit(origin, L_m, dist - 0.01);
                if (!in_shadow) {
                    direct += albedo * m_color * m_intensity * n_dot_lm * atten;
                }
            }
        }
    }

    // --- AO (stratified hemisphere) ---
    //
    // ao_mode (rt_flags2.w):
    //   0 = off — no AO, ao = 1.0 (lighting reads as fully lit at concavities)
    //   1 = "fast contact AO" — short hemisphere rays at HALF the radius and
    //       capped to 2 samples. Reads similar to GTAO at far less cost
    //       than full RTAO, but is still inline ray-traced (true GTAO needs
    //       a separate screen-space pass with depth+normal bound; deferred).
    //   2 = full RTAO — N hemisphere rays at user-set radius, the original
    //       quality path. Picks up off-screen occluders correctly.
    float ao = 1.0;
    int ao_mode = scene.rt_flags2.w;
    if (ao_mode > 0 && scene.rt_flags.z > 0) {
        // mode 1 caps to 2 samples; mode 2 honors the slider.
        int requested = (ao_mode == 1) ? min(scene.rt_flags.z, 2)
                                       : scene.rt_flags.z;
        int N_ao = lod_samples(requested, cam_dist);
        // mode 1 halves the search radius — bias toward contact AO rather
        // than ambient occlusion of the whole hemisphere.
        float ao_radius = scene.rt_params.y * (ao_mode == 1 ? 0.5 : 1.0);
        vec3 origin_ao = vWorldPos + N * 0.01;
        int strata = int(ceil(sqrt(float(N_ao))));
        float inv_strata = 1.0 / float(strata);
        int taken = 0;
        float occluded = 0.0;
        for (int sy = 0; sy < strata && taken < N_ao; ++sy) {
            for (int sx = 0; sx < strata && taken < N_ao; ++sx) {
                float r1 = rand(seed_base + uvec3(taken, 7u, 53u));
                float r2 = rand(seed_base + uvec3(taken, 41u, 5u));
                float u1 = (float(sx) + r1) * inv_strata;
                float u2 = (float(sy) + r2) * inv_strata;
                vec3 d = cos_hemi(u1, u2, N);
                if (any_hit(origin_ao, d, ao_radius)) occluded += 1.0;
                ++taken;
            }
        }
        ao = 1.0 - (occluded / float(taken));
    }

    // --- Path-traced GI (1..N bounces) ---
    // Each sample follows a path of up to N_bounces ray segments.
    //   - On a surface hit: accumulate emissive + albedo×scene_light (a cheap
    //     local-light approximation), tint the throughput by surface albedo,
    //     bounce in a cosine-weighted hemisphere oriented around the
    //     approximate hit normal (-ray_dir).
    //   - On a miss (sky): accumulate throughput × procedural sky and stop.
    // Samples are stratified on a sqrt(N)×sqrt(N) grid for clean low-N look.
    vec3 gi_indirect = vec3(0.0);
    int N_gi = scene.rt_flags2.x > 0 ? lod_samples(scene.rt_flags2.x, cam_dist) : 0;
    int N_bounces = max(1, scene.rt_flags2.z);
    if (N_gi > 0) {
        float gi_radius = scene.rt_params2.y;

        // Cheap "what light would land on a hit surface" estimate: a blended
        // view of the sun + sky.
        vec3 sun_term = scene.sun_color.rgb * scene.sun_color.a * 0.45;
        vec3 sky_term = scene.sky_color.rgb * 0.55;
        vec3 scene_light = sun_term + sky_term;

        int strata = int(ceil(sqrt(float(N_gi))));
        float inv_strata = 1.0 / float(strata);
        int taken = 0;
        vec3 sum = vec3(0.0);
        for (int sy = 0; sy < strata && taken < N_gi; ++sy) {
            for (int sx = 0; sx < strata && taken < N_gi; ++sx) {
                float r1 = rand(seed_base + uvec3(taken, 73u, 11u));
                float r2 = rand(seed_base + uvec3(taken, 91u, 47u));
                float u1 = (float(sx) + r1) * inv_strata;
                float u2 = (float(sy) + r2) * inv_strata;

                vec3 ray_dir = cos_hemi(u1, u2, N);
                vec3 ray_origin = vWorldPos + N * 0.01;
                vec3 throughput = vec3(1.0);
                vec3 path = vec3(0.0);

                for (int b = 0; b < N_bounces; ++b) {
                    float t;
                    int idx;
                    if (!closest_hit_material(ray_origin, ray_dir, gi_radius, t, idx)) {
                        path += throughput * sample_sky(ray_dir);
                        break;
                    }
                    Material m = materials[idx];
                    path += throughput * (m.emissive.rgb +
                                          m.color.rgb * scene_light);
                    if (b + 1 >= N_bounces) break;

                    // Tint the path's throughput by the surface albedo and
                    // continue from the hit point in a new cosine-hemisphere.
                    throughput *= m.color.rgb;
                    vec3 hit_pos = ray_origin + ray_dir * t;
                    vec3 hit_n = -ray_dir;  // approximate outward normal
                    ray_origin = hit_pos + hit_n * 0.01;
                    float br1 = rand(seed_base + uvec3(taken * 7 + b, 31u, 17u));
                    float br2 = rand(seed_base + uvec3(taken * 7 + b, 7u, 91u));
                    ray_dir = cos_hemi(br1, br2, hit_n);
                }
                sum += path;
                ++taken;
            }
        }
        gi_indirect = albedo * (sum / float(taken)) * scene.rt_params2.x;
    }

    // --- AO --- (also stratified to spread samples cleanly)
    // [moved below, recomputed from N_ao]

    // --- Specular reflection (only for surfaces flagged via emissive.b > 0.5
    //     in the model — repurposed cheaply: white pedestal & similar). To keep
    //     scope small we just enable it on surfaces with brightness > 0.85, a
    //     proxy for "shiny" — turn off via the toggle. ---
    vec3 reflection = vec3(0.0);
    bool shiny = scene.rt_flags2.y != 0 &&
                 (vColor.r > 0.80 && vColor.g > 0.80 && vColor.b > 0.80);
    if (shiny) {
        vec3 V = normalize(vWorldPos - vec3(0.0));  // approximate camera dir later
        // Use the surface tangent of vWorldPos as a proxy for view direction
        // by reflecting the WORLD-POS-from-origin vector across the normal.
        // It's a cheap pseudo-reflection that still moves with the surface
        // orientation; a true reflection needs the camera position pushed
        // through to the shader.
        vec3 R = reflect(normalize(vWorldPos), N);
        float t;
        if (closest_hit(vWorldPos + N * 0.01, R, 100.0, t)) {
            reflection = vec3(0.0);  // hit something — fallback to dim
        } else {
            reflection = sample_sky(R);
        }
        reflection *= scene.rt_params2.z;
    }

    vec3 indirect = albedo * ambient * ao + gi_indirect;
    vec3 final = direct + indirect + vEmissive.rgb + reflection * (shiny ? 1.0 : 0.0);

    outColor = vec4(final, 1.0);
}
