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
    // Terrain shader knobs:
    //   terrain_params.x = fog_strength (atmospheric perspective)
    //   terrain_params.y = wrap_strength (half-Lambert)
    //   terrain_params.z = detail_strength (texture brightness)
    //   terrain_params.w = shadow_softness_scale (PCSS cone × this)
    vec4  terrain_params;
    // Per-layer height-blend smoothstep edges:
    //   terrain_h_low.xy  = sand→grass start..end
    //   terrain_h_low.zw  = grass→dirt start..end
    //   terrain_h_high.xy = dirt→rock start..end
    //   terrain_h_high.zw = rock→snow start..end
    vec4  terrain_h_low;
    vec4  terrain_h_high;
    vec4  grass_extra;       // unused in cube.frag — laid out so grass_extra2.w is reachable
    vec4  grass_extra2;      // .w = terrain_debug_mode (0=off, 1=Lambert, 2=normal, 3=face)
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
// Pre-baked heightmap sun-shadow texture (R8, 0 = lit, 255 = shadowed).
// Sampled here as a distance fallback for terrain self-shadow — the
// BLAS holds the full heightmap detail, so RT shadow rays from a low-
// LOD rasterised surface false-hit BLAS peaks the raster missed. The
// bake was traced against the heightmap directly so it matches the
// rasterised LOD surface at distance with no false hits.
layout(set = 0, binding = 6) uniform sampler2D u_terrain_shadow;

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

// Sky tint without the sun halo — for atmospheric fog where adding
// the localized halo brightness on distant terrain pixels makes them
// glow blindingly when looking near the sun. Same horizon→zenith
// gradient as `sample_sky` but no halo term.
vec3 sample_sky_atmosphere(vec3 dir) {
    float up = clamp(dir.y, 0.0, 1.0);
    vec3 horizon = scene.sky_color.rgb * 0.55 + scene.sun_color.rgb * 0.10;
    vec3 zenith  = scene.sky_color.rgb;
    return mix(horizon, zenith, pow(up, 0.45));
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

// Closest-hit + material lookup. Returns instance custom index AND
// primitive index. The merged-static BLAS instance carries
// kStaticBlasInstSentinel as its custom index; the shader recovers the
// per-brush material via primitive_id / 12 in that case (12 tris/brush).
// Dynamic instances carry their materials-buffer slot directly.
bool closest_hit_material(vec3 origin, vec3 dir, float t_max,
                          out float out_t, out int out_inst, out int out_prim) {
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT,
                          0xFF, origin, 0.001, dir, t_max);
    while (rayQueryProceedEXT(rq)) {}
    if (rayQueryGetIntersectionTypeEXT(rq, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT) {
        return false;
    }
    out_t = rayQueryGetIntersectionTEXT(rq, true);
    out_inst = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
    out_prim = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
    return true;
}

// Sentinel for the merged-static BLAS instance — must match
// `kStaticBlasInstSentinel` in src/engine/vk_engine/rt.cpp. Picked at the
// top of the 24-bit instanceCustomIndex range so it never collides with a
// real materials-buffer slot.
const int kStaticBlasSentinel = 0xFFFFFF;
const int kCubeTrisPerBox     = 12;

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

    bool is_terrain_pre = vTexParams.w > 1.5;

    // ---- Terrain debug visualisation (set in pause-menu Graphics) ----
    // Bypasses everything except a basic Lambert / normal preview so we
    // can isolate which downstream effect (RT shadow, AO, GI, slope
    // blend, cavity, fog, triplanar) is producing visible artifacts.
    if (is_terrain_pre) {
        int dbg = int(scene.grass_extra2.w + 0.5);
        if (dbg == 1) {
            // Pure Lambert with constant grey albedo. No RT, no AO,
            // no slope blend, no fog. If the artifact persists here
            // it's a geometry/normal problem; if it disappears the
            // bug is in one of the disabled effects.
            vec3 L = normalize(scene.sun_direction.xyz);
            float ndl = max(dot(N, L), 0.0);
            vec3 lit = vec3(0.55) * (0.25 + 0.75 * ndl);
            outColor = vec4(lit, 1.0);
            return;
        } else if (dbg == 2) {
            // Per-vertex normal as RGB. Adjacent triangles with
            // mismatched corner normals will show as visible colour
            // jumps — direct view into the LOD/Gouraud issue.
            outColor = vec4(N * 0.5 + 0.5, 1.0);
            return;
        } else if (dbg == 3) {
            // Geometric face normal (cross of world-pos derivatives,
            // sign-guarded for Vulkan y-down). Each large LOD triangle
            // is a uniform colour — confirms what flat-shaded geometry
            // looks like at distance.
            vec3 fn = cross(dFdx(vWorldPos), dFdy(vWorldPos));
            if (fn.y < 0.0) fn = -fn;
            outColor = vec4(normalize(fn) * 0.5 + 0.5, 1.0);
            return;
        } else if (dbg == 4) {
            // Lambert + sun shadow. Hybrid RT/bake: RT only for the
            // first 40m where LOD 0 raster matches the BLAS exactly,
            // then full heightmap bake. The bake was traced from the
            // same heightmap so the shadow values match the rasterised
            // surface at any LOD — no false hits.
            vec3 L = normalize(scene.sun_direction.xyz);
            float ndl = max(dot(N, L), 0.0);
            float dist_to_cam = distance(vWorldPos, scene.camera_pos.xyz);

            // Bake — sample first so we always have it.
            ivec2 sz = textureSize(u_terrain_shadow, 0);
            float side = float(sz.x - 1) * 1.0;     // dim cells × cell_size
            vec2 uv = (vWorldPos.xz / side) + vec2(0.5);
            float sh_bake = 0.0;
            if (all(greaterThanEqual(uv, vec2(0.0))) &&
                all(lessThanEqual(uv, vec2(1.0)))) {
                sh_bake = step(0.5, texture(u_terrain_shadow, uv).r);
            }
            float sh = sh_bake;
            // Mix in RT only at very close range to capture small
            // castle/dyn-prop shadows the bake doesn't cover.
            if (dist_to_cam < 80.0) {
                vec3 origin = vWorldPos + N * 0.04;
                float sh_rt = any_hit(origin, L, 200.0) ? 1.0 : 0.0;
                float near_t = 1.0 - smoothstep(40.0, 80.0, dist_to_cam);
                sh = mix(sh, sh_rt, near_t);
            }
            vec3 lit = vec3(0.55) * (0.25 + 0.75 * ndl * (1.0 - sh));
            outColor = vec4(lit, 1.0);
            return;
        }
    }

    vec3 L = normalize(scene.sun_direction.xyz);

    // --- Albedo + bump mapping (triplanar projection) ---
    // Choose object-space (texture glued to the model) for dynamic objects,
    // world-space (texture aligned to the world grid) for static brushes.
    int   albedo_idx_raw = int(vTexParams.x);
    int   normal_idx_raw = int(vTexParams.y);
    float scale          = max(0.001, vTexParams.z);
    // tex_params.w convention:
    //   0 = world-space triplanar (default — castle brushes)
    //   1 = object-space triplanar (dynamic crates so the texture
    //       rotates with the body)
    //   2 = terrain shading: sand→grass→dirt→rock→snow blended by
    //       world height + slope, with optional triplanar Ground054
    //       detail multiplied on top.
    bool  is_terrain     = vTexParams.w > 1.5;
    bool  obj_space      = !is_terrain && vTexParams.w > 0.5;
    // Terrain uses world-space triplanar like the static brushes — the
    // mesh model is identity so vObjectPos == vWorldPos already, but we
    // pick world-space explicitly so the height read uses true world Y.
    vec3  sample_pos     = (obj_space ? vObjectPos : vWorldPos) * scale;
    vec3  proj_n         = obj_space ? normalize(vObjectNormal) : N;
    bool  use_albedo     = albedo_idx_raw >= 0 && albedo_idx_raw < 5;
    bool  use_normal     = normal_idx_raw >= 0 && normal_idx_raw < 5 &&
                           !obj_space && !is_terrain;
    // Texture sampling lives in branches gated on use_albedo / use_normal.
    // Sampler-array indexing on Vulkan requires a known-valid index; clamp
    // before indexing to keep the access well-defined regardless of any
    // hoisting the optimiser might do.
    vec3 albedo = vColor;
    if (is_terrain) {
        // ---- Terrain layer blend (height + slope) ----
        // Layer palette tuned for "natural earth" — desaturated, not too
        // toy-coloured. Heights match the heightmap params:
        //   plateau ~22, mountain crests ~120-140 (height_scale=140 in
        //   the generator). Sea level isn't real here, but we still
        //   give the bottom band a sandy hue.
        const vec3 sand   = vec3(0.78, 0.71, 0.50);
        const vec3 grass  = vec3(0.31, 0.45, 0.18);
        const vec3 dirt   = vec3(0.42, 0.30, 0.18);
        const vec3 rock   = vec3(0.40, 0.38, 0.36);
        const vec3 snow   = vec3(0.95, 0.95, 0.97);

        float h = vWorldPos.y;
        // Slope: 0 = flat (normal up), 1 = vertical wall.
        float slope = 1.0 - clamp(N.y, 0.0, 1.0);

        // ---- Layer-transition break-up ----
        // Without a per-pixel offset, layer boundaries form perfectly
        // horizontal contour lines that read as obvious "stripes" on
        // gentle slopes. We jitter each transition's effective height
        // by a low-frequency hash on world XZ — same as how natural
        // soils don't change at exactly the same elevation everywhere.
        // The jitter is in metres and matched to ~half the smoothstep
        // width so it breaks up the line without erasing the gradient.
        float n0 = ign(vWorldPos.xz * 0.04) - 0.5;   // ±0.5 noise
        float n1 = ign(vWorldPos.xz * 0.13 + vec2(13.7, 41.3)) - 0.5;
        float jitter = (n0 + 0.5 * n1);              // ~±0.75
        float jh = h + jitter * 4.0;                 // ±3m offset
        float t_sand = smoothstep(scene.terrain_h_low.x,  scene.terrain_h_low.y,  jh);
        float t_dirt = smoothstep(scene.terrain_h_low.z,  scene.terrain_h_low.w,  jh);
        float t_rock = smoothstep(scene.terrain_h_high.x, scene.terrain_h_high.y, jh);
        float t_snow = smoothstep(scene.terrain_h_high.z, scene.terrain_h_high.w, jh);

        vec3 base = mix(sand, grass, t_sand);
        base = mix(base, dirt, t_dirt);
        base = mix(base, rock, t_rock);
        base = mix(base, snow, t_snow);

        // Steep faces become rocky regardless of altitude. Slope jitter
        // breaks up the cliff/grass border the same way the height
        // jitter breaks up horizontal layer transitions.
        // Fade slope-driven rock with camera distance: at LOD 2/3 the
        // Gouraud-interpolated normals between widely-spaced verts
        // produce per-triangle slope swings, and adjacent large
        // triangles end up classified as rock vs grass at random,
        // reading as patchy "different faces". Past ~120 m we let the
        // height-band layering carry the look — visually identical at
        // distance, no per-triangle classification noise.
        float slope_jitter = (ign(vWorldPos.xz * 0.09 + vec2(7.0, 19.0)) - 0.5) * 0.10;
        float steep_raw = smoothstep(0.45 + slope_jitter, 0.75 + slope_jitter, slope);
        float steep_dist = distance(vWorldPos, scene.camera_pos.xyz);
        float steep_w = 1.0 - smoothstep(80.0, 200.0, steep_dist);
        float steep = steep_raw * steep_w;
        base = mix(base, rock, steep);

        // ---- Cavity AO from local height curvature ----
        // dFdx(N)·dFdx(vWorldPos) gives concavity per fragment. Adds
        // "natural shadowing in cracks" for free near the camera, but
        // on distant LOD-2/3 chunks the screen-space derivatives of
        // Gouraud-interpolated normals between widely-spaced verts
        // become erratic (large per-pixel jumps), and the cavity term
        // oscillates 0.45↔1.0 across adjacent triangles — visible as
        // patchy dark faces on far ridges. Fade the effect out past
        // ~120 m so distant terrain reads as smooth.
        float curvature = -dot(dFdx(N), dFdx(vWorldPos)) -
                          dot(dFdy(N), dFdy(vWorldPos));
        float cav_dist  = distance(vWorldPos, scene.camera_pos.xyz);
        float cav_w     = 1.0 - smoothstep(80.0, 200.0, cav_dist);
        float cavity    = clamp(0.5 - curvature * 0.4, 0.45, 1.0);
        base *= mix(1.0, cavity, cav_w);

        // Optional triplanar detail using the engine's Ground054 albedo
        // (texture index 0). When textures are off (use_albedo=false)
        // we just show the layer-blended colour.
        if (use_albedo) {
            int  albedo_idx = clamp(albedo_idx_raw, 0, 4);
            vec3 detail = triplanar_sample(u_albedo[albedo_idx],
                                           sample_pos, proj_n);
            // Multiply detail × layer base × user-tunable detail
            // brightness so the user can dial detail in/out.
            albedo = base * detail * scene.terrain_params.z;
        } else {
            albedo = base;
        }
    } else {
        if (use_albedo) {
            int  albedo_idx = clamp(albedo_idx_raw, 0, 4);
            vec3 tex_albedo = triplanar_sample(u_albedo[albedo_idx],
                                               sample_pos, proj_n);
            albedo = tex_albedo * vColor;
        }
        if (use_normal) {
            int  normal_idx = clamp(normal_idx_raw, 0, 4);
            N = triplanar_normal(u_normal[normal_idx], sample_pos, N);
        }
    }

    float up = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
    // Split ambient into a sky-derived component (fades when the surface
    // is enclosed — sky_factor below) and a constant ground/reflected
    // component that exists indoors too (a real room has a floor that
    // reflects light, not a black void). Combining them later avoids
    // the "AO * sky_factor" double-darkening that crushed interior
    // edges to black.
    vec3 ambient_ground = scene.ambient.rgb * scene.rt_params.z;
    vec3 ambient_sky    = scene.sky_color.rgb * 0.45 * scene.rt_params.z;

    // Direct lighting uses pure Lambert (max(0, N·L)). The "wrap" lift
    // for back-of-mountain pixels is applied LATER as a sky-tinted
    // ambient bounce — applying it to direct here would produce the
    // "back-face fully lit while front-face shadow-rayed" artifact:
    // back faces have n_dot_l_raw = 0 so no shadow ray fires, but the
    // wrap pushes their effective n_dot_l above zero and the missing
    // shadow term means full sun colour leaks through. Front-facing
    // pixels resolved correctly via the shadow ray. Adjacent pixels
    // looking radically different is what the user reported.
    float n_dot_l_raw = max(dot(N, L), 0.0);
    float n_dot_l = n_dot_l_raw;

    // Per-pixel seed WITH the frame counter — TAA accumulates over ~8
    // frames, so animating the noise lets temporal averaging resolve the
    // residual dither into a clean signal. We tried a frame-stable seed
    // earlier (less moment-to-moment shimmer) but it left a fixed dither
    // pattern visible on flat surfaces; with TAA on (default) the temporal
    // average wins. We mod the frame counter by a small power-of-two so
    // the cycle is short and TAA always reaches steady state quickly.
    uvec3 seed_base = uvec3(uint(gl_FragCoord.x),
                            uint(gl_FragCoord.y),
                            uint(scene.rt_flags.w) & 7u);

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
    if (n_dot_l_raw > 0.0 && scene.rt_flags.x != 0) {
        int  N_s = lod_samples(max(1, scene.rt_flags.y), cam_dist);
        // Per-pixel softness — terrain optionally tightens the cone via
        // the Settings UI to reduce PCSS dither. Tightening trades soft
        // shadow edges for cleaner per-pixel results at the same sample
        // count. Brushes/dyn props always use the global slider value.
        float base_softness = scene.rt_params.x;
        if (is_terrain_pre) {
            base_softness *= max(0.05, scene.terrain_params.w);
        }

        vec3 ref = abs(L.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        vec3 tan_u = normalize(cross(ref, L));
        vec3 tan_v = cross(L, tan_u);

        // Terrain shadow-ray bias has a non-linear ramp with camera
        // distance because the chunked raster mesh uses distance-LOD
        // (stride 1/2/4/8) while the BLAS holds the full heightmap.
        // At LOD 3 (≥320 m) the rasterised straight-line interpolation
        // can sit 5–15 m BELOW true heightmap peaks; shadow rays fired
        // upward then false-hit those missed peaks → dark patches on
        // distant ridges. We hold near bias tight (no peter-panning on
        // close shadows) and ramp hard past LOD 0's range:
        //   ≤ 80 m  : 0.04 m (matches the LOD-0 / BLAS gap)
        //   320 m   : ~6 m  (covers typical LOD-3 peak undershoot)
        //   ≥ 600 m : ~12 m (covers worst-case)
        float dist_to_cam = distance(vWorldPos, scene.camera_pos.xyz);
        float far_t = clamp((dist_to_cam - 80.0) / 320.0, 0.0, 1.0);
        float terrain_far_bias = far_t * far_t * 8.0;   // 0 → 8 across the ramp
        float bias = is_terrain_pre
            ? (max(0.04, terrain_far_bias) + 0.10 * (1.0 - n_dot_l_raw))
            : (0.005 + 0.02 * (1.0 - n_dot_l_raw));
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

        // Terrain hybrid: blend the RT PCSS result toward the heightmap
        // shadow bake at distance. The BLAS holds the full heightmap
        // detail while the rasterised LOD-2/3 surface under-shoots
        // peaks → upward shadow rays false-hit the BLAS, producing
        // patchy dark "faces" on far ridges. The bake was traced
        // directly against the heightmap so its values match the
        // rasterised surface at any LOD; using it past 80 m
        // eliminates the false hits while RT keeps crisp near-shadows
        // (incl. boxes / castle on terrain) within ~40 m of the camera.
        if (is_terrain_pre) {
            float blend_far = smoothstep(40.0, 80.0, cam_dist);
            if (blend_far > 0.0) {
                ivec2 sz_b = textureSize(u_terrain_shadow, 0);
                float side_b = float(sz_b.x - 1) * 1.0;
                vec2 uv_b = (vWorldPos.xz / side_b) + vec2(0.5);
                float sh_bake = 0.0;
                if (all(greaterThanEqual(uv_b, vec2(0.0))) &&
                    all(lessThanEqual(uv_b, vec2(1.0)))) {
                    sh_bake = step(0.5, texture(u_terrain_shadow, uv_b).r);
                }
                shadow = mix(shadow, sh_bake * scene.rt_params.w, blend_far);
            }
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
        // Unstratified random samples driven by IGN + temporal frame seed.
        // The stratified version (ceil(sqrt(N))²-cell grid) was biased
        // when N wasn't a perfect square — at N=2 strata=2 picked only
        // the cells with sy=0 → all rays in one half of the hemisphere
        // azimuth, which read as systematically darker edges. Random +
        // TAA temporal accumulation converges to true AO over ~8 frames
        // regardless of N.
        int taken = 0;
        float occluded = 0.0;
        for (int i = 0; i < N_ao; ++i) {
            float u1 = rand(seed_base + uvec3(i, 7u, 53u));
            float u2 = rand(seed_base + uvec3(i, 41u, 5u));
            vec3 d = cos_hemi(u1, u2, N);
            if (any_hit(origin_ao, d, ao_radius)) occluded += 1.0;
            ++taken;
        }
        // Two-step shaping to control corner overlap-darkness:
        //   1. sqrt curve on raw AO — compresses the [0..1] range so the
        //      delta between 30% occluded (1 edge) and 60% (2 edges
        //      overlapping) reads as a small darkening step instead of
        //      "twice as dark". Linear AO stacked occlusions
        //      multiplicatively, perceptually wrong at inner corners.
        //   2. ao_floor remap so even fully-occluded never crushes to
        //      pure black.
        float raw = 1.0 - (occluded / float(taken));
        raw = sqrt(raw);
        float ao_floor_v = scene.rt_lod.w;
        ao = mix(ao_floor_v, 1.0, raw);
        // Terrain at distance: the rasterised LOD-2/3 surface sits BELOW
        // the BLAS detail. AO rays from vWorldPos upward hit BLAS peaks
        // the raster doesn't show → adjacent large triangles get
        // different occluded-ray fractions → patchy dark "faces". Fade
        // AO to 1.0 (no occlusion) past 80 m for terrain so the
        // BLAS-vs-raster mismatch can't drive per-triangle darkening.
        if (is_terrain_pre) {
            float ao_far_t = smoothstep(80.0, 200.0,
                                         distance(vWorldPos, scene.camera_pos.xyz));
            ao = mix(ao, 1.0, ao_far_t);
        }
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
    // Sky visibility — fraction of first-bounce GI rays that escape the
    // scene. Used after the GI loop to attenuate the (sky-derived) ambient
    // term: enclosed surfaces (e.g. inside the keep) get sky_vis ≈ 0 and
    // their ambient drops to near zero, giving the room a real "interior"
    // feel instead of the same flat brightness as outdoors. Default 1.0
    // when GI is disabled so existing behaviour is unchanged.
    float sky_vis = 1.0;
    int sky_misses = 0;
    int sky_total  = 0;
    int N_gi = scene.rt_flags2.x > 0 ? lod_samples(scene.rt_flags2.x, cam_dist) : 0;
    int N_bounces = max(1, scene.rt_flags2.z);
    if (N_gi > 0) {
        float gi_radius = scene.rt_params2.y;

        // Sky-only ambient term — used when a GI bounce hits a surface
        // that CANNOT see the sun. Approximates the soft fill light from
        // the sky dome that reaches indoor surfaces near openings. 0.05
        // (~5% of sky color) keeps deep interiors believably dark; the
        // earlier 0.18 made every shadowed bounce hit accumulate more
        // light than a real enclosed room would have.
        vec3 sky_fill = scene.sky_color.rgb * 0.05;

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
                    int prim;
                    if (!closest_hit_material(ray_origin, ray_dir, gi_radius, t, idx, prim)) {
                        path += throughput * sample_sky(ray_dir);
                        // First-bounce miss = the surface can see sky in
                        // this direction. Tracks "open vs enclosed" for the
                        // ambient attenuation below.
                        if (b == 0) ++sky_misses;
                        break;
                    }
                    // Static-castle hit: 1 BLAS instance covers all brushes
                    // — recover the per-brush material slot via primitive id.
                    int mat_idx = (idx == kStaticBlasSentinel)
                                    ? (prim / kCubeTrisPerBox)
                                    : idx;
                    Material m = materials[mat_idx];
                    vec3 hit_pos = ray_origin + ray_dir * t;
                    vec3 hit_n = -ray_dir;  // approximate outward normal

                    // *** GI bounce lighting *** — fires a shadow ray
                    // from the hit point to the sun for bounce levels up
                    // to gi_shadow_max_bounce (rt_lod.z). Bounces beyond
                    // that fall back to sky_fill alone. 0 = no GI sun
                    // shadows (cheap, every shadowed bounce uses fill);
                    // 1 = first bounce only (default, the cheap one
                    // that gives the indoor-vs-outdoor delta we want);
                    // higher = more accurate at proportional cost.
                    int gi_shadow_max = int(scene.rt_lod.z);
                    vec3 hit_light = sky_fill;
                    if (b < gi_shadow_max) {
                        float n_dot_sun = dot(hit_n, scene.sun_direction.xyz);
                        if (n_dot_sun > 0.0) {
                            if (!any_hit(hit_pos + hit_n * 0.01,
                                         scene.sun_direction.xyz, 200.0)) {
                                hit_light += scene.sun_color.rgb *
                                             scene.sun_color.a * n_dot_sun;
                            }
                        }
                    }
                    path += throughput * (m.emissive.rgb +
                                          m.color.rgb * hit_light);
                    if (b + 1 >= N_bounces) break;

                    // Tint the path's throughput by the surface albedo and
                    // continue from the hit point in a new cosine-hemisphere.
                    throughput *= m.color.rgb;
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
        sky_total = taken;
        sky_vis = float(sky_misses) / float(max(1, sky_total));
    }

    // --- AO --- (also stratified to spread samples cleanly)
    // [moved below, recomputed from N_ao]

    // Sky-occlusion term. Surfaces that can't see the sky get a much
    // darker ambient — that's the natural look of an enclosed room. The
    // 0.15 floor stops pure-interior pixels from going completely black
    // (real rooms have some bounced light), and the smoothstep curve
    // rolls in the sky contribution faster than linear so the open arena
    // still reads bright. GI-disabled fallback (sky_vis=1.0) leaves the
    // outdoor look identical to before this change.
    // Combine: sky_factor only attenuates the sky-derived part of
    // ambient, the ground term carries through everywhere. AO multiplies
    // the whole thing so edges still darken, but never to zero (the
    // ground term keeps a baseline). 0.10 floor on sky_factor is a tiny
    // gesture toward "windowless rooms still have some light leakage."
    float sky_factor = mix(0.10, 1.0, smoothstep(0.0, 0.6, sky_vis));
    vec3 ambient_combined = mix(ambient_ground,
                                 ambient_sky * sky_factor, up);
    vec3 indirect = albedo * ambient_combined * ao + gi_indirect;
    // Terrain sky-bounce wrap. Back-facing pixels (-N·L > 0) pick up a
    // sky-tinted lift to model atmospheric scattering bouncing onto the
    // shadow side of mountains. Applied as ambient here (not as direct)
    // so it never overrides a real shadow-ray result. wrap_strength
    // tunes the intensity from the Settings UI.
    if (is_terrain_pre) {
        float wrap_amt = clamp(scene.terrain_params.y, 0.0, 1.0);
        float back = max(0.0, -dot(N, L));
        indirect += albedo * scene.sky_color.rgb * back * 0.30 * wrap_amt;
    }
    vec3 final = direct + indirect + vEmissive.rgb;

    // Atmospheric perspective for terrain. Distant ground fades toward
    // the sky tint along the view direction — the standard "real
    // mountains" look. Onset starts at 200m and reaches 75% blend at
    // 1500m; never fully hides the silhouette so the scene retains
    // depth. Cheaper than volumetric fog and visually indistinguishable
    // for clear-day weather.
    if (is_terrain_pre) {
        vec3 view_dir = normalize(vWorldPos - scene.camera_pos.xyz);
        // sample_sky_atmosphere: no sun halo. The halo is fine for the
        // actual sky pixels (compose pass / GI rays), but applying it
        // to FOG made distant terrain near the sun direction look like
        // glowing white blobs — sun's halo × intensity 2 was multiplied
        // INTO the fog tint and bled bright-white over the mountains.
        vec3 fog_color = sample_sky_atmosphere(view_dir);
        // 95% sky by ~1.3km — both edges (corner ray reaches ~2km of
        // world at 80° FOV / far=1500m) AND centre (clipped at 1500m
        // hard) are deep into fog before the cut, so the asymmetric
        // visible disc is invisible.
        float fog_t = clamp((cam_dist - 250.0) / 1100.0, 0.0, 0.95);
        fog_t = fog_t * fog_t * (3.0 - 2.0 * fog_t);
        fog_t *= clamp(scene.terrain_params.x, 0.0, 1.0);
        final = mix(final, fog_color, fog_t);
    }

    outColor = vec4(final, 1.0);
}
