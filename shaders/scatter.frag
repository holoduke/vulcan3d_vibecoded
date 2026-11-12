#version 460
#extension GL_GOOGLE_include_directive : require

// Ground clutter fragment: rock-textured Lambert with the scene's sun +
// shadow + sky ambient. Ships pre-shaded colour through the gbuffer as
// material 5 (pass-through), same contract as grass.frag, so the
// deferred pass keeps the forward look exactly.

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec3 vWorldPos;
layout(location = 3) in float vSunShadow;
layout(location = 4) in float vCullKill;
layout(location = 5) in float vDistToCam;
layout(location = 6) in vec4 vPrevClip;
layout(location = 7) in vec2 vUv;
layout(location = 8) in vec3 vObj;
layout(location = 9) flat in vec4 vInst;
layout(location = 10) flat in vec2 vRotSeed;

layout(push_constant) uniform PC {
    mat4 mvp;
    mat4 model;
    mat4 prev_mvp;
    vec4 color;
    vec4 emissive;
    vec4 tex_params;
    vec4 grass_params;   // .y = 1 for plants
} pc;

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outMotion;
layout(location = 2) out vec4 outGBuffer0;
layout(location = 3) out vec4 outGBuffer1;

const int kTextureCount = 19;
layout(set = 0, binding = 3) uniform sampler2D u_albedo[kTextureCount];

#include "scene_ubo.glsl"

#include "rock_sdf.glsl"

vec2 _octa_encode(vec3 n) {
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    vec2 e = (n.z >= 0.0)
        ? n.xy
        : ((1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0,
                                     n.y >= 0.0 ? 1.0 : -1.0));
    return e * 0.5 + 0.5;
}

void main() {
    if (vCullKill > 0.5) discard;

    vec3 N = normalize(vNormal);
    bool plant = pc.grass_params.y > 0.5;
    vec3 wpos = vWorldPos;
    gl_FragDepth = gl_FragCoord.z;
    float stone_cav = 0.0, stone_ground = 0.0;
    if (!plant) {
        // Ray-marched stone: every instance its own shape (seed from its
        // position), half sunk in the ground.
        float sc = max(vInst.w, 1e-3);
        float rc = cos(vRotSeed.x), rsn = sin(vRotSeed.x);
        vec3 cw = scene.camera_pos.xyz - vInst.xyz;
        vec3 co = vec3(rc * cw.x + rsn * cw.z, cw.y, -rsn * cw.x + rc * cw.z) / sc;
        const vec3 kOff = kRockC - vec3(0.0, 0.30, 0.0);        // stone centre 0.3 above its base
        vec3 ro = vObj + kOff;
        vec3 rdo = normalize(vObj - co);
        float seed = vRotSeed.y;
        // Per-stone variety: lumpiness, layered ones, and flat slab pieces.
        float sv = fract(seed * 0.618);
        float s_amp = 0.15 + 0.22 * fract(seed * 0.37);
        float s_strata = sv < 0.22 ? -1.0 : (sv > 0.72 ? 0.5 : 0.0);
        float fp = distance(vWorldPos, scene.camera_pos.xyz) * 2.0 * scene.viewport.w / sc;
        float fine = 1.0 - smoothstep(0.006, 0.03, fp);
        float eps = max(fp * 0.6, 0.0015);
        float th = rock_march(ro, rdo, 3.4, seed, s_amp, s_strata, fine, eps);
        if (th < 0.0) discard;
        vec3 po = ro + rdo * th;
        rock_sdf(po, seed, s_amp, s_strata, fine, stone_cav);
        vec3 no = rock_normal(po, seed, s_amp, s_strata, fine, max(eps * 1.2, 0.002));
        vec3 pl = (po - kOff) * sc;
        wpos = vec3(rc * pl.x - rsn * pl.z, pl.y, rsn * pl.x + rc * pl.z) + vInst.xyz;
        N = normalize(vec3(rc * no.x - rsn * no.z, no.y, rsn * no.x + rc * no.z));
        vec4 clipr = pc.mvp * vec4(wpos, 1.0);
        gl_FragDepth = clipr.z / max(clipr.w, 1e-4);
        // Foot of the stone: soil-stained, normal eased toward the ground.
        stone_ground = 1.0 - smoothstep(0.02, 0.30, (wpos.y - vInst.y) / sc);
        N = normalize(mix(N, vec3(0.0, 1.0, 0.0), stone_ground * 0.5));
    }

    if (plant) {
        // Two-sided cards / fronds: face the normal toward the viewer.
        vec3 V = normalize(scene.camera_pos.xyz - vWorldPos);
        if (dot(N, V) < 0.0) N = -N;
    }

    // Rock albedo: 2-axis "biplanar" sample of the terrain rock layer
    // (slot 7) — full triplanar is overkill for fist-sized clutter.
    vec3 bw = abs(N);
    vec2 uvA = wpos.zy * 2.0;
    vec2 uvB = wpos.xz * 2.0;
    float wA = bw.x + bw.z * 0.5;
    float wB = bw.y;
    vec3 tex = (texture(u_albedo[7], uvA).rgb * wA +
                texture(u_albedo[7], uvB).rgb * wB) / max(wA + wB, 1e-3);
    vec3 albedo = tex * vColor;
    if (!plant) {
        albedo *= 1.0 - 0.5 * clamp(stone_cav, 0.0, 1.0);
        albedo = mix(albedo, albedo * vec3(0.52, 0.47, 0.38), stone_ground * 0.85);
    }
    if (plant) {
        // Part colours: stems / leaves green (ferns carry their own
        // tint), petals = instance colour, flower centre golden.
        int part = int(vUv.x + 0.5);
        bool fern = vColor.g > vColor.r * 1.5 && vColor.g > vColor.b * 1.5 && vColor.g < 0.35;
        vec3 leaf = fern ? vColor : vec3(0.11, 0.22, 0.05);
        leaf *= 0.8 + 0.4 * vUv.y;                // tips catch more light
        albedo = (part == 0) ? leaf : (part == 1 ? vColor : vec3(0.75, 0.55, 0.04));
        if (part == 4) {
            // Petal: the photographed petal structure (atlas right half), tinted.
            float pu = clamp((vUv.x - 4.0) / 0.4, -1.0, 1.0);
            float pt = clamp(fract(vUv.y * 32.0) / 0.98, 0.0, 1.0);
            vec3 pet = texture(u_albedo[18], vec2(0.5 + (pu * 0.5 + 0.5) * 0.5, 1.0 - pt)).rgb;
            vec3 pm  = textureLod(u_albedo[18], vec2(0.75, 0.5), 8.0).rgb;
            albedo = vColor * clamp(pet / max(pm, vec3(0.05)), vec3(0.5), vec3(1.4));
            albedo *= mix(0.80, 1.0, smoothstep(0.0, 0.35, pt));      // deeper toward the centre
        }
        if (part == 3) {
            // Leaf blade with real leaf-space coordinates: u across (-1..1),
            // t along (0 base .. 1 tip). Procedural venation - a pale midrib,
            // side veins sweeping forward from it, fine cross-veining - plus
            // the colour structure of a real leaf (paler along the veins and
            // toward the base, darker blade, slightly yellowed margin) and a
            // matching relief in the shading normal.
            float u = clamp((vUv.x - 3.0) / 0.4, -1.0, 1.0);
            float t = clamp(fract(vUv.y * 32.0) / 0.98, 0.0, 1.0);
            float au = abs(u);
            float fw = max(fwidth(u), 1e-3);
            float mid = 1.0 - smoothstep(0.035, 0.035 + fw * 1.5 + 0.03, au);
            float vph = t * 7.0 - au * 1.6;                       // veins angle toward the tip
            float vd  = abs(fract(vph) - 0.5) * 1.8;
            float vfw = max(fwidth(vph), 1e-3);
            float side = (1.0 - smoothstep(0.03, 0.03 + vfw * 1.5 + 0.04, vd)) *
                         smoothstep(0.02, 0.15, au) * (1.0 - smoothstep(0.75, 1.0, au)) *
                         (1.0 - smoothstep(0.25, 0.9, vfw));     // fade out when sub-pixel
            float fine = sin(vph * 6.2832 * 4.0 + au * 23.0) * 0.5 + 0.5;
            fine *= 1.0 - smoothstep(0.08, 0.5, vfw * 4.0);
            float vein = max(mid * 0.8, side * 0.38);
            vec3 blade = vec3(0.085, 0.20, 0.045) * (0.86 + 0.20 * fine);
            blade = mix(blade, blade * vec3(1.25, 1.12, 0.75), smoothstep(0.7, 1.0, au));   // margin
            blade *= mix(1.15, 0.92, t);                                                   // paler base
            albedo = mix(blade, vec3(0.30, 0.42, 0.14), vein);
            // Photographed leaf surface (atlas left half) as the fine structure.
            {
                vec3 lt = texture(u_albedo[18], vec2((u * 0.5 + 0.5) * 0.5, 1.0 - t)).rgb;
                vec3 lm = textureLod(u_albedo[18], vec2(0.25, 0.5), 8.0).rgb;
                albedo *= clamp(lt / max(lm, vec3(0.03)), vec3(0.55), vec3(1.6));
                if (fern) albedo = albedo * (vColor / vec3(0.085, 0.20, 0.045)) * 0.9;   // ferns keep their own green
            }
            // Relief: the blade bulges between the veins.
            vec3 across = normalize(cross(N, vec3(0.0, 1.0, 0.0)) + vec3(1e-4));
            float bump = (fract(vph) - 0.5) * (1.0 - vein) * (1.0 - smoothstep(0.25, 0.9, vfw));
            N = normalize(N + across * (sign(u) * 0.10 * (1.0 - mid)) + cross(N, across) * (bump * 0.35));
        }
    }

    vec3  L = normalize(scene.sun_direction.xyz);
    // Half-Lambert wrap (same reason as grass.frag): hard Lambert on
    // randomly-rotated faceted rocks floors half the faces to black.
    float ndl_raw = dot(N, L);
    float wrap = ndl_raw * 0.5 + 0.5;
    float ndl = wrap * wrap;
    float lit_sun = ndl * (1.0 - vSunShadow * 0.7);
    float sky_up  = N.y * 0.5 + 0.5;
    vec3  amb = mix(scene.ambient.rgb, scene.sky_color.rgb * 0.45, sky_up) *
                scene.rt_params.z;
    vec3 lit = albedo * (scene.sun_color.rgb * scene.sun_color.a * lit_sun
                          + amb);

    outColor = vec4(lit, 1.0);
    // Material 1: the deferred pass re-shades with RT sun shadows, AO and
    // GI like the rest of the world (was pre-lit material 5 with only
    // the coarse shadow map).
    outGBuffer0 = vec4(clamp(albedo, vec3(0.0), vec3(1.0)), 1.0 / 255.0);
    // Plants: flagged as leaves so the deferred pass lights them with the
    // foliage scattering model (wrap + transmission).
    outGBuffer1 = vec4(_octa_encode(N), 0.6, plant ? 1.0 : 0.0);
    {
        vec2 current_uv = gl_FragCoord.xy * scene.viewport.zw;
        if (vPrevClip.w > 0.0) {
            vec2 prev_ndc = vPrevClip.xy / vPrevClip.w;
            // Unjittered motion, same convention as cube.frag.
            outMotion = current_uv - (prev_ndc * 0.5 + 0.5) - scene.restir_params.zw;
        } else {
            outMotion = vec2(0.0);
        }
    }
}
