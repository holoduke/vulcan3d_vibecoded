// Physically based sky / atmosphere — Hillaire 2020, "A Scalable and
// Production Ready Sky and Atmosphere Rendering Technique" (EGSR).
// Shared by atmo_*.comp (LUT generation) and compose.frag (sky + aerial
// perspective). Units: kilometres, planet centre at the origin; the
// camera sits at (0, R_ground + height, 0) in this space.

#ifndef ATMOSPHERE_GLSL
#define ATMOSPHERE_GLSL

const float kAtmoPI       = 3.14159265358979;
const float kBottomRadius = 6360.0;
const float kTopRadius    = 6460.0;

const vec3  kRayleighScattering = vec3(5.802, 13.558, 33.1) * 1e-3;   // /km
const float kRayleighScaleH     = 8.0;
const vec3  kMieScattering      = vec3(3.996e-3);
const vec3  kMieExtinction      = vec3(4.440e-3);
const float kMieScaleH          = 1.2;
const float kMieG               = 0.8;
const vec3  kOzoneAbsorption    = vec3(0.650, 1.881, 0.085) * 1e-3;
const vec3  kGroundAlbedo       = vec3(0.3);

// LUT sizes (must match atmosphere.cpp).
const ivec2 kTransLut   = ivec2(256, 64);
const ivec2 kMsLut      = ivec2(32, 32);
const ivec2 kSkyViewLut = ivec2(192, 108);

struct Medium {
    vec3 scat_r;   // Rayleigh scattering
    vec3 scat_m;   // Mie scattering
    vec3 ext;      // total extinction
};

Medium sample_medium(float h) {   // h = height above ground (km)
    h = max(h, 0.0);
    float dr = exp(-h / kRayleighScaleH);
    float dm = exp(-h / kMieScaleH);
    float doz = max(0.0, 1.0 - abs(h - 25.0) / 15.0);
    Medium m;
    m.scat_r = kRayleighScattering * dr;
    m.scat_m = kMieScattering * dm;
    m.ext = m.scat_r + kMieExtinction * dm + kOzoneAbsorption * doz;
    return m;
}

float phase_rayleigh(float c) { return 3.0 / (16.0 * kAtmoPI) * (1.0 + c * c); }
float phase_mie(float c) {        // Cornette-Shanks
    float g = kMieG, g2 = g * g;
    float k = 3.0 / (8.0 * kAtmoPI) * (1.0 - g2) / (2.0 + g2);
    return k * (1.0 + c * c) / pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5);
}

// Nearest positive hit distance of a ray with a sphere at the origin;
// -1 when missed.
float ray_sphere(vec3 ro, vec3 rd, float r) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float d = b * b - c;
    if (d < 0.0) return -1.0;
    float s = sqrt(d);
    float t0 = -b - s, t1 = -b + s;
    if (t0 > 0.0) return t0;
    if (t1 > 0.0) return t1;
    return -1.0;
}

// Transmittance LUT parameterisation: x = cos(view zenith) remapped,
// y = height. Simple (non-Bruneton) mapping with extra precision near the
// horizon through a sqrt warp on mu.
vec2 trans_uv(float r, float mu) {
    float x = 0.5 + 0.5 * sign(mu) * sqrt(abs(mu));
    float y = clamp((r - kBottomRadius) / (kTopRadius - kBottomRadius), 0.0, 1.0);
    return vec2(x, sqrt(y));
}
void trans_params(vec2 uv, out float r, out float mu) {
    float m = uv.x * 2.0 - 1.0;
    mu = sign(m) * m * m;
    r = kBottomRadius + uv.y * uv.y * (kTopRadius - kBottomRadius);
}

// Multi-scattering LUT: x = cos(sun zenith), y = height.
vec2 ms_uv(float r, float mu_s) {
    return vec2(clamp(0.5 + 0.5 * mu_s, 0.0, 1.0),
                clamp((r - kBottomRadius) / (kTopRadius - kBottomRadius), 0.0, 1.0));
}

// Sky-view LUT: x = azimuth relative to the sun, y = view elevation with
// a quadratic warp so the horizon gets most of the texels.
vec2 skyview_uv(float elev, float az) {      // elev in [-pi/2, pi/2], az in [0, pi]
    float v = elev / (0.5 * kAtmoPI);          // [-1, 1]
    float y = 0.5 + 0.5 * sign(v) * sqrt(abs(v));
    return vec2(az / kAtmoPI, y);
}
void skyview_params(vec2 uv, out float elev, out float az) {
    float v = uv.y * 2.0 - 1.0;
    elev = sign(v) * v * v * 0.5 * kAtmoPI;
    az = uv.x * kAtmoPI;
}

#endif
