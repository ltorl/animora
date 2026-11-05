// Animora native renderer
// ----------------------------------------------------------------------------
// A multithreaded, physically based unidirectional path tracer with:
//   * binned-SAH BVH with motion-blur (two time samples per vertex)
//   * principled BSDF: Burley-ish diffuse + sheen, GGX metal/dielectric
//     specular (VNDF sampling), clearcoat, rough/smooth/thin glass with Beer
//     absorption, random-walk subsurface scattering, emission, alpha cut-outs
//   * next-event estimation + multiple importance sampling (power heuristic)
//     for emissive meshes, sphere point/spot lights, distant (sun) lights with
//     angular size and an importance-sampled HDR environment
//   * homogeneous participating media slab (fog / god rays) with HG phase
//   * Owen-scrambled Sobol sampling, Gaussian pixel filter, thin-lens depth of
//     field with polygonal bokeh, adaptive sampling, firefly clamping
//   * AOVs (albedo, normal, depth, variance) + edge-avoiding a-trous denoiser
//
// Exposed through a small C ABI that is consumed via Python ctypes.
// Build: clang++ -O3 -std=c++17 -shared -fPIC -o libanimora.so pathtracer.cpp
// ----------------------------------------------------------------------------
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#define AN_EXPORT extern "C" __attribute__((visibility("default")))

static const float PI = 3.14159265358979323846f;
static const float INV_PI = 0.31830988618379067f;
static const float INF = 1e30f;

// ============================================================================
// Math
// ============================================================================
struct V3 {
  float x, y, z;
  V3() : x(0), y(0), z(0) {}
  explicit V3(float a) : x(a), y(a), z(a) {}
  V3(float a, float b, float c) : x(a), y(b), z(c) {}
  float operator[](int i) const { return (&x)[i]; }
  float &operator[](int i) { return (&x)[i]; }
};
static inline V3 operator+(const V3 &a, const V3 &b) { return V3(a.x + b.x, a.y + b.y, a.z + b.z); }
static inline V3 operator-(const V3 &a, const V3 &b) { return V3(a.x - b.x, a.y - b.y, a.z - b.z); }
static inline V3 operator-(const V3 &a) { return V3(-a.x, -a.y, -a.z); }
static inline V3 operator*(const V3 &a, const V3 &b) { return V3(a.x * b.x, a.y * b.y, a.z * b.z); }
static inline V3 operator*(const V3 &a, float s) { return V3(a.x * s, a.y * s, a.z * s); }
static inline V3 operator*(float s, const V3 &a) { return V3(a.x * s, a.y * s, a.z * s); }
static inline V3 operator/(const V3 &a, float s) { float i = 1.f / s; return V3(a.x * i, a.y * i, a.z * i); }
static inline V3 operator/(const V3 &a, const V3 &b) { return V3(a.x / b.x, a.y / b.y, a.z / b.z); }
static inline V3 &operator+=(V3 &a, const V3 &b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static inline V3 &operator*=(V3 &a, const V3 &b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static inline V3 &operator*=(V3 &a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }
static inline float dot(const V3 &a, const V3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(const V3 &a, const V3 &b) {
  return V3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
static inline float length(const V3 &a) { return std::sqrt(dot(a, a)); }
static inline V3 normalize(const V3 &a) {
  float l = length(a);
  return l > 0 ? a / l : V3(0, 0, 1);
}
static inline V3 lerp(const V3 &a, const V3 &b, float t) { return a + (b - a) * t; }
static inline float lerpf(float a, float b, float t) { return a + (b - a) * t; }
static inline float maxc(const V3 &a) { return std::max(a.x, std::max(a.y, a.z)); }
static inline float lum(const V3 &a) { return 0.2126f * a.x + 0.7152f * a.y + 0.0722f * a.z; }
static inline V3 vmin(const V3 &a, const V3 &b) { return V3(std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)); }
static inline V3 vmax(const V3 &a, const V3 &b) { return V3(std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)); }
static inline V3 vexp(const V3 &a) { return V3(std::exp(a.x), std::exp(a.y), std::exp(a.z)); }
static inline float clampf(float v, float a, float b) { return v < a ? a : (v > b ? b : v); }
static inline float saturate(float v) { return clampf(v, 0.f, 1.f); }
static inline V3 reflect(const V3 &v, const V3 &n) { return v - n * (2.f * dot(v, n)); }
static inline bool isfinite3(const V3 &v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); }
static inline float smoothstepf(float a, float b, float x) {
  if (b <= a) return x >= b ? 1.f : 0.f;
  float t = saturate((x - a) / (b - a));
  return t * t * (3 - 2 * t);
}
static inline float sqr(float x) { return x * x; }

struct Frame {
  V3 t, b, n;
  Frame() {}
  explicit Frame(const V3 &nn) {
    n = nn;
    // Duff et al. 2017 branchless ONB
    float sign = std::copysign(1.0f, n.z);
    float a = -1.0f / (sign + n.z);
    float bb = n.x * n.y * a;
    t = V3(1.0f + sign * n.x * n.x * a, sign * bb, -sign * n.x);
    b = V3(bb, sign + n.y * n.y * a, -n.y);
  }
  Frame(const V3 &nn, const V3 &tangent) {
    n = nn;
    V3 tt = tangent - n * dot(n, tangent);
    float l = length(tt);
    if (l < 1e-6f) { *this = Frame(nn); return; }
    t = tt / l;
    b = cross(n, t);
  }
  V3 to_local(const V3 &v) const { return V3(dot(v, t), dot(v, b), dot(v, n)); }
  V3 to_world(const V3 &v) const { return t * v.x + b * v.y + n * v.z; }
};

// Robust ray origin offset (Waechter & Binder, Ray Tracing Gems ch. 6)
static inline float int_as_float(int32_t i) { float f; std::memcpy(&f, &i, 4); return f; }
static inline int32_t float_as_int(float f) { int32_t i; std::memcpy(&i, &f, 4); return i; }
static inline V3 offset_ray(const V3 &p, const V3 &n) {
  const float origin = 1.0f / 32.0f, float_scale = 1.0f / 65536.0f, int_scale = 256.0f;
  int32_t ox = (int32_t)(int_scale * n.x), oy = (int32_t)(int_scale * n.y), oz = (int32_t)(int_scale * n.z);
  V3 pi(int_as_float(float_as_int(p.x) + ((p.x < 0) ? -ox : ox)),
        int_as_float(float_as_int(p.y) + ((p.y < 0) ? -oy : oy)),
        int_as_float(float_as_int(p.z) + ((p.z < 0) ? -oz : oz)));
  return V3(std::fabs(p.x) < origin ? p.x + float_scale * n.x : pi.x,
            std::fabs(p.y) < origin ? p.y + float_scale * n.y : pi.y,
            std::fabs(p.z) < origin ? p.z + float_scale * n.z : pi.z);
}

// ============================================================================
// Sampling: Owen-scrambled, shuffled Sobol (Burley 2020)
// ============================================================================
static inline uint32_t hash_u32(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}
static inline uint32_t float_bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static inline uint32_t hash_combine(uint32_t seed, uint32_t v) {
  return hash_u32(seed ^ (v + 0x9e3779b9U + (seed << 6) + (seed >> 2)));
}
static inline uint32_t reverse_bits(uint32_t x) {
  x = ((x >> 1) & 0x55555555U) | ((x & 0x55555555U) << 1);
  x = ((x >> 2) & 0x33333333U) | ((x & 0x33333333U) << 2);
  x = ((x >> 4) & 0x0F0F0F0FU) | ((x & 0x0F0F0F0FU) << 4);
  x = ((x >> 8) & 0x00FF00FFU) | ((x & 0x00FF00FFU) << 8);
  return (x >> 16) | (x << 16);
}
static inline uint32_t lk_perm(uint32_t x, uint32_t seed) {
  x += seed; x ^= x * 0x6c50b47cU; x ^= x * 0xb82f1e52U; x ^= x * 0xc7afe638U; x ^= x * 0x8d22f6e6U;
  return x;
}
static inline uint32_t nus(uint32_t x, uint32_t seed) { return reverse_bits(lk_perm(reverse_bits(x), seed)); }
static inline uint32_t sobol1(uint32_t i) {
  uint32_t r = 0;
  for (uint32_t v = 1U << 31; i; i >>= 1, v ^= v >> 1)
    if (i & 1) r ^= v;
  return r;
}
static inline float u32_to_float(uint32_t x) { return std::min((float)(x >> 8) * (1.0f / 16777216.0f), 0.99999994f); }

struct Sampler {
  uint32_t seed = 0, index = 0, dim = 0;
  void start(uint32_t pixel_hash, uint32_t idx) { seed = pixel_hash; index = idx; dim = 0; }
  void get2(float &a, float &b) {
    uint32_t s = hash_combine(seed, dim++);
    uint32_t i = nus(index, s);
    a = u32_to_float(nus(reverse_bits(i), hash_u32(s ^ 0xa511e9b3U)));
    b = u32_to_float(nus(sobol1(i), hash_u32(s ^ 0x63d83595U)));
  }
  float get1() { float a, b; get2(a, b); return a; }
};

// Simple PCG for non-stratified decisions (random walk SSS, etc.)
struct PCG {
  uint64_t state;
  explicit PCG(uint64_t s) : state(s * 6364136223846793005ULL + 1442695040888963407ULL) {}
  uint32_t next() {
    uint64_t old = state;
    state = old * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t xs = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xs >> rot) | (xs << ((-rot) & 31));
  }
  float uniform() { return u32_to_float(next()); }
};

static inline V3 cosine_hemisphere(float u1, float u2) {
  float r = std::sqrt(u1), phi = 2 * PI * u2;
  return V3(r * std::cos(phi), r * std::sin(phi), std::sqrt(std::max(0.f, 1 - u1)));
}
static inline V3 uniform_sphere(float u1, float u2) {
  float z = 1 - 2 * u1, r = std::sqrt(std::max(0.f, 1 - z * z)), phi = 2 * PI * u2;
  return V3(r * std::cos(phi), r * std::sin(phi), z);
}
static inline V3 uniform_cone(float u1, float u2, float cos_max) {
  float c = 1 - u1 * (1 - cos_max), s = std::sqrt(std::max(0.f, 1 - c * c)), phi = 2 * PI * u2;
  return V3(s * std::cos(phi), s * std::sin(phi), c);
}
static inline void concentric_disk(float u1, float u2, float &dx, float &dy) {
  float a = 2 * u1 - 1, b = 2 * u2 - 1;
  if (a == 0 && b == 0) { dx = dy = 0; return; }
  float r, phi;
  if (std::fabs(a) > std::fabs(b)) { r = a; phi = (PI / 4) * (b / a); }
  else { r = b; phi = (PI / 2) - (PI / 4) * (a / b); }
  dx = r * std::cos(phi); dy = r * std::sin(phi);
}
static inline float power_heuristic(float a, float b) {
  float a2 = a * a, b2 = b * b;
  if (!(a2 + b2 > 0)) return 0;
  if (std::isinf(a2)) return 1;
  return a2 / (a2 + b2);
}

// ============================================================================
// C ABI structures (mirrored in animora/render/native.py)
// ============================================================================
extern "C" {
struct AnTexture {
  int32_t type, width, height, channels;
  int32_t offset, space, wrap, interp;
  float color1[3];
  float color2[3];
  float scale;
  float params[6];
  float uv_scale[2];
  float uv_offset[2];
  float uv_rotation;
  float pad;
};

struct AnMaterial {
  float base_color[3];
  float metallic;
  float roughness, ior, transmission, specular;
  float specular_tint, clearcoat, clearcoat_roughness, sheen;
  float sheen_tint, subsurface;
  float subsurface_radius[3];
  float subsurface_scale;
  float emission[3];
  float emission_strength;
  float alpha, normal_strength, bump_strength, transmission_roughness;
  float attenuation_color[3];
  float attenuation_distance;
  int32_t tex_base_color, tex_roughness, tex_metallic, tex_normal;
  int32_t tex_emission, tex_alpha, tex_bump, tex_transmission;
  int32_t thin_walled, emission_one_sided, shadow_catcher, flags;
};

struct AnLight {
  int32_t type;  // 0 point, 1 spot, 2 distant
  int32_t visible, cast_shadows, pad0;
  float position[3];
  float radius;
  float direction[3];
  float angle;  // spot: outer half angle; distant: angular radius (radians)
  float color[3];
  float intensity;
  float blend;
  float pad1[3];
};

struct AnCamera {
  float pos0[3], fwd0[3], up0[3];
  float pos1[3], fwd1[3], up1[3];
  float fov_y;
  int32_t projection;  // 0 perspective, 1 orthographic, 2 equirect panorama, 3 fisheye
  float ortho_height, aperture_radius, focus_distance;
  int32_t blades;
  float blade_rotation, shift_x, shift_y, near_clip;
};

// Heterogeneous participating medium bounded by a box. world_to_local maps world space into the
// unit cube [0,1]^3 (row-major 3x4). Density is either a dense grid (voldata) or procedural fbm.
struct AnVolume {
  int32_t type;            // 0 grid, 1 procedural
  int32_t nx, ny, nz;
  int32_t density_offset;  // into voldata (grid)
  int32_t heat_offset;     // into voldata, -1 = none
  int32_t bb_offset, bb_size;  // blackbody rgb lut (heat 0..1) in voldata, size entries
  int32_t shape;           // procedural: 0 box, 1 sphere, 2 cylinder
  int32_t octaves;
  float world_to_local[12];
  float albedo[3];
  float density;           // extinction multiplier (1/world unit)
  float emission[3];
  float emission_strength; // emission per unit length (scaled by normalized density)
  float anisotropy;
  float blackbody_strength;
  float majorant;          // >= max extinction anywhere in the box
  float frequency, gain, lacunarity, falloff, threshold, contrast;
  float offset[3];
  float heat_from_density; // procedural/grid without heat: heat = density_norm * this
};

struct AnScene {
  int32_t num_tris, has_motion;
  const float *P0;
  const float *P1;
  const float *N0;
  const float *N1;
  const float *UV;
  const float *PREF;
  const int32_t *tri_mat;
  const int32_t *tri_obj;
  int32_t num_materials, num_textures;
  const AnMaterial *materials;
  const AnTexture *textures;
  const float *texdata;
  int32_t num_lights, env_tex;
  const AnLight *lights;
  float env_strength, env_rotation;
  int32_t env_visible, pad0;
  float env_bg[3];
  float fog_density;
  float fog_color[3];
  float fog_anisotropy;
  float fog_height_min, fog_height_max;
  AnCamera camera;
  int32_t num_volumes, pad1;
  const AnVolume *volumes;
  const float *voldata;
};

struct AnSettings {
  int32_t width, height, spp, min_spp;
  float adaptive_threshold;
  int32_t max_depth, threads;
  uint32_t seed;
  float clamp_indirect;
  int32_t integrator, film_transparent, caustics;
  float filter_width, ao_distance;
  int32_t rr_depth, tile_size;
  float wire_width;
  int32_t region_x0, region_y0, region_x1, region_y1;
  float exposure_hint;
};

typedef int (*AnProgress)(float fraction);
}

enum { TEX_IMAGE = 0, TEX_CHECKER, TEX_NOISE, TEX_MARBLE, TEX_WOOD, TEX_GRID, TEX_GRADIENT, TEX_VORONOI, TEX_BRICK, TEX_STRIPES, TEX_CONSTANT };
enum { INT_PATH = 0, INT_DIRECT, INT_AO, INT_NORMALS, INT_ALBEDO, INT_DEPTH, INT_CLAY, INT_UV, INT_WIREFRAME };

// ============================================================================
// Procedural noise
// ============================================================================
struct PerlinTable {
  int p[512];
  PerlinTable() {
    int perm[256];
    for (int i = 0; i < 256; ++i) perm[i] = i;
    uint32_t s = 1337u;
    for (int i = 255; i > 0; --i) {
      s = s * 1664525u + 1013904223u;
      int j = (int)((s >> 8) % (uint32_t)(i + 1));
      std::swap(perm[i], perm[j]);
    }
    for (int i = 0; i < 512; ++i) p[i] = perm[i & 255];
  }
};
static const PerlinTable &perlin_table() {
  static PerlinTable t;
  return t;
}
static inline float fade(float t) { return t * t * t * (t * (t * 6 - 15) + 10); }
static inline float grad3(int h, float x, float y, float z) {
  int hh = h & 15;
  float u = hh < 8 ? x : y;
  float v = hh < 4 ? y : ((hh == 12 || hh == 14) ? x : z);
  return ((hh & 1) ? -u : u) + ((hh & 2) ? -v : v);
}
static float perlin(const V3 &q) {
  const int *p = perlin_table().p;
  float fx = std::floor(q.x), fy = std::floor(q.y), fz = std::floor(q.z);
  int X = ((int)fx) & 255, Y = ((int)fy) & 255, Z = ((int)fz) & 255;
  float x = q.x - fx, y = q.y - fy, z = q.z - fz;
  float u = fade(x), v = fade(y), w = fade(z);
  int A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z, B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;
  return lerpf(lerpf(lerpf(grad3(p[AA], x, y, z), grad3(p[BA], x - 1, y, z), u),
                     lerpf(grad3(p[AB], x, y - 1, z), grad3(p[BB], x - 1, y - 1, z), u), v),
               lerpf(lerpf(grad3(p[AA + 1], x, y, z - 1), grad3(p[BA + 1], x - 1, y, z - 1), u),
                     lerpf(grad3(p[AB + 1], x, y - 1, z - 1), grad3(p[BB + 1], x - 1, y - 1, z - 1), u), v),
               w);
}
static float fbm(V3 q, int octaves, float gain, float lacunarity = 2.0f) {
  float sum = 0, amp = 1, norm = 0;
  octaves = std::max(1, std::min(octaves, 12));
  for (int i = 0; i < octaves; ++i) {
    sum += amp * perlin(q);
    norm += amp;
    amp *= gain;
    q = q * lacunarity + V3(17.13f, 3.71f, 9.27f);
  }
  return sum / norm;
}
static inline float hash3f(int x, int y, int z, int k = 0) {
  uint32_t h = hash_combine(hash_combine(hash_combine((uint32_t)x * 73856093u, (uint32_t)y * 19349663u), (uint32_t)z * 83492791u), (uint32_t)k);
  return (h >> 8) * (1.0f / 16777216.0f);
}

// ============================================================================
// Scene (internal)
// ============================================================================
struct AABB {
  V3 lo{INF, INF, INF}, hi{-INF, -INF, -INF};
  void grow(const V3 &p) { lo = vmin(lo, p); hi = vmax(hi, p); }
  void grow(const AABB &b) { lo = vmin(lo, b.lo); hi = vmax(hi, b.hi); }
  float area() const {
    V3 d = hi - lo;
    if (d.x < 0) return 0;
    return 2 * (d.x * d.y + d.y * d.z + d.z * d.x);
  }
  V3 center() const { return (lo + hi) * 0.5f; }
};

struct BVHNode {
  float lo[3];
  int32_t left_first;
  float hi[3];
  int32_t count;  // >0: leaf
};

struct TriPre { V3 v0, e1, e2; };

struct Ray { V3 o, d; float time; };
struct Hit { float t; int tri; float u, v; };

struct EnvSampler {
  int w = 0, h = 0;
  std::vector<float> marg, cond, func;  // marg: h+1, cond: h*(w+1)
  float integral = 0;
  bool valid = false;
};

struct LightEntry { int type; int index; };  // type 0 analytic light, 1 emissive tri, 2 env

struct Surface {
  V3 p, ng, ns, dpdu, dpdv, orco;
  float u, v;
  int tri, mat, obj;
  bool front;
  float area;
};

struct MatEval {
  V3 base;
  float metallic, roughness, ior, transmission, specular, spec_tint, clearcoat, cc_rough, sheen, sheen_tint, subsurface;
  float trans_rough;
  V3 sss_radius;
  V3 emission;
  float alpha;
  bool thin;
  V3 atten_sigma;
  bool has_atten;
  bool one_sided;
  bool shadow_catcher;
};

struct Engine {
  const AnScene *S = nullptr;
  const AnSettings *ST = nullptr;
  int nt = 0;
  bool motion = false;
  std::vector<TriPre> pre;
  std::vector<BVHNode> nodes;
  std::vector<int32_t> idx;
  std::vector<uint8_t> tri_opaque;
  std::vector<AABB> tb;
  std::vector<V3> tc;
  // lights
  std::vector<LightEntry> lights;
  std::vector<float> light_cdf;  // size n+1
  std::vector<float> light_prob;
  std::vector<int32_t> tri_light;  // tri -> entry index or -1
  int env_entry = -1;
  EnvSampler env;
  V3 scene_center;
  float scene_radius = 1;
  AnMaterial default_mat;

  // ------------------------------------------------------------------ setup
  void tri_verts(int i, float time, V3 &a, V3 &b, V3 &c) const {
    const float *p = S->P0 + (size_t)i * 9;
    a = V3(p[0], p[1], p[2]); b = V3(p[3], p[4], p[5]); c = V3(p[6], p[7], p[8]);
    if (motion) {
      const float *q = S->P1 + (size_t)i * 9;
      a = lerp(a, V3(q[0], q[1], q[2]), time);
      b = lerp(b, V3(q[3], q[4], q[5]), time);
      c = lerp(c, V3(q[6], q[7], q[8]), time);
    }
  }

  const AnMaterial &mat_of(int tri) const {
    int m = S->tri_mat ? S->tri_mat[tri] : 0;
    if (m < 0 || m >= S->num_materials) return default_mat;
    return S->materials[m];
  }

  void setup() {
    std::memset(&default_mat, 0, sizeof(default_mat));
    default_mat.base_color[0] = default_mat.base_color[1] = default_mat.base_color[2] = 0.8f;
    default_mat.roughness = 0.5f; default_mat.ior = 1.5f; default_mat.specular = 0.5f; default_mat.alpha = 1;
    default_mat.tex_base_color = default_mat.tex_roughness = default_mat.tex_metallic = default_mat.tex_normal = -1;
    default_mat.tex_emission = default_mat.tex_alpha = default_mat.tex_bump = default_mat.tex_transmission = -1;
    default_mat.transmission_roughness = -1;

    nt = S->num_tris;
    motion = S->has_motion && S->P1;
    tb.resize(nt);
    tc.resize(nt);
    AABB scene_box;
    for (int i = 0; i < nt; ++i) {
      const float *p = S->P0 + (size_t)i * 9;
      AABB b;
      for (int k = 0; k < 3; ++k) b.grow(V3(p[3 * k], p[3 * k + 1], p[3 * k + 2]));
      if (motion) {
        const float *q = S->P1 + (size_t)i * 9;
        for (int k = 0; k < 3; ++k) b.grow(V3(q[3 * k], q[3 * k + 1], q[3 * k + 2]));
      }
      tb[i] = b;
      tc[i] = b.center();
      scene_box.grow(b);
    }
    if (nt > 0) {
      scene_center = scene_box.center();
      scene_radius = std::max(1e-3f, length(scene_box.hi - scene_box.lo) * 0.5f);
    }
    if (!motion) {
      pre.resize(nt);
      for (int i = 0; i < nt; ++i) {
        V3 a, b, c;
        tri_verts(i, 0, a, b, c);
        pre[i].v0 = a; pre[i].e1 = b - a; pre[i].e2 = c - a;
      }
    }
    tri_opaque.resize(nt);
    for (int i = 0; i < nt; ++i) {
      const AnMaterial &m = mat_of(i);
      bool transparent = (m.alpha < 1.f || m.tex_alpha >= 0) ||
                         (!ST->caustics && m.transmission > 0 && m.metallic < 1.f);
      // 2 = invisible to shadow rays (material flags bit 1, e.g. area-light cards)
      tri_opaque[i] = (m.flags & 2) ? 2 : (transparent ? 0 : 1);
    }
    build_bvh();
    build_lights();
  }

  // ------------------------------------------------------------------ BVH
  void build_bvh() {
    idx.resize(nt);
    for (int i = 0; i < nt; ++i) idx[i] = i;
    nodes.clear();
    nodes.reserve(std::max(1, 2 * nt));
    nodes.push_back(BVHNode());
    if (nt == 0) {
      BVHNode &n = nodes[0];
      n.lo[0] = n.lo[1] = n.lo[2] = 1; n.hi[0] = n.hi[1] = n.hi[2] = -1;
      n.left_first = 0; n.count = 0;
      return;
    }
    subdivide(0, 0, nt, 0);
    tb.clear(); tb.shrink_to_fit();
    tc.clear(); tc.shrink_to_fit();
  }

  void set_node(int ni, const AABB &b, int first, int count) {
    BVHNode &n = nodes[ni];
    n.lo[0] = b.lo.x; n.lo[1] = b.lo.y; n.lo[2] = b.lo.z;
    n.hi[0] = b.hi.x; n.hi[1] = b.hi.y; n.hi[2] = b.hi.z;
    n.left_first = first; n.count = count;
  }

  void subdivide(int ni, int first, int count, int depth) {
    AABB bounds, cb;
    for (int i = first; i < first + count; ++i) { bounds.grow(tb[idx[i]]); cb.grow(tc[idx[i]]); }
    if (count <= 2 || depth > 60) { set_node(ni, bounds, first, count); return; }
    const int BINS = 16;
    float best_cost = INF; int best_axis = -1, best_split = 0;
    for (int axis = 0; axis < 3; ++axis) {
      float lo = cb.lo[axis], hi = cb.hi[axis];
      if (hi - lo < 1e-9f) continue;
      AABB bb[BINS]; int bc[BINS] = {0};
      float scale = BINS / (hi - lo);
      for (int i = first; i < first + count; ++i) {
        int t = idx[i];
        int b = std::min(BINS - 1, (int)((tc[t][axis] - lo) * scale));
        bc[b]++; bb[b].grow(tb[t]);
      }
      float la[BINS - 1], ra[BINS - 1]; int lc[BINS - 1], rc[BINS - 1];
      AABB acc; int cnt = 0;
      for (int i = 0; i < BINS - 1; ++i) { acc.grow(bb[i]); cnt += bc[i]; la[i] = acc.area(); lc[i] = cnt; }
      acc = AABB(); cnt = 0;
      for (int i = BINS - 1; i > 0; --i) { acc.grow(bb[i]); cnt += bc[i]; ra[i - 1] = acc.area(); rc[i - 1] = cnt; }
      for (int i = 0; i < BINS - 1; ++i) {
        if (lc[i] == 0 || rc[i] == 0) continue;
        float c = la[i] * lc[i] + ra[i] * rc[i];
        if (c < best_cost) { best_cost = c; best_axis = axis; best_split = i; }
      }
    }
    float parent_area = std::max(bounds.area(), 1e-20f);
    float split_cost = 0.5f + best_cost / parent_area;
    if (best_axis < 0 || (split_cost >= (float)count && count <= 8)) {
      if (best_axis < 0 && count > 8) {
        // degenerate centroids: median split by index
        int half = count / 2;
        int l = (int)nodes.size();
        nodes.push_back(BVHNode()); nodes.push_back(BVHNode());
        set_node(ni, bounds, l, 0);
        subdivide(l, first, half, depth + 1);
        subdivide(l + 1, first + half, count - half, depth + 1);
        return;
      }
      set_node(ni, bounds, first, count);
      return;
    }
    float lo = cb.lo[best_axis], hi = cb.hi[best_axis];
    float scale = BINS / (hi - lo);
    int *beg = idx.data() + first, *end = beg + count;
    int *mid = std::partition(beg, end, [&](int t) {
      int b = std::min(BINS - 1, (int)((tc[t][best_axis] - lo) * scale));
      return b <= best_split;
    });
    int left_count = (int)(mid - beg);
    if (left_count == 0 || left_count == count) {
      left_count = count / 2;
      std::nth_element(beg, beg + left_count, end, [&](int a, int b) { return tc[a][best_axis] < tc[b][best_axis]; });
    }
    int l = (int)nodes.size();
    nodes.push_back(BVHNode()); nodes.push_back(BVHNode());
    set_node(ni, bounds, l, 0);
    subdivide(l, first, left_count, depth + 1);
    subdivide(l + 1, first + left_count, count - left_count, depth + 1);
  }

  static inline float box_hit(const BVHNode &n, const V3 &o, const V3 &inv, float tmax) {
    float tx1 = (n.lo[0] - o.x) * inv.x, tx2 = (n.hi[0] - o.x) * inv.x;
    float tmin = std::min(tx1, tx2), tmx = std::max(tx1, tx2);
    float ty1 = (n.lo[1] - o.y) * inv.y, ty2 = (n.hi[1] - o.y) * inv.y;
    tmin = std::max(tmin, std::min(ty1, ty2)); tmx = std::min(tmx, std::max(ty1, ty2));
    float tz1 = (n.lo[2] - o.z) * inv.z, tz2 = (n.hi[2] - o.z) * inv.z;
    tmin = std::max(tmin, std::min(tz1, tz2)); tmx = std::min(tmx, std::max(tz1, tz2));
    if (tmx >= std::max(tmin, 0.f) && tmin < tmax) return std::max(tmin, 0.f);
    return INF;
  }

  inline bool tri_hit(int t, const Ray &r, float tmax, float &th, float &u, float &v) const {
    V3 v0, e1, e2;
    if (!motion) { v0 = pre[t].v0; e1 = pre[t].e1; e2 = pre[t].e2; }
    else { V3 a, b, c; tri_verts(t, r.time, a, b, c); v0 = a; e1 = b - a; e2 = c - a; }
    V3 p = cross(r.d, e2);
    float det = dot(e1, p);
    if (std::fabs(det) < 1e-14f) return false;
    float inv = 1.f / det;
    V3 s = r.o - v0;
    u = dot(s, p) * inv;
    if (u < 0.f || u > 1.f) return false;
    V3 q = cross(s, e1);
    v = dot(r.d, q) * inv;
    if (v < 0.f || u + v > 1.f) return false;
    th = dot(e2, q) * inv;
    return th > 0.f && th < tmax;
  }

  bool intersect(const Ray &r, float tmax, Hit &h) const {
    if (nt == 0) return false;
    V3 inv(1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z);
    int stack[96]; int sp = 0;
    h.t = tmax; h.tri = -1;
    if (box_hit(nodes[0], r.o, inv, tmax) >= INF) return false;
    int ni = 0;
    for (;;) {
      const BVHNode &n = nodes[ni];
      if (n.count > 0) {
        for (int i = 0; i < n.count; ++i) {
          int t = idx[n.left_first + i];
          float th, u, v;
          if (tri_hit(t, r, h.t, th, u, v)) { h.t = th; h.tri = t; h.u = u; h.v = v; }
        }
        if (sp == 0) break;
        ni = stack[--sp];
        continue;
      }
      int c0 = n.left_first, c1 = c0 + 1;
      float d0 = box_hit(nodes[c0], r.o, inv, h.t), d1 = box_hit(nodes[c1], r.o, inv, h.t);
      if (d0 > d1) { std::swap(d0, d1); std::swap(c0, c1); }
      if (d0 >= INF) { if (sp == 0) break; ni = stack[--sp]; continue; }
      ni = c0;
      if (d1 < INF) stack[sp++] = c1;
    }
    return h.tri >= 0;
  }

  // returns 0 = clear, 1 = blocked by opaque, 2 = only non-opaque occluders found
  int occluded_any(const Ray &r, float tmax) const {
    if (nt == 0) return 0;
    V3 inv(1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z);
    int stack[96]; int sp = 0;
    if (box_hit(nodes[0], r.o, inv, tmax) >= INF) return 0;
    int ni = 0;
    bool transparent = false;
    for (;;) {
      const BVHNode &n = nodes[ni];
      if (n.count > 0) {
        for (int i = 0; i < n.count; ++i) {
          int t = idx[n.left_first + i];
          float th, u, v;
          if (tri_hit(t, r, tmax, th, u, v)) {
            if (tri_opaque[t] == 1) return 1;
            if (tri_opaque[t] == 0) transparent = true;
          }
        }
        if (sp == 0) break;
        ni = stack[--sp];
        continue;
      }
      int c0 = n.left_first, c1 = c0 + 1;
      float d0 = box_hit(nodes[c0], r.o, inv, tmax), d1 = box_hit(nodes[c1], r.o, inv, tmax);
      if (d0 < INF && d1 < INF) { ni = c0; stack[sp++] = c1; }
      else if (d0 < INF) ni = c0;
      else if (d1 < INF) ni = c1;
      else { if (sp == 0) break; ni = stack[--sp]; }
    }
    return transparent ? 2 : 0;
  }

  // ------------------------------------------------------------- surfaces
  Surface surface(const Hit &h, const Ray &r) const {
    Surface s;
    int t = h.tri;
    s.tri = t;
    V3 a, b, c;
    tri_verts(t, r.time, a, b, c);
    float w = 1 - h.u - h.v;
    s.p = a * w + b * h.u + c * h.v;
    V3 e1 = b - a, e2 = c - a;
    V3 n = cross(e1, e2);
    float nl = length(n);
    s.area = 0.5f * nl;
    s.ng = nl > 0 ? n / nl : V3(0, 0, 1);
    s.front = dot(r.d, s.ng) < 0;
    if (S->N0) {
      const float *q = S->N0 + (size_t)t * 9;
      V3 n0(q[0], q[1], q[2]), n1(q[3], q[4], q[5]), n2(q[6], q[7], q[8]);
      if (motion && S->N1) {
        const float *q1 = S->N1 + (size_t)t * 9;
        n0 = lerp(n0, V3(q1[0], q1[1], q1[2]), r.time);
        n1 = lerp(n1, V3(q1[3], q1[4], q1[5]), r.time);
        n2 = lerp(n2, V3(q1[6], q1[7], q1[8]), r.time);
      }
      // Vertex normals define "outside": orient the geometric normal to agree with them so
      // meshes with inconsistent winding shade correctly.
      if (dot(n0 + n1 + n2, s.ng) < 0) { s.ng = -s.ng; s.front = !s.front; }
      V3 ns = n0 * w + n1 * h.u + n2 * h.v;
      float l = length(ns);
      s.ns = l > 1e-8f ? ns / l : s.ng;
    } else {
      s.ns = s.ng;
    }
    float u0 = 0, v0 = 0, u1 = 1, v1 = 0, u2 = 0, v2 = 1;
    if (S->UV) {
      const float *q = S->UV + (size_t)t * 6;
      u0 = q[0]; v0 = q[1]; u1 = q[2]; v1 = q[3]; u2 = q[4]; v2 = q[5];
    }
    s.u = u0 * w + u1 * h.u + u2 * h.v;
    s.v = v0 * w + v1 * h.u + v2 * h.v;
    // dP/du, dP/dv
    float du1 = u1 - u0, du2 = u2 - u0, dv1 = v1 - v0, dv2 = v2 - v0;
    float det = du1 * dv2 - dv1 * du2;
    if (std::fabs(det) > 1e-12f) {
      float inv = 1.f / det;
      s.dpdu = (e1 * dv2 - e2 * dv1) * inv;
      s.dpdv = (e2 * du1 - e1 * du2) * inv;
    } else {
      Frame f(s.ng);
      s.dpdu = f.t; s.dpdv = f.b;
    }
    if (S->PREF) {
      const float *q = S->PREF + (size_t)t * 9;
      s.orco = V3(q[0], q[1], q[2]) * w + V3(q[3], q[4], q[5]) * h.u + V3(q[6], q[7], q[8]) * h.v;
    } else {
      s.orco = s.p;
    }
    s.mat = S->tri_mat ? S->tri_mat[t] : 0;
    s.obj = S->tri_obj ? S->tri_obj[t] : 0;
    return s;
  }

  // ------------------------------------------------------------- textures
  V3 image_lookup(const AnTexture &t, float u, float v) const {
    if (!S->texdata || t.width <= 0 || t.height <= 0) return V3(1);
    const float *d = S->texdata + t.offset;
    int W = t.width, H = t.height, C = std::max(1, t.channels);
    float fx = u * W - 0.5f, fy = (1.f - v) * H - 0.5f;
    auto fetch = [&](int x, int y) -> V3 {
      if (t.wrap == 0) { x %= W; if (x < 0) x += W; y %= H; if (y < 0) y += H; }
      else { x = std::max(0, std::min(W - 1, x)); y = std::max(0, std::min(H - 1, y)); }
      const float *px = d + ((size_t)y * W + x) * C;
      if (C >= 3) return V3(px[0], px[1], px[2]);
      return V3(px[0]);
    };
    if (t.interp == 1) return fetch((int)std::floor(u * W), (int)std::floor((1.f - v) * H));
    int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy);
    float ax = fx - x0, ay = fy - y0;
    return lerp(lerp(fetch(x0, y0), fetch(x0 + 1, y0), ax), lerp(fetch(x0, y0 + 1), fetch(x0 + 1, y0 + 1), ax), ay);
  }

  float image_alpha(const AnTexture &t, float u, float v) const {
    if (!S->texdata || t.channels < 4) return 1.f;
    const float *d = S->texdata + t.offset;
    int W = t.width, H = t.height;
    int x = (int)std::floor(u * W), y = (int)std::floor((1.f - v) * H);
    x %= W; if (x < 0) x += W; y %= H; if (y < 0) y += H;
    return d[((size_t)y * W + x) * 4 + 3];
  }

  void tex_coords(const AnTexture &t, float u, float v, const V3 &orco, const V3 &pw, float &uu, float &vv, V3 &q) const {
    if (t.space == 0) {
      float su = u * t.uv_scale[0], sv = v * t.uv_scale[1];
      if (t.uv_rotation != 0) {
        float c = std::cos(t.uv_rotation), s = std::sin(t.uv_rotation);
        float ru = su * c - sv * s, rv = su * s + sv * c;
        su = ru; sv = rv;
      }
      uu = su + t.uv_offset[0]; vv = sv + t.uv_offset[1];
      q = V3(uu, vv, 0);
    } else {
      q = (t.space == 1 ? orco : pw);
      uu = q.x; vv = q.y;
    }
  }

  V3 eval_tex(int ti, float u, float v, const V3 &orco, const V3 &pw, float *alpha_out = nullptr) const {
    if (ti < 0 || ti >= S->num_textures) return V3(1);
    const AnTexture &t = S->textures[ti];
    float uu, vv; V3 q;
    tex_coords(t, u, v, orco, pw, uu, vv, q);
    V3 c1(t.color1[0], t.color1[1], t.color1[2]), c2(t.color2[0], t.color2[1], t.color2[2]);
    float sc = t.scale;
    float f = 0;
    switch (t.type) {
      case TEX_IMAGE:
        if (alpha_out) *alpha_out = image_alpha(t, uu, vv);
        return image_lookup(t, uu, vv);
      case TEX_CONSTANT:
        return c1;
      case TEX_CHECKER: {
        V3 p = q * sc;
        int k = (int)std::floor(p.x) + (int)std::floor(p.y) + (t.space == 0 ? 0 : (int)std::floor(p.z));
        f = (k & 1) ? 1.f : 0.f;
        break;
      }
      case TEX_NOISE: {
        V3 p = q * sc;
        if (t.params[2] > 0) p = p + V3(perlin(p + V3(5.2f, 1.3f, 7.7f)), perlin(p + V3(1.7f, 9.2f, 3.1f)), perlin(p + V3(8.3f, 2.8f, 4.4f))) * t.params[2];
        float n = fbm(p, (int)(t.params[0] > 0 ? t.params[0] : 5), t.params[1] > 0 ? t.params[1] : 0.5f);
        float contrast = t.params[3] > 0 ? t.params[3] : 1.f;
        f = saturate(0.5f + 0.9f * n * contrast);
        break;
      }
      case TEX_MARBLE: {
        V3 p = q * sc;
        float turb = fbm(p, (int)(t.params[0] > 0 ? t.params[0] : 6), 0.55f);
        float d = t.params[2] > 0 ? t.params[2] : 5.f;
        f = 0.5f + 0.5f * std::sin(p.x * 2.f + d * turb);
        f = std::pow(saturate(f), t.params[1] > 0 ? t.params[1] : 1.f);
        break;
      }
      case TEX_WOOD: {
        V3 p = q * sc;
        float r = std::sqrt(p.x * p.x + p.z * p.z) * (t.params[0] > 0 ? t.params[0] : 8.f);
        r += (t.params[2] > 0 ? t.params[2] : 1.5f) * fbm(p * 0.8f, 4, 0.5f) + 0.15f * perlin(V3(p.x, p.y * 20.f, p.z));
        float ring = r - std::floor(r);
        f = smoothstepf(0.0f, 0.8f, ring) * (1 - smoothstepf(0.8f, 1.0f, ring));
        break;
      }
      case TEX_GRID: {
        float lw = t.params[0] > 0 ? t.params[0] : 0.05f;
        float gx = q.x * sc, gy = q.y * sc;
        float fx = gx - std::floor(gx), fy = gy - std::floor(gy);
        bool line = fx < lw || fy < lw;
        if (t.space != 0) { float gz = q.z * sc; float fz = gz - std::floor(gz); line = (fx < lw) + (fy < lw) + (fz < lw) >= 2; }
        f = line ? 1.f : 0.f;
        break;
      }
      case TEX_GRADIENT: {
        int axis = (int)t.params[0];
        float val = (axis == 0 ? q.x : (axis == 1 ? q.y : q.z)) * sc;
        if (t.space == 0 && axis == 2) val = vv * sc;
        float lo = t.params[1], hi = t.params[2] != 0 ? t.params[2] : 1.f;
        f = saturate((val - lo) / (hi - lo));
        f = f * f * (3 - 2 * f);
        break;
      }
      case TEX_VORONOI: {
        V3 p = q * sc;
        int ix = (int)std::floor(p.x), iy = (int)std::floor(p.y), iz = (int)std::floor(p.z);
        float best = INF, second = INF; int bx = 0, by = 0, bz = 0;
        int zr = t.space == 0 ? 0 : 1;
        for (int dz = -zr; dz <= zr; ++dz)
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              int cx = ix + dx, cy = iy + dy, cz = iz + dz;
              V3 fp(cx + hash3f(cx, cy, cz, 1), cy + hash3f(cx, cy, cz, 2), t.space == 0 ? p.z : cz + hash3f(cx, cy, cz, 3));
              float d = length(fp - p);
              if (d < best) { second = best; best = d; bx = cx; by = cy; bz = cz; }
              else if (d < second) second = d;
            }
        int mode = (int)t.params[0];
        if (mode == 0) f = hash3f(bx, by, bz, 7);          // cell colour
        else if (mode == 1) f = saturate(best);            // F1 distance
        else f = saturate((second - best) * (t.params[1] > 0 ? t.params[1] : 4.f));  // edges
        break;
      }
      case TEX_BRICK: {
        float mortar = t.params[0] > 0 ? t.params[0] : 0.03f;
        float ratio = t.params[1] > 0 ? t.params[1] : 2.f;
        float bx = q.x * sc / ratio, by = q.y * sc;
        float row = std::floor(by);
        if (((int)row) & 1) bx += 0.5f;
        float col = std::floor(bx);
        float fx = bx - col, fy = by - row;
        bool m = fx < mortar / ratio || fy < mortar || fx > 1 - mortar / ratio || fy > 1 - mortar;
        if (m) return c2;
        float var = t.params[2];
        float hsh = hash3f((int)col, (int)row, 0, 11);
        return c1 * (1.f - var * 0.5f + var * hsh);
      }
      case TEX_STRIPES: {
        int axis = (int)t.params[0];
        float val = (axis == 0 ? q.x : (axis == 1 ? q.y : q.z)) * sc;
        float soft = t.params[1];
        float s = 0.5f + 0.5f * std::sin(val * 2 * PI);
        f = soft > 0 ? smoothstepf(0.5f - soft, 0.5f + soft, s) : (s > 0.5f ? 1.f : 0.f);
        break;
      }
      default:
        return V3(1);
    }
    return lerp(c1, c2, f);
  }

  float eval_tex_scalar(int ti, float u, float v, const V3 &orco, const V3 &pw) const {
    V3 c = eval_tex(ti, u, v, orco, pw);
    return (c.x + c.y + c.z) * (1.f / 3.f);
  }

  // ------------------------------------------------------------- materials
  MatEval eval_material(const Surface &s) const {
    const AnMaterial &M = (s.mat >= 0 && s.mat < S->num_materials) ? S->materials[s.mat] : default_mat;
    MatEval m;
    m.base = V3(M.base_color[0], M.base_color[1], M.base_color[2]);
    m.alpha = M.alpha;
    if (M.tex_base_color >= 0) {
      float a = 1.f;
      m.base = m.base * eval_tex(M.tex_base_color, s.u, s.v, s.orco, s.p, &a);
      if (M.tex_alpha < 0) m.alpha *= a;
    }
    m.metallic = M.metallic;
    if (M.tex_metallic >= 0) m.metallic *= eval_tex_scalar(M.tex_metallic, s.u, s.v, s.orco, s.p);
    m.roughness = M.roughness;
    if (M.tex_roughness >= 0) m.roughness *= eval_tex_scalar(M.tex_roughness, s.u, s.v, s.orco, s.p);
    m.transmission = M.transmission;
    if (M.tex_transmission >= 0) m.transmission *= eval_tex_scalar(M.tex_transmission, s.u, s.v, s.orco, s.p);
    if (M.tex_alpha >= 0) m.alpha *= eval_tex_scalar(M.tex_alpha, s.u, s.v, s.orco, s.p);
    m.ior = M.ior > 1.0001f ? M.ior : 1.0001f;
    m.specular = M.specular;
    m.spec_tint = M.specular_tint;
    m.clearcoat = M.clearcoat;
    m.cc_rough = M.clearcoat_roughness;
    m.sheen = M.sheen;
    m.sheen_tint = M.sheen_tint;
    m.subsurface = M.subsurface;
    m.sss_radius = V3(M.subsurface_radius[0], M.subsurface_radius[1], M.subsurface_radius[2]) * std::max(1e-6f, M.subsurface_scale);
    m.trans_rough = M.transmission_roughness >= 0 ? M.transmission_roughness : m.roughness;
    m.emission = V3(M.emission[0], M.emission[1], M.emission[2]) * M.emission_strength;
    if (M.tex_emission >= 0 && M.emission_strength > 0) m.emission = m.emission * eval_tex(M.tex_emission, s.u, s.v, s.orco, s.p);
    m.thin = M.thin_walled != 0;
    m.one_sided = M.emission_one_sided != 0;
    m.shadow_catcher = M.shadow_catcher != 0;
    m.has_atten = M.attenuation_distance > 0 && m.transmission > 0;
    if (m.has_atten) {
      V3 ac(std::max(1e-4f, M.attenuation_color[0]), std::max(1e-4f, M.attenuation_color[1]), std::max(1e-4f, M.attenuation_color[2]));
      m.atten_sigma = V3(-std::log(ac.x), -std::log(ac.y), -std::log(ac.z)) / M.attenuation_distance;
    }
    m.metallic = saturate(m.metallic);
    m.roughness = saturate(m.roughness);
    m.transmission = saturate(m.transmission);
    m.subsurface = saturate(m.subsurface);
    m.alpha = saturate(m.alpha);
    return m;
  }

  // Apply normal / bump map to shading normal
  void perturb_normal(Surface &s) const {
    const AnMaterial &M = (s.mat >= 0 && s.mat < S->num_materials) ? S->materials[s.mat] : default_mat;
    if (M.tex_normal >= 0 && M.normal_strength != 0) {
      V3 c = eval_tex(M.tex_normal, s.u, s.v, s.orco, s.p);
      V3 tn(c.x * 2 - 1, c.y * 2 - 1, c.z * 2 - 1);
      tn.x *= M.normal_strength; tn.y *= M.normal_strength;
      Frame f(s.ns, s.dpdu);
      V3 b = s.dpdv;
      // respect UV handedness
      if (dot(cross(f.n, f.t), b) < 0) f.b = -f.b;
      V3 n = normalize(f.to_world(tn));
      if (isfinite3(n)) s.ns = n;
    }
    if (M.tex_bump >= 0 && M.bump_strength != 0) {
      const AnTexture &t = S->textures[M.tex_bump];
      float h0 = eval_tex_scalar(M.tex_bump, s.u, s.v, s.orco, s.p);
      Frame f(s.ns, s.dpdu);
      float hu, hv;
      if (t.space == 0) {
        float du = 1e-3f / std::max(1e-6f, std::fabs(t.uv_scale[0])), dv = 1e-3f / std::max(1e-6f, std::fabs(t.uv_scale[1]));
        hu = (eval_tex_scalar(M.tex_bump, s.u + du, s.v, s.orco, s.p) - h0) / du;
        hv = (eval_tex_scalar(M.tex_bump, s.u, s.v + dv, s.orco, s.p) - h0) / dv;
        float lu = std::max(1e-6f, length(s.dpdu)), lv = std::max(1e-6f, length(s.dpdv));
        hu /= lu; hv /= lv;
        V3 tu = normalize(s.dpdu), tv = normalize(s.dpdv);
        V3 n = normalize(s.ns - (tu * hu + tv * hv) * (M.bump_strength * 0.01f));
        if (isfinite3(n)) s.ns = n;
      } else {
        float eps = 1e-3f / std::max(1e-3f, t.scale);
        hu = (eval_tex_scalar(M.tex_bump, s.u, s.v, s.orco + f.t * eps, s.p + f.t * eps) - h0) / eps;
        hv = (eval_tex_scalar(M.tex_bump, s.u, s.v, s.orco + f.b * eps, s.p + f.b * eps) - h0) / eps;
        V3 n = normalize(s.ns - (f.t * hu + f.b * hv) * (M.bump_strength * 0.01f));
        if (isfinite3(n)) s.ns = n;
      }
    }
  }

  // ------------------------------------------------------------- lights
  float light_power(const AnLight &L) const {
    V3 c(L.color[0], L.color[1], L.color[2]);
    float I = L.intensity * std::max(1e-6f, lum(c));
    if (L.type == 0) return 4 * PI * I;
    if (L.type == 1) return 2 * PI * (1 - std::cos(std::max(1e-3f, L.angle))) * I;
    return I * PI * scene_radius * scene_radius;
  }

  void build_env() {
    env.valid = false;
    if (S->env_tex < 0 || S->env_tex >= S->num_textures || S->env_strength <= 0) return;
    const AnTexture &t = S->textures[S->env_tex];
    if (t.type != TEX_IMAGE || t.width <= 0 || t.height <= 0) return;
    // sampling resolution capped for speed
    int W = std::min(t.width, 1024), H = std::min(t.height, 512);
    env.w = W; env.h = H;
    env.func.assign((size_t)W * H, 0.f);
    env.cond.assign((size_t)H * (W + 1), 0.f);
    env.marg.assign(H + 1, 0.f);
    std::vector<float> rowsum(H, 0.f);
    for (int y = 0; y < H; ++y) {
      float theta = (y + 0.5f) / H * PI;
      float st = std::sin(theta);
      for (int x = 0; x < W; ++x) {
        // average a small footprint of the source image
        float u = (x + 0.5f) / W, v = 1.f - (y + 0.5f) / H;
        float val = lum(image_lookup(t, u, v));
        val = std::max(val, 0.f) * st + 1e-8f;
        env.func[(size_t)y * W + x] = val;
      }
      float *c = &env.cond[(size_t)y * (W + 1)];
      c[0] = 0;
      for (int x = 0; x < W; ++x) c[x + 1] = c[x] + env.func[(size_t)y * W + x] / W;
      rowsum[y] = c[W];
      if (c[W] > 0) for (int x = 1; x <= W; ++x) c[x] /= c[W];
      else for (int x = 1; x <= W; ++x) c[x] = (float)x / W;
    }
    env.marg[0] = 0;
    for (int y = 0; y < H; ++y) env.marg[y + 1] = env.marg[y] + rowsum[y] / H;
    env.integral = env.marg[H];
    if (env.integral > 0) for (int y = 1; y <= H; ++y) env.marg[y] /= env.integral;
    env.valid = env.integral > 0;
  }

  static int sample_cdf(const float *cdf, int n, float u, float &frac) {
    // cdf has n+1 entries, cdf[0]=0, cdf[n]=1
    int lo = 0, hi = n;
    while (lo + 1 < hi) {
      int mid = (lo + hi) >> 1;
      if (cdf[mid] <= u) lo = mid; else hi = mid;
    }
    float d = cdf[lo + 1] - cdf[lo];
    frac = d > 0 ? (u - cdf[lo]) / d : 0.5f;
    return lo;
  }

  V3 env_dir_rotate(const V3 &d, float ang) const {
    float c = std::cos(ang), s = std::sin(ang);
    return V3(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
  }

  void dir_to_uv(const V3 &dw, float &u, float &v) const {
    V3 d = env_dir_rotate(dw, -S->env_rotation);
    u = 0.5f + std::atan2(d.x, -d.z) * (0.5f * INV_PI);
    v = 1.f - std::acos(clampf(d.y, -1.f, 1.f)) * INV_PI;
  }
  V3 uv_to_dir(float u, float v) const {
    float phi = (u - 0.5f) * 2 * PI;
    float theta = (1.f - v) * PI;
    float st = std::sin(theta);
    V3 d(st * std::sin(phi), std::cos(theta), -st * std::cos(phi));
    return env_dir_rotate(d, S->env_rotation);
  }

  V3 env_radiance(const V3 &d) const {
    if (S->env_tex < 0 || S->env_tex >= S->num_textures || S->env_strength <= 0) return V3(0);
    const AnTexture &t = S->textures[S->env_tex];
    if (t.type == TEX_CONSTANT) return V3(t.color1[0], t.color1[1], t.color1[2]) * S->env_strength;
    float u, v;
    dir_to_uv(d, u, v);
    return image_lookup(t, u, v) * S->env_strength;
  }

  float env_pdf(const V3 &d) const {
    if (!env.valid) return 1.f / (4 * PI);
    float u, v;
    dir_to_uv(d, u, v);
    int x = std::min(env.w - 1, std::max(0, (int)(u * env.w)));
    int y = std::min(env.h - 1, std::max(0, (int)((1.f - v) * env.h)));
    float theta = (y + 0.5f) / env.h * PI;
    float st = std::sin(theta);
    if (st <= 0) return 0;
    float pdf_uv = env.func[(size_t)y * env.w + x] / env.integral;
    return pdf_uv / (2 * PI * PI * st);
  }

  V3 env_sample(float u1, float u2, float &pdf) const {
    if (!env.valid) {
      V3 d = uniform_sphere(u1, u2);
      pdf = 1.f / (4 * PI);
      return d;
    }
    float fy, fx;
    int y = sample_cdf(env.marg.data(), env.h, u1, fy);
    int x = sample_cdf(&env.cond[(size_t)y * (env.w + 1)], env.w, u2, fx);
    float u = (x + fx) / env.w;
    float vv = 1.f - (y + fy) / env.h;
    float theta = (y + fy) / env.h * PI;
    float st = std::sin(theta);
    float pdf_uv = env.func[(size_t)y * env.w + x] / env.integral;
    pdf = st > 0 ? pdf_uv / (2 * PI * PI * st) : 0;
    return uv_to_dir(u, vv);
  }

  void build_lights() {
    lights.clear();
    std::vector<float> w;
    for (int i = 0; i < S->num_lights; ++i) {
      const AnLight &L = S->lights[i];
      if (L.intensity <= 0) continue;
      lights.push_back({0, i});
      w.push_back(light_power(L));
    }
    tri_light.assign(nt, -1);
    for (int t = 0; t < nt; ++t) {
      const AnMaterial &m = mat_of(t);
      if (m.emission_strength <= 0) continue;
      V3 e(m.emission[0], m.emission[1], m.emission[2]);
      float le = lum(e) * m.emission_strength;
      if (m.tex_emission >= 0) le *= 0.5f;
      if (le <= 0) continue;
      V3 a, b, c;
      tri_verts(t, 0.5f, a, b, c);
      float area = 0.5f * length(cross(b - a, c - a));
      if (area <= 0) continue;
      tri_light[t] = (int)lights.size();
      lights.push_back({1, t});
      w.push_back(PI * area * le * (m.emission_one_sided ? 1.f : 2.f));
    }
    build_env();
    if (S->env_tex >= 0 && S->env_tex < S->num_textures && S->env_strength > 0) {
      const AnTexture &t = S->textures[S->env_tex];
      float avg = 0;
      if (t.type == TEX_CONSTANT) avg = lum(V3(t.color1[0], t.color1[1], t.color1[2]));
      else if (env.valid) avg = env.integral * 2.f / PI;  // integral includes sin(theta), mean(sin)=2/pi
      avg *= S->env_strength;
      if (avg > 0) {
        env_entry = (int)lights.size();
        lights.push_back({2, 0});
        w.push_back(PI * PI * scene_radius * scene_radius * avg);
      }
    }
    int n = (int)lights.size();
    light_cdf.assign(n + 1, 0.f);
    light_prob.assign(n, 0.f);
    double total = 0;
    for (int i = 0; i < n; ++i) total += std::max(0.f, w[i]);
    if (total <= 0) {
      for (int i = 0; i < n; ++i) light_prob[i] = 1.f / n;
    } else {
      for (int i = 0; i < n; ++i) light_prob[i] = (float)(std::max(0.f, w[i]) / total);
    }
    // Robust selection: mix the power-proportional distribution with a uniform share per light
    // "group" (each analytic light, the environment, and all emissive triangles together count as
    // one group). Power estimates of sun / sky scale with the scene radius, so on scenes with a
    // huge ground plane a pure power CDF starves small spot / point lights (white fireflies).
    if (n > 1) {
      int ngroups = 0, ntri = 0;
      double tri_pow = 0;
      for (int i = 0; i < n; ++i) {
        if (lights[i].type == 1) { ++ntri; tri_pow += std::max(0.f, w[i]); }
        else ++ngroups;
      }
      if (ntri) ++ngroups;
      const float mix = 0.5f;
      for (int i = 0; i < n; ++i) {
        float u;
        if (lights[i].type == 1)
          u = tri_pow > 0 ? (float)(std::max(0.f, w[i]) / tri_pow) / ngroups : 1.f / (ntri * ngroups);
        else
          u = 1.f / ngroups;
        light_prob[i] = (1 - mix) * light_prob[i] + mix * u;
      }
      double s = 0;
      for (int i = 0; i < n; ++i) s += light_prob[i];
      for (int i = 0; i < n; ++i) light_prob[i] = (float)(light_prob[i] / s);
    }
    for (int i = 0; i < n; ++i) light_cdf[i + 1] = light_cdf[i] + light_prob[i];
    if (n > 0) light_cdf[n] = 1.f;
  }

  struct LightSample {
    V3 wi;      // direction to light
    float dist; // distance (INF for distant/env)
    V3 Le;      // radiance arriving
    float pdf;  // solid angle pdf including selection
    bool delta;
  };

  // spot falloff
  float spot_factor(const AnLight &L, const V3 &dir_from_light) const {
    if (L.type != 1) return 1.f;
    V3 d = normalize(V3(L.direction[0], L.direction[1], L.direction[2]));
    float c = dot(d, dir_from_light);
    float cos_outer = std::cos(L.angle);
    float cos_inner = std::cos(L.angle * (1.f - saturate(L.blend)));
    return smoothstepf(cos_outer, std::max(cos_inner, cos_outer + 1e-4f), c);
  }

  bool sample_light(const V3 &p, float time, float ul, float u1, float u2, LightSample &ls) const {
    int n = (int)lights.size();
    if (n == 0) return false;
    float frac;
    int li = sample_cdf(light_cdf.data(), n, ul, frac);
    float psel = light_prob[li];
    if (psel <= 0) return false;
    const LightEntry &E = lights[li];
    if (E.type == 0) {
      const AnLight &L = S->lights[E.index];
      V3 col = V3(L.color[0], L.color[1], L.color[2]) * L.intensity;
      if (L.type == 2) {
        V3 to = -normalize(V3(L.direction[0], L.direction[1], L.direction[2]));
        if (L.angle > 1e-5f) {
          float cmax = std::cos(L.angle);
          Frame f(to);
          ls.wi = f.to_world(uniform_cone(u1, u2, cmax));
          float sa = 2 * PI * (1 - cmax);
          ls.pdf = psel / sa;
          ls.Le = col / (PI * sqr(std::sin(L.angle)));
          ls.delta = false;
        } else {
          ls.wi = to; ls.pdf = psel; ls.Le = col; ls.delta = true;
        }
        ls.dist = INF;
        return true;
      }
      V3 lp(L.position[0], L.position[1], L.position[2]);
      V3 d = lp - p;
      float d2 = dot(d, d);
      float dl = std::sqrt(d2);
      if (dl <= 1e-6f) return false;
      V3 to = d / dl;
      if (L.radius > 1e-5f && dl > L.radius * 1.0001f) {
        float sin2 = sqr(L.radius) / d2;
        float cmax = std::sqrt(std::max(0.f, 1 - sin2));
        Frame f(to);
        V3 wi = f.to_world(uniform_cone(u1, u2, cmax));
        // distance to sphere surface
        float b = dot(wi, d);
        float disc = b * b - (d2 - sqr(L.radius));
        float t = b - std::sqrt(std::max(0.f, disc));
        ls.wi = wi;
        ls.dist = std::max(0.f, t);
        ls.pdf = psel / (2 * PI * (1 - cmax));
        ls.Le = col / (PI * sqr(L.radius)) * spot_factor(L, -wi);
        ls.delta = false;
      } else {
        ls.wi = to;
        ls.dist = dl;
        ls.pdf = psel;
        ls.Le = col / d2 * spot_factor(L, -to);
        ls.delta = true;
      }
      return maxc(ls.Le) > 0;
    }
    if (E.type == 1) {
      int t = E.index;
      V3 a, b, c;
      tri_verts(t, time, a, b, c);
      float su = std::sqrt(u1);
      float b0 = 1 - su, b1 = u2 * su;
      V3 q = a * b0 + b * b1 + c * (1 - b0 - b1);
      V3 nrm = cross(b - a, c - a);
      float area = 0.5f * length(nrm);
      if (area <= 0) return false;
      nrm = normalize(nrm);
      V3 d = q - p;
      float d2 = dot(d, d);
      float dl = std::sqrt(d2);
      if (dl < 1e-6f) return false;
      ls.wi = d / dl;
      float cosl = dot(nrm, -ls.wi);
      const AnMaterial &m = mat_of(t);
      if (m.emission_one_sided && cosl <= 0) return false;
      cosl = std::fabs(cosl);
      if (cosl < 1e-6f) return false;
      ls.dist = dl;
      ls.pdf = psel * d2 / (cosl * area);
      V3 e = V3(m.emission[0], m.emission[1], m.emission[2]) * m.emission_strength;
      if (m.tex_emission >= 0) {
        float u0 = 0, v0 = 0, uu1 = 1, vv1 = 0, uu2 = 0, vv2 = 1;
        if (S->UV) { const float *uv = S->UV + (size_t)t * 6; u0 = uv[0]; v0 = uv[1]; uu1 = uv[2]; vv1 = uv[3]; uu2 = uv[4]; vv2 = uv[5]; }
        float tu = u0 * b0 + uu1 * b1 + uu2 * (1 - b0 - b1), tv = v0 * b0 + vv1 * b1 + vv2 * (1 - b0 - b1);
        e = e * eval_tex(m.tex_emission, tu, tv, q, q);
      }
      ls.Le = e;
      ls.delta = false;
      return maxc(e) > 0;
    }
    // environment
    float pdf;
    V3 d = env_sample(u1, u2, pdf);
    if (pdf <= 0) return false;
    ls.wi = d;
    ls.dist = INF;
    ls.pdf = psel * pdf;
    ls.Le = env_radiance(d);
    ls.delta = false;
    return maxc(ls.Le) > 0;
  }

  float pdf_light_tri(int tri, const V3 &p, const V3 &wi, float dist, float area, const V3 &ng) const {
    int e = tri_light[tri];
    if (e < 0) return 0;
    float cosl = std::fabs(dot(ng, wi));
    if (cosl < 1e-6f || area <= 0) return 0;
    return light_prob[e] * dist * dist / (cosl * area);
  }

  float pdf_light_env(const V3 &d) const {
    if (env_entry < 0) return 0;
    return light_prob[env_entry] * env_pdf(d);
  }

  // ------------------------------------------------------------- BSDF
  static inline float ggx_D(float cos_h, float a2) {
    float c2 = cos_h * cos_h;
    float d = c2 * (a2 - 1) + 1;
    return a2 / (PI * d * d);
  }
  static inline float ggx_lambda(float cos_t, float a2) {
    float c2 = cos_t * cos_t;
    if (c2 <= 0) return 1e10f;
    float t2 = std::max(0.f, (1 - c2) / c2);
    return 0.5f * (-1 + std::sqrt(1 + a2 * t2));
  }
  static inline float ggx_G1(float cos_t, float a2) { return 1.f / (1.f + ggx_lambda(std::fabs(cos_t), a2)); }
  static inline float ggx_G2(float co, float ci, float a2) {
    return 1.f / (1.f + ggx_lambda(std::fabs(co), a2) + ggx_lambda(std::fabs(ci), a2));
  }
  static V3 sample_vndf(const V3 &Ve, float a, float u1, float u2) {
    V3 Vh = normalize(V3(a * Ve.x, a * Ve.y, Ve.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    V3 T1 = lensq > 0 ? V3(-Vh.y, Vh.x, 0) * (1.f / std::sqrt(lensq)) : V3(1, 0, 0);
    V3 T2 = cross(Vh, T1);
    float r = std::sqrt(u1), phi = 2 * PI * u2;
    float t1 = r * std::cos(phi), t2 = r * std::sin(phi);
    float s = 0.5f * (1 + Vh.z);
    t2 = (1 - s) * std::sqrt(std::max(0.f, 1 - t1 * t1)) + s * t2;
    V3 Nh = T1 * t1 + T2 * t2 + Vh * std::sqrt(std::max(0.f, 1 - t1 * t1 - t2 * t2));
    return normalize(V3(a * Nh.x, a * Nh.y, std::max(1e-6f, Nh.z)));
  }
  static inline float schlick_w(float c) { float m = saturate(1 - c); float m2 = m * m; return m2 * m2 * m; }
  static inline V3 schlick(const V3 &F0, float c) { return F0 + (V3(1) - F0) * schlick_w(c); }
  static float fresnel_dielectric(float cos_i, float eta) {
    cos_i = saturate(cos_i);
    float sin2_t = eta * eta * (1 - cos_i * cos_i);
    if (sin2_t >= 1) return 1;
    float cos_t = std::sqrt(1 - sin2_t);
    float rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    float rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5f * (rs * rs + rp * rp);
  }
  static bool refract_dir(const V3 &wo, const V3 &h, float eta, V3 &wt) {
    float c = dot(wo, h);
    float k = 1 - eta * eta * (1 - c * c);
    if (k < 0) return false;
    wt = normalize(-wo * eta + h * (eta * c - std::sqrt(k)));
    return true;
  }

  struct BSDF {
    MatEval m;
    V3 n, ng;  // oriented to wo side
    Frame f;
    V3 F0, tint_spec;
    float a2, a, acc2, acc, at;
    float pd, ps, pc, pt, psss;
    float kd, kspec, kt, ksss;
    float eta;
    bool front;
    bool nonspecular() const { return pd + ps + pc > 0; }
  };

  void make_bsdf(BSDF &b, const MatEval &m, const Surface &s, const V3 &wo) const {
    b.m = m;
    b.front = s.front;
    b.ng = s.front ? s.ng : -s.ng;
    V3 n = s.front ? s.ns : -s.ns;
    // keep shading normal on the visible side
    float cn = dot(n, wo);
    if (cn < 0.02f) {
      n = normalize(n + b.ng * (0.02f - cn + 1e-3f) * 1.5f);
      if (dot(n, wo) <= 0) n = b.ng;
    }
    b.n = n;
    b.f = Frame(n, s.dpdu);
    float r = std::max(m.roughness, 0.02f);
    b.a = r * r; b.a2 = b.a * b.a;
    float rc = std::max(m.cc_rough, 0.02f);
    b.acc = rc * rc; b.acc2 = b.acc * b.acc;
    float rt = std::max(m.trans_rough, 0.f);
    b.at = rt * rt;
    b.eta = s.front ? 1.f / m.ior : m.ior;
    V3 tint = lum(m.base) > 0 ? m.base / lum(m.base) : V3(1);
    float f0d = sqr((m.ior - 1) / (m.ior + 1));  // dielectric F0 from IOR
    f0d *= m.specular * 2.f;                     // specular 0.5 -> physical F0
    V3 F0d = lerp(V3(1), tint, m.spec_tint) * f0d;
    b.F0 = lerp(F0d, m.base, m.metallic);
    float cos_o = saturate(dot(n, wo));
    float dielectric = 1 - m.metallic;
    b.kt = dielectric * m.transmission;
    b.kspec = 1 - b.kt;
    float diffuse_w = dielectric * (1 - m.transmission);
    float fo = schlick_w(cos_o);
    float spec_albedo = lum(b.F0 + (V3(1) - b.F0) * fo);
    b.kd = diffuse_w * (1 - m.subsurface);
    b.ksss = diffuse_w * m.subsurface;
    float wd = b.kd * std::max(0.05f, lum(m.base)) * (1 - f0d);
    float wsss = b.ksss * std::max(0.05f, lum(m.base));
    float ws = b.kspec > 0 ? b.kspec * std::max(spec_albedo, 0.04f) : 0;
    float wc = m.clearcoat > 0 ? m.clearcoat * 0.25f * std::max(0.04f + 0.96f * fo, 0.1f) : 0;
    float wt = b.kt;
    float sum = wd + wsss + ws + wc + wt;
    if (sum <= 0) { b.pd = b.ps = b.pc = b.pt = b.psss = 0; return; }
    b.pd = wd / sum; b.psss = wsss / sum; b.ps = ws / sum; b.pc = wc / sum; b.pt = wt / sum;
  }

  // Evaluate reflective lobes (diffuse, sheen, specular, clearcoat) -> f (without cos)
  V3 bsdf_eval(const BSDF &b, const V3 &wo, const V3 &wi, float &pdf) const {
    pdf = 0;
    float ci = dot(b.n, wi), co = dot(b.n, wo);
    if (ci <= 0 || co <= 0 || dot(b.ng, wi) <= 0) return V3(0);
    V3 h = normalize(wo + wi);
    float ch = saturate(dot(b.n, h)), cih = saturate(dot(wi, h));
    V3 f(0);
    float cc_att = 1.f;
    if (b.m.clearcoat > 0) {
      float Fc = 0.04f + 0.96f * schlick_w(cih);
      float D = ggx_D(ch, b.acc2), G = ggx_G2(co, ci, b.acc2);
      f += V3(b.m.clearcoat * Fc * D * G / (4 * co * ci));
      cc_att = 1 - b.m.clearcoat * (0.04f + 0.96f * schlick_w(co));
      pdf += b.pc * D * ggx_G1(co, b.acc2) / (4 * co);
    }
    if (b.kd > 0) {
      // Burley diffuse retro-reflection
      float fl = schlick_w(ci), fv = schlick_w(co);
      float rr = 2 * b.m.roughness * cih * cih;
      float fd = (1 - 0.5f * fl) * (1 - 0.5f * fv) + rr * (fl + fv + fl * fv * (rr - 1));
      V3 diff = b.m.base * (INV_PI * fd);
      if (b.m.sheen > 0) {
        V3 tint = lum(b.m.base) > 0 ? b.m.base / lum(b.m.base) : V3(1);
        V3 sc = lerp(V3(1), tint, b.m.sheen_tint);
        diff += sc * (b.m.sheen * schlick_w(cih) * INV_PI);
      }
      f += diff * (b.kd * cc_att);
      pdf += b.pd * ci * INV_PI;
    }
    if (b.kspec > 0) {
      V3 F = schlick(b.F0, cih);
      float D = ggx_D(ch, b.a2), G = ggx_G2(co, ci, b.a2);
      // energy compensation for multiple scattering (approximate, Kulla-Conty style)
      float ems = 1.f + b.m.roughness * b.m.roughness * 0.6f * lum(b.F0);
      f += F * (b.kspec * cc_att * ems * D * G / (4 * co * ci));
      pdf += b.ps * D * ggx_G1(co, b.a2) / (4 * co);
    }
    return f;
  }

  struct BSample {
    V3 wi, weight;
    float pdf;
    int kind;  // 0 reflect-nonspec, 1 specular/transmission (delta-like), 2 sss
    float lobe_rough = 1.f;  // roughness of the sampled lobe (path regularisation)
  };

  bool bsdf_sample(const BSDF &b, const V3 &wo, float ul, float u1, float u2, BSample &bs) const {
    float c = ul;
    if (c < b.pt) {
      // dielectric transmission lobe (incl. its own fresnel reflection)
      float uc = c / b.pt;
      V3 wol = b.f.to_local(wo);
      bool rough = b.at > 1e-4f;
      V3 hl = rough ? sample_vndf(wol, std::max(b.at, 1e-4f), u1, u2) : V3(0, 0, 1);
      V3 h = b.f.to_world(hl);
      float coh = dot(wo, h);
      if (coh <= 0) return false;
      float eta = b.m.thin ? 1.f / b.m.ior : b.eta;
      float F = fresnel_dielectric(coh, eta);
      if (b.m.thin) F = 2 * F / (1 + F);
      float a2t = b.at * b.at;
      if (uc < F) {
        bs.wi = reflect(-wo, h);
        if (dot(bs.wi, b.ng) <= 0 || dot(bs.wi, b.n) <= 0) return false;
        bs.weight = V3(rough ? ggx_G1(dot(b.n, bs.wi), a2t) : 1.f);
      } else {
        if (b.m.thin) {
          V3 r = reflect(-wo, h);
          bs.wi = normalize(r - b.n * (2 * dot(r, b.n)));
        } else if (!refract_dir(wo, h, eta, bs.wi)) {
          bs.wi = reflect(-wo, h);
        }
        if (dot(bs.wi, b.ng) >= 0) return false;
        bs.weight = b.m.base * (rough ? ggx_G1(dot(b.n, bs.wi), a2t) : 1.f);
      }
      bs.weight *= b.kt / b.pt;
      bs.pdf = 0;
      bs.kind = 1;
      bs.lobe_rough = b.m.trans_rough;
      return true;
    }
    c -= b.pt;
    if (c < b.psss) {
      bs.kind = 2;
      float fr = 1.f - fresnel_dielectric(dot(b.n, wo), 1.f / b.m.ior);
      bs.weight = V3(b.ksss * fr / b.psss);
      bs.wi = b.f.to_world(cosine_hemisphere(u1, u2));
      bs.wi = -bs.wi;  // into the surface
      bs.pdf = 0;
      return true;
    }
    c -= b.psss;
    V3 wi;
    if (c < b.pd) {
      wi = b.f.to_world(cosine_hemisphere(u1, u2));
      bs.lobe_rough = 1.f;
    } else if (c < b.pd + b.ps) {
      bs.lobe_rough = b.m.roughness;
      V3 wol = b.f.to_local(wo);
      V3 hl = sample_vndf(wol, b.a, u1, u2);
      wi = reflect(-wo, b.f.to_world(hl));
    } else {
      V3 wol = b.f.to_local(wo);
      V3 hl = sample_vndf(wol, b.acc, u1, u2);
      wi = reflect(-wo, b.f.to_world(hl));
      bs.lobe_rough = b.m.cc_rough;
    }
    float pdf;
    V3 f = bsdf_eval(b, wo, wi, pdf);
    if (pdf <= 0 || maxc(f) <= 0) return false;
    bs.wi = wi;
    bs.weight = f * (dot(b.n, wi) / pdf);
    bs.pdf = pdf;
    bs.kind = (b.m.roughness < 0.05f && c >= b.pd) ? 1 : 0;
    if (bs.kind == 1) bs.pdf = pdf;  // still valid for MIS
    bs.kind = 0;
    return true;
  }

  // ------------------------------------------------------------- media
  bool fog_on() const { return S->fog_density > 0; }
  // intersect ray with fog slab, clipped to [0,tmax]
  bool fog_segment(const Ray &r, float tmax, float &t0, float &t1) const {
    float y0 = S->fog_height_min, y1 = S->fog_height_max;
    if (y1 <= y0) { t0 = 0; t1 = tmax; return tmax > 0; }
    if (std::fabs(r.d.y) < 1e-8f) {
      if (r.o.y < y0 || r.o.y > y1) return false;
      t0 = 0; t1 = tmax; return true;
    }
    float ta = (y0 - r.o.y) / r.d.y, tb = (y1 - r.o.y) / r.d.y;
    if (ta > tb) std::swap(ta, tb);
    t0 = std::max(ta, 0.f); t1 = std::min(tb, tmax);
    return t1 > t0;
  }
  float fog_transmittance(const Ray &r, float tmax) const {
    if (!fog_on()) return 1.f;
    float t0, t1;
    if (!fog_segment(r, tmax, t0, t1)) return 1.f;
    float len = t1 - t0;
    if (len > 1e20f) return 0.f;
    return std::exp(-S->fog_density * len);
  }
  static float hg_eval(float cos_t, float g) {
    float d = 1 + g * g - 2 * g * cos_t;
    return (1 - g * g) / (4 * PI * d * std::sqrt(std::max(d, 1e-8f)));
  }
  static V3 hg_sample(const V3 &wo_forward, float g, float u1, float u2) {
    float cos_t;
    if (std::fabs(g) < 1e-3f) cos_t = 1 - 2 * u1;
    else {
      float s = (1 - g * g) / (1 - g + 2 * g * u1);
      cos_t = (1 + g * g - s * s) / (2 * g);
    }
    cos_t = clampf(cos_t, -1.f, 1.f);
    float sin_t = std::sqrt(std::max(0.f, 1 - cos_t * cos_t)), phi = 2 * PI * u2;
    Frame f(wo_forward);
    return f.to_world(V3(sin_t * std::cos(phi), sin_t * std::sin(phi), cos_t));
  }


  // ------------------------------------------------------------- heterogeneous volumes
  bool vol_on() const { return S->num_volumes > 0 && S->volumes; }
  static V3 vxf(const float *m, const V3 &p) {
    return V3(m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
              m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
  }
  static V3 vxd(const float *m, const V3 &d) {
    return V3(m[0] * d.x + m[1] * d.y + m[2] * d.z, m[4] * d.x + m[5] * d.y + m[6] * d.z,
              m[8] * d.x + m[9] * d.y + m[10] * d.z);
  }
  // ray interval inside volume box (world t)
  bool vol_interval(const AnVolume &v, const Ray &r, float tmax, float &t0, float &t1) const {
    V3 o = vxf(v.world_to_local, r.o), d = vxd(v.world_to_local, r.d);
    t0 = 0; t1 = tmax;
    for (int a = 0; a < 3; ++a) {
      float oa = o[a], da = d[a];
      if (std::fabs(da) < 1e-12f) { if (oa < 0 || oa > 1) return false; continue; }
      float inv = 1.f / da, ta = (0 - oa) * inv, tb = (1 - oa) * inv;
      if (ta > tb) std::swap(ta, tb);
      t0 = std::max(t0, ta); t1 = std::min(t1, tb);
      if (t1 <= t0) return false;
    }
    return true;
  }
  float vol_grid(const AnVolume &v, int off, const V3 &q) const {
    const float *g = S->voldata + off;
    float x = q.x * v.nx - 0.5f, y = q.y * v.ny - 0.5f, z = q.z * v.nz - 0.5f;
    x = clampf(x, 0.f, (float)(v.nx - 1)); y = clampf(y, 0.f, (float)(v.ny - 1)); z = clampf(z, 0.f, (float)(v.nz - 1));
    int x0 = (int)x, y0 = (int)y, z0 = (int)z;
    int x1 = std::min(x0 + 1, v.nx - 1), y1 = std::min(y0 + 1, v.ny - 1), z1 = std::min(z0 + 1, v.nz - 1);
    float fx = x - x0, fy = y - y0, fz = z - z0;
    auto at = [&](int i, int j, int k) { return g[((size_t)k * v.ny + j) * v.nx + i]; };
    float c00 = at(x0, y0, z0) * (1 - fx) + at(x1, y0, z0) * fx;
    float c10 = at(x0, y1, z0) * (1 - fx) + at(x1, y1, z0) * fx;
    float c01 = at(x0, y0, z1) * (1 - fx) + at(x1, y0, z1) * fx;
    float c11 = at(x0, y1, z1) * (1 - fx) + at(x1, y1, z1) * fx;
    return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
  }
  // normalized density (0..1 for procedural, raw grid value for grids) and heat at local point q
  void vol_sample(const AnVolume &v, const V3 &q, float &dn, float &heat) const {
    heat = 0;
    if (v.type == 0) {
      dn = std::max(0.f, vol_grid(v, v.density_offset, q));
      if (v.heat_offset >= 0) heat = std::max(0.f, vol_grid(v, v.heat_offset, q));
      else heat = dn * v.heat_from_density;
      return;
    }
    V3 c = q * 2.f - V3(1);
    float edge;
    if (v.shape == 1) edge = 1 - length(c);
    else if (v.shape == 2) edge = std::min(1 - std::sqrt(c.x * c.x + c.z * c.z), 1 - std::fabs(c.y));
    else edge = 1 - std::max(std::fabs(c.x), std::max(std::fabs(c.y), std::fabs(c.z)));
    if (edge <= 0) { dn = 0; return; }
    float base = v.falloff > 1e-6f ? smoothstepf(0.f, v.falloff, edge) : 1.f;
    float d = base;
    if (v.octaves > 0) {
      // edge-biased noise carving: the core stays dense, the boundary breaks up into billows
      V3 pq = q * v.frequency + V3(v.offset[0], v.offset[1], v.offset[2]);
      float n = fbm(pq, v.octaves, v.gain, v.lacunarity);
      d = (base - v.threshold) + n * v.contrast;
      d = std::min(d, base * 4.f);
    }
    dn = clampf(d, 0.f, 1.f);
    heat = dn * v.heat_from_density;
  }
  V3 vol_blackbody(const AnVolume &v, float heat) const {
    if (v.bb_size <= 0 || heat <= 0) return V3(0);
    const float *t = S->voldata + v.bb_offset;
    float x = clampf(heat, 0.f, 1.f) * (v.bb_size - 1);
    int i = std::min((int)x, v.bb_size - 2); float f = x - i;
    if (v.bb_size == 1) return V3(t[0], t[1], t[2]);
    return V3(t[3 * i] * (1 - f) + t[3 * i + 3] * f, t[3 * i + 1] * (1 - f) + t[3 * i + 4] * f,
              t[3 * i + 2] * (1 - f) + t[3 * i + 5] * f);
  }
  // combined medium properties at world point p
  struct VolPt { float sigma_t; V3 sigma_s; V3 Le; float g; };
  VolPt vol_eval(const V3 &p) const {
    VolPt r{0, V3(0), V3(0), 0};
    float gw = 0;
    for (int i = 0; i < S->num_volumes; ++i) {
      const AnVolume &v = S->volumes[i];
      V3 q = vxf(v.world_to_local, p);
      if (q.x < 0 || q.y < 0 || q.z < 0 || q.x > 1 || q.y > 1 || q.z > 1) continue;
      float dn, heat;
      vol_sample(v, q, dn, heat);
      float st = dn * v.density;
      if (st > 0) {
        V3 alb(v.albedo[0], v.albedo[1], v.albedo[2]);
        r.sigma_t += st;
        r.sigma_s += alb * st;
        float w = st * (alb.x + alb.y + alb.z);
        r.g += v.anisotropy * w; gw += w;
      }
      if (v.emission_strength > 0)
        r.Le += V3(v.emission[0], v.emission[1], v.emission[2]) * (v.emission_strength * dn);
      if (v.blackbody_strength > 0 && heat > 0) r.Le += vol_blackbody(v, heat) * v.blackbody_strength;
    }
    if (gw > 0) r.g /= gw;
    return r;
  }
  // union of volume intervals along the ray and the summed majorant over them
  bool vol_range(const Ray &r, float tmax, float &t0, float &t1, float &smaj) const {
    t0 = INF; t1 = 0; smaj = 0;
    for (int i = 0; i < S->num_volumes; ++i) {
      const AnVolume &v = S->volumes[i];
      float a, b;
      if (!vol_interval(v, r, tmax, a, b)) continue;
      t0 = std::min(t0, a); t1 = std::max(t1, b);
      smaj += std::max(v.majorant, 1e-6f);
    }
    return t1 > t0 && smaj > 0;
  }
  // ratio-tracking transmittance (random stream hashed from the ray, so shadow() stays const)
  float vol_transmittance(const Ray &r, float tmax) const {
    if (!vol_on()) return 1.f;
    float t0, t1, smaj;
    if (!vol_range(r, tmax, t0, t1, smaj)) return 1.f;
    uint32_t h = hash_combine(hash_combine(hash_combine(float_bits(r.o.x), float_bits(r.o.y)),
                                           hash_combine(float_bits(r.o.z), float_bits(r.d.x))),
                              hash_combine(float_bits(r.d.y), float_bits(r.d.z)));
    PCG rng((uint64_t)h * 2654435761ULL + 17);
    float T = 1.f, t = t0;
    for (int it = 0; it < 4096; ++it) {
      t -= std::log(1 - rng.uniform()) / smaj;
      if (t >= t1) break;
      VolPt vp = vol_eval(r.o + r.d * t);
      T *= std::max(0.f, 1 - vp.sigma_t / smaj);
      if (T < 0.05f) {  // russian roulette on low transmittance
        if (rng.uniform() < 0.5f) return 0.f;
        T *= 2;
      }
    }
    return T;
  }

  // Visibility toward light with transparent shadows. Returns transmittance colour.
  V3 shadow(const V3 &p, const V3 &wi, float dist, float time, bool &through_transmission) const {
    through_transmission = false;
    Ray r{p, wi, time};
    float tmax = dist < INF ? dist * (1 - 1e-4f) : INF;
    int occ = occluded_any(r, tmax);
    V3 T(1);
    if (occ == 1) return V3(0);
    if (occ == 2) {
      V3 o = p;
      float remaining = tmax;
      for (int iter = 0; iter < 32; ++iter) {
        Ray rr{o, wi, time};
        Hit h;
        if (!intersect(rr, remaining, h)) break;
        Surface s = surface(h, rr);
        if (tri_opaque[h.tri] == 1) return V3(0);
        if (tri_opaque[h.tri] == 2) {
          o = offset_ray(s.p, dot(wi, s.ng) > 0 ? s.ng : -s.ng);
          remaining = remaining < INF ? remaining - h.t : INF;
          if (remaining <= 0) break;
          continue;
        }
        MatEval m = eval_material(s);
        float transmit = 1.f - m.alpha;
        V3 tr(transmit);
        if (m.alpha > 0 && m.transmission > 0 && !ST->caustics) {
          float k = (1 - m.metallic) * m.transmission;
          float F = fresnel_dielectric(std::fabs(dot(s.ns, wi)), s.front ? 1.f / m.ior : m.ior);
          V3 glass = m.base * (k * (1 - F));
          if (!s.front && m.has_atten) glass *= vexp(-m.atten_sigma * h.t);
          tr = tr + glass * m.alpha;
          through_transmission = true;
        }
        T *= tr;
        if (maxc(T) < 1e-4f) return V3(0);
        o = offset_ray(s.p, dot(wi, s.ng) > 0 ? s.ng : -s.ng);
        remaining = remaining < INF ? remaining - h.t : INF;
        if (remaining <= 0) break;
      }
    }
    T *= fog_transmittance(r, dist);
    if (vol_on()) T *= vol_transmittance(r, tmax);
    return T;
  }

  // ------------------------------------------------------------- sphere lights
  bool hit_sphere_light(const Ray &r, float tmax, int &which, float &th) const {
    which = -1; th = tmax;
    for (int i = 0; i < S->num_lights; ++i) {
      const AnLight &L = S->lights[i];
      if (L.type == 2 || L.radius <= 1e-5f || L.intensity <= 0) continue;
      V3 oc = r.o - V3(L.position[0], L.position[1], L.position[2]);
      float b = dot(oc, r.d);
      float c = dot(oc, oc) - L.radius * L.radius;
      float disc = b * b - c;
      if (disc < 0) continue;
      float sq = std::sqrt(disc);
      float t = -b - sq;
      if (t <= 1e-5f) continue;  // ignore if inside
      if (t < th) { th = t; which = i; }
    }
    return which >= 0;
  }
  int analytic_entry(int light_index) const {
    for (size_t i = 0; i < lights.size(); ++i)
      if (lights[i].type == 0 && lights[i].index == light_index) return (int)i;
    return -1;
  }

  // ------------------------------------------------------------- integrators
  struct Aux { V3 albedo, normal; float depth; bool set; float alpha; };

  // Next event estimation at a surface (bsdf) or medium point (phase g when bsdf==nullptr)
  V3 nee(const V3 &p, const V3 &offset_n, const V3 &wo, float time, const BSDF *b, float g, Sampler &smp, bool lambert = false, const V3 &lam_n = V3()) const {
    float ul = smp.get1();
    float u1, u2; smp.get2(u1, u2);
    LightSample ls;
    if (!sample_light(p, time, ul, u1, u2, ls)) return V3(0);
    V3 f; float bpdf;
    if (b) {
      f = bsdf_eval(*b, wo, ls.wi, bpdf);
      f *= std::max(0.f, dot(b->n, ls.wi));
    } else if (lambert) {
      float c = dot(lam_n, ls.wi);
      if (c <= 0) return V3(0);
      f = V3(c * INV_PI); bpdf = c * INV_PI;
    } else {
      float ph = hg_eval(dot(-wo, ls.wi), g);
      f = V3(ph); bpdf = ph;
    }
    if (maxc(f) <= 0) return V3(0);
    V3 org = (b || lambert) ? offset_ray(p, dot(ls.wi, offset_n) > 0 ? offset_n : -offset_n) : p;
    bool through = false;
    V3 T = shadow(org, ls.wi, ls.dist, time, through);
    if (maxc(T) <= 0) return V3(0);
    float w = (ls.delta || through) ? 1.f : power_heuristic(ls.pdf, bpdf);
    return f * ls.Le * T * (w / ls.pdf);
  }

  V3 sss_alpha(const V3 &A) const {
    V3 r;
    for (int i = 0; i < 3; ++i) {
      float a = clampf(A[i], 0.f, 0.999f);
      r[i] = 1 - std::exp(a * (-5.09406f + a * (2.61188f - a * 4.31805f)));
    }
    return r;
  }

  V3 Li(Ray ray, Sampler &smp, PCG &rng, Aux &aux) const {
    V3 L(0), beta(1);
    bool specular = true, had_nonspec = false, trans_chain = false;
    float path_rough = 0.f;  // max lobe roughness along the path so far
    float prev_pdf = 0;
    int depth = 0, skips = 0;
    const int maxd = std::max(0, ST->max_depth);
    const int integ = ST->integrator;
    float clampv = ST->clamp_indirect;
    aux.set = false; aux.alpha = 0; aux.depth = 0;
    auto add = [&](const V3 &c) {
      V3 v = c;
      if (!isfinite3(v)) return;
      if (clampv > 0 && depth > 0) {
        float m = maxc(v);
        if (m > clampv) v = v * (clampv / m);
      }
      L += v;
    };
    for (;;) {
      Hit h;
      bool hit = intersect(ray, INF, h);
      float tmax = hit ? h.t : INF;
      // sphere lights
      int sl; float slt;
      bool lhit = hit_sphere_light(ray, tmax, sl, slt);
      if (lhit) tmax = slt;
      // fog scattering
      if (fog_on() && integ == INT_PATH) {
        float t0, t1;
        if (fog_segment(ray, tmax, t0, t1)) {
          float u = smp.get1();
          float ts = t0 - std::log(1 - u) / S->fog_density;
          if (ts < t1) {
            V3 p = ray.o + ray.d * ts;
            V3 alb(S->fog_color[0], S->fog_color[1], S->fog_color[2]);
            beta *= alb;
            if (depth == 0 && !aux.set) { aux.albedo = alb; aux.normal = V3(0); aux.depth = ts; aux.set = true; aux.alpha = 1; }
            if (depth >= maxd) break;
            add(beta * nee(p, V3(0), -ray.d, ray.time, nullptr, S->fog_anisotropy, smp));
            float u1, u2; smp.get2(u1, u2);
            V3 nd = hg_sample(ray.d, S->fog_anisotropy, u1, u2);
            prev_pdf = hg_eval(dot(ray.d, nd), S->fog_anisotropy);
            specular = false; had_nonspec = true; trans_chain = false; path_rough = 1.f;
            ray = Ray{p, nd, ray.time};
            depth++;
            if (depth >= std::max(3, ST->rr_depth)) {
              float q = std::max(0.05f, 1 - maxc(beta));
              if (rng.uniform() < q) break;
              beta = beta / (1 - q);
            }
            continue;
          }
        }
      }
      // heterogeneous volumes: delta tracking with null collisions over the summed majorant
      if (vol_on() && (integ == INT_PATH || integ == INT_DIRECT)) {
        float t0, t1, smaj;
        if (vol_range(ray, tmax, t0, t1, smaj)) {
          float t = t0;
          bool scattered = false, absorbed = false;
          V3 sp; float sg = 0;
          for (int it = 0; it < 100000; ++it) {
            t -= std::log(1 - rng.uniform()) / smaj;
            if (t >= t1) break;
            V3 p = ray.o + ray.d * t;
            VolPt vp = vol_eval(p);
            if (maxc(vp.Le) > 0 && !(integ == INT_DIRECT && depth > 1)) add(beta * vp.Le / smaj);
            if (vp.sigma_t <= 0) continue;
            float ss = (vp.sigma_s.x + vp.sigma_s.y + vp.sigma_s.z) / 3.f;
            float sa = std::max(0.f, vp.sigma_t - ss);
            float u = rng.uniform() * smaj;
            if (u < sa) { absorbed = true; sp = p; break; }
            if (u < sa + ss) {
              beta *= vp.sigma_s / std::max(ss, 1e-12f);
              scattered = true; sp = p; sg = vp.g; break;
            }
          }
          if (absorbed || scattered) {
            if (depth == 0 && !aux.set) {
              aux.albedo = scattered ? beta : V3(0); aux.normal = V3(0); aux.depth = t; aux.set = true;
            }
            if (depth == 0) aux.alpha = 1;
          }
          if (absorbed) break;
          if (scattered) {
            if (depth >= maxd) break;
            add(beta * nee(sp, V3(0), -ray.d, ray.time, nullptr, sg, smp));
            float u1, u2; smp.get2(u1, u2);
            V3 nd = hg_sample(ray.d, sg, u1, u2);
            prev_pdf = hg_eval(dot(ray.d, nd), sg);
            specular = false; had_nonspec = true; trans_chain = false; path_rough = 1.f;
            ray = Ray{sp, nd, ray.time};
            depth++;
            if (depth >= std::max(3, ST->rr_depth)) {
              float q = std::max(0.05f, 1 - maxc(beta));
              if (rng.uniform() < q) break;
              beta = beta / (1 - q);
            }
            continue;
          }
        }
      }
      if (lhit) {
        const AnLight &Lt = S->lights[sl];
        if (depth > 0 || Lt.visible) {
          V3 col = V3(Lt.color[0], Lt.color[1], Lt.color[2]) * Lt.intensity / (PI * sqr(Lt.radius));
          V3 lp(Lt.position[0], Lt.position[1], Lt.position[2]);
          V3 hp = ray.o + ray.d * slt;
          col = col * spot_factor(Lt, normalize(hp - lp) * -1.f * -1.f) ;
          col = col * spot_factor(Lt, ray.d) / std::max(1e-6f, spot_factor(Lt, normalize(hp - lp)));
          // spot factor uses direction from light toward the shading point (i.e. -ray.d)
          col = V3(Lt.color[0], Lt.color[1], Lt.color[2]) * Lt.intensity / (PI * sqr(Lt.radius)) * spot_factor(Lt, -ray.d);
          float w = 1;
          if (!specular && depth > 0) {
            int e = analytic_entry(sl);
            V3 d = lp - ray.o;
            float d2 = dot(d, d);
            float cmax = std::sqrt(std::max(0.f, 1 - sqr(Lt.radius) / d2));
            float lpdf = e >= 0 ? light_prob[e] / (2 * PI * (1 - cmax)) : 0;
            w = power_heuristic(prev_pdf, lpdf);
          }
          if (!(trans_chain && !ST->caustics)) add(beta * col * w);
          if (depth == 0) { aux.alpha = 1; if (!aux.set) { aux.albedo = V3(1); aux.normal = -ray.d; aux.depth = slt; aux.set = true; } }
        }
        break;
      }
      if (!hit) {
        V3 Le = env_radiance(ray.d);
        bool camera_ray = (depth == 0) || (specular && !had_nonspec);
        if (camera_ray && !S->env_visible) {
          if (depth == 0) Le = V3(S->env_bg[0], S->env_bg[1], S->env_bg[2]);
          else Le = V3(S->env_bg[0], S->env_bg[1], S->env_bg[2]);
        }
        float w = 1;
        if (!specular && depth > 0) w = power_heuristic(prev_pdf, pdf_light_env(ray.d));
        if (integ == INT_DIRECT && depth > 1) Le = V3(0);
        // visible sun discs
        for (int i = 0; i < S->num_lights; ++i) {
          const AnLight &Lt = S->lights[i];
          if (Lt.type != 2 || Lt.angle <= 1e-5f || Lt.intensity <= 0) continue;
          if (depth == 0 && !Lt.visible) continue;
          V3 to = -normalize(V3(Lt.direction[0], Lt.direction[1], Lt.direction[2]));
          float cmax = std::cos(Lt.angle);
          if (dot(ray.d, to) >= cmax) {
            V3 col = V3(Lt.color[0], Lt.color[1], Lt.color[2]) * Lt.intensity / (PI * sqr(std::sin(Lt.angle)));
            float wl = 1;
            if (!specular && depth > 0) {
              int e = analytic_entry(i);
              float lpdf = e >= 0 ? light_prob[e] / (2 * PI * (1 - cmax)) : 0;
              wl = power_heuristic(prev_pdf, lpdf);
            }
            if (!(trans_chain && !ST->caustics)) add(beta * col * wl);
          }
        }
        if (!(trans_chain && !ST->caustics)) add(beta * Le * w);
        if (depth == 0) {
          aux.alpha = 0;
          if (!aux.set) { V3 a = env_radiance(ray.d); aux.albedo = V3(saturate(a.x), saturate(a.y), saturate(a.z)); aux.normal = V3(0); aux.depth = 0; aux.set = true; }
        } else if (!aux.set) {
          V3 a = env_radiance(ray.d); aux.albedo = V3(saturate(a.x), saturate(a.y), saturate(a.z)); aux.normal = V3(0); aux.set = true;
        }
        break;
      }

      Surface s = surface(h, ray);
      perturb_normal(s);
      MatEval m = eval_material(s);
      if (integ == INT_CLAY || integ == INT_WIREFRAME) {
        V3 em = m.emission;
        float al = m.alpha;
        std::memset(&m, 0, sizeof(m));
        m.base = V3(0.72f); m.roughness = 0.6f; m.ior = 1.5f; m.specular = 0.3f; m.alpha = al; m.emission = em;
        m.trans_rough = 0.6f;
      }
      // Path regularisation ("filter glossy"): after a rough bounce, widen sharp glossy lobes so
      // diffuse -> mirror -> small-light paths become low-variance instead of firefly caustics.
      if (!ST->caustics && path_rough > 0.f) {
        float k = 0.5f * path_rough;
        m.roughness = std::max(m.roughness, k);
        m.cc_rough = std::max(m.cc_rough, k);
        m.trans_rough = std::max(m.trans_rough, k);
      }
      if (depth == 0) {
        aux.alpha = 1;
        aux.depth = h.t;
      }
      // camera-invisible surfaces (material flags bit 0), e.g. area-light cards
      if (depth == 0 && s.mat >= 0 && s.mat < S->num_materials && (S->materials[s.mat].flags & 1)) {
        ray.o = offset_ray(s.p, s.front ? -s.ng : s.ng);
        if (++skips > 64) break;
        aux.alpha = 0;
        continue;
      }
      // alpha cut-out / stochastic transparency
      if (m.alpha < 1.f) {
        if (rng.uniform() >= m.alpha) {
          ray.o = offset_ray(s.p, s.front ? -s.ng : s.ng);
          if (++skips > 64) break;
          if (depth == 0) aux.alpha = 0;
          continue;
        }
      }
      // Beer-Lambert absorption inside dielectrics
      if (!s.front && m.has_atten) beta *= vexp(-m.atten_sigma * h.t);

      // AOVs
      bool spec_surface = (m.transmission > 0.5f && m.trans_rough < 0.1f) || (m.metallic > 0.5f && m.roughness < 0.1f);
      if (!aux.set && (!spec_surface || depth >= 3)) {
        aux.albedo = (m.transmission > 0.5f) ? V3(1) : m.base;
        aux.normal = s.front ? s.ns : -s.ns;
        aux.set = true;
      }

      // debug / utility integrators
      if (integ == INT_NORMALS) { V3 n = s.front ? s.ns : -s.ns; L = (n + V3(1)) * 0.5f; break; }
      if (integ == INT_ALBEDO) { L = m.base; break; }
      if (integ == INT_DEPTH) { L = V3(h.t); break; }
      if (integ == INT_UV) { L = V3(s.u - std::floor(s.u), s.v - std::floor(s.v), 0); break; }
      if (integ == INT_AO) {
        V3 n = s.front ? s.ns : -s.ns;
        V3 ngo = s.front ? s.ng : -s.ng;
        float u1, u2; smp.get2(u1, u2);
        Frame f(n);
        V3 d = f.to_world(cosine_hemisphere(u1, u2));
        if (dot(d, ngo) <= 0) { L = V3(0); break; }
        Ray r{offset_ray(s.p, ngo), d, ray.time};
        float dist = ST->ao_distance > 0 ? ST->ao_distance : scene_radius;
        L = occluded_any(r, dist) == 1 ? V3(0) : V3(1);
        break;
      }

      // emission
      if (maxc(m.emission) > 0 && (!m.one_sided || s.front)) {
        float w = 1;
        if (!specular && depth > 0) {
          float lp = pdf_light_tri(s.tri, ray.o, ray.d, h.t, s.area, s.ng);
          w = power_heuristic(prev_pdf, lp);
        }
        if (!(trans_chain && !ST->caustics)) add(beta * m.emission * w);
      }
      // wireframe overlay
      if (integ == INT_WIREFRAME && depth == 0) {
        V3 a, b, c;
        tri_verts(s.tri, ray.time, a, b, c);
        float w0 = 1 - h.u - h.v;
        float ea = length(c - b), eb = length(a - c), ec = length(b - a);
        float area2 = 2 * s.area;
        float d0 = w0 * area2 / std::max(1e-9f, ea), d1 = h.u * area2 / std::max(1e-9f, eb), d2 = h.v * area2 / std::max(1e-9f, ec);
        float dmin = std::min(d0, std::min(d1, d2));
        float px = h.t * 2 * std::tan(S->camera.fov_y * 0.5f) / ST->height;
        if (dmin < ST->wire_width * px) { L = V3(0.02f); break; }
      }
      if (depth >= maxd) break;

      V3 wo = -ray.d;
      BSDF b;
      make_bsdf(b, m, s, wo);

      // shadow catcher: only record shadowing in alpha-like fashion (approximation)
      if (m.shadow_catcher && depth == 0) {
        // compute unshadowed vs shadowed direct light ratio
        float ul = smp.get1(); float u1, u2; smp.get2(u1, u2);
        LightSample ls;
        float vis = 1;
        if (sample_light(s.p, ray.time, ul, u1, u2, ls)) {
          bool through;
          V3 T = shadow(offset_ray(s.p, b.ng), ls.wi, ls.dist, ray.time, through);
          vis = lum(T);
          if (dot(ls.wi, b.n) <= 0) vis = 1;
        }
        aux.alpha = 1 - vis;
        L = V3(0);
        break;
      }

      // direct light
      if (b.nonspecular()) {
        add(beta * nee(s.p, b.ng, wo, ray.time, &b, 0, smp));
      }
      // sample continuation
      float ul = smp.get1();
      float u1, u2; smp.get2(u1, u2);
      BSample bs;
      if (!bsdf_sample(b, wo, ul, u1, u2, bs)) break;
      if (bs.kind == 2) {
        // random walk subsurface scattering
        V3 sigma_t;
        for (int k = 0; k < 3; ++k) sigma_t[k] = 1.f / std::max(1e-5f, m.sss_radius[k]);
        V3 alpha = sss_alpha(m.base);
        V3 sigma_s = alpha * sigma_t;
        beta *= bs.weight;
        V3 pos = offset_ray(s.p, -b.ng);
        V3 dir = bs.wi;
        bool exited = false;
        Surface es;
        for (int step = 0; step < 256; ++step) {
          int ch = std::min(2, (int)(rng.uniform() * 3));
          float tdist = -std::log(1 - rng.uniform()) / sigma_t[ch];
          Ray wr{pos, dir, ray.time};
          Hit wh;
          bool whit = intersect(wr, tdist, wh);
          if (whit) {
            float tt = wh.t;
            V3 Tr = vexp(-sigma_t * tt);
            float pdfc = (Tr.x + Tr.y + Tr.z) / 3.f;
            if (pdfc <= 0) { beta = V3(0); break; }
            beta *= Tr / pdfc;
            es = surface(wh, wr);
            exited = true;
            break;
          }
          V3 Tr = vexp(-sigma_t * tdist);
          V3 pdfv = sigma_t * Tr;
          float pdfc = (pdfv.x + pdfv.y + pdfv.z) / 3.f;
          if (pdfc <= 0) { beta = V3(0); break; }
          beta *= sigma_s * Tr / pdfc;
          pos = pos + dir * tdist;
          dir = uniform_sphere(rng.uniform(), rng.uniform());
          if (step > 8) {
            float q = std::max(0.02f, 1 - maxc(beta));
            if (rng.uniform() < q) { beta = V3(0); break; }
            beta = beta / (1 - q);
          }
        }
        if (!exited || maxc(beta) <= 0) break;
        // exit through boundary: Lambertian transmission out of surface
        V3 nout = dot(es.ng, dir) > 0 ? es.ng : -es.ng;
        V3 nsh = dot(es.ns, nout) > 0 ? es.ns : -es.ns;
        depth++;
        add(beta * nee(es.p, nout, V3(0), ray.time, nullptr, 0, smp, true, nsh));
        float v1, v2; smp.get2(v1, v2);
        Frame ef(nsh);
        V3 nd = ef.to_world(cosine_hemisphere(v1, v2));
        if (dot(nd, nout) <= 0) break;
        prev_pdf = dot(nsh, nd) * INV_PI;
        specular = false; had_nonspec = true; trans_chain = false; path_rough = 1.f;
        ray = Ray{offset_ray(es.p, nout), nd, ray.time};
        if (depth >= maxd) break;
        continue;
      }
      if (!isfinite3(bs.weight) || maxc(bs.weight) <= 0) break;
      beta *= bs.weight;
      path_rough = std::max(path_rough, bs.kind == 2 ? 1.f : bs.lobe_rough);
      if (bs.kind == 1) {
        specular = true;
        if (had_nonspec) trans_chain = true;
      } else {
        specular = false; had_nonspec = true; trans_chain = false;
      }
      prev_pdf = bs.pdf;
      depth++;
      if (integ == INT_DIRECT && depth >= 1 && bs.kind != 1) {
        // direct-lighting integrator: follow only specular chains
        if (!specular) {
          // allow emission hit with MIS but nothing else
        }
      }
      if (depth >= std::max(1, ST->rr_depth)) {
        float q = std::max(0.05f, 1 - maxc(beta));
        if (rng.uniform() < q) break;
        beta = beta / (1 - q);
      }
      ray = Ray{offset_ray(s.p, dot(bs.wi, b.ng) > 0 ? b.ng : -b.ng), bs.wi, ray.time};
    }
    if (!aux.set) { aux.albedo = V3(0); aux.normal = V3(0); }
    return L;
  }

  // ------------------------------------------------------------- camera
  Ray camera_ray(float px, float py, float lu1, float lu2, float time) const {
    const AnCamera &C = S->camera;
    int W = ST->width, H = ST->height;
    float aspect = (float)W / (float)H;
    V3 pos = lerp(V3(C.pos0[0], C.pos0[1], C.pos0[2]), V3(C.pos1[0], C.pos1[1], C.pos1[2]), time);
    V3 fwd = normalize(lerp(V3(C.fwd0[0], C.fwd0[1], C.fwd0[2]), V3(C.fwd1[0], C.fwd1[1], C.fwd1[2]), time));
    V3 up = normalize(lerp(V3(C.up0[0], C.up0[1], C.up0[2]), V3(C.up1[0], C.up1[1], C.up1[2]), time));
    V3 right = normalize(cross(fwd, up));
    up = cross(right, fwd);
    float sx = (px / W) * 2 - 1;
    float sy = 1 - (py / H) * 2;
    Ray r;
    r.time = time;
    if (C.projection == 1) {
      float hh = C.ortho_height * 0.5f;
      r.o = pos + right * (sx * hh * aspect) + up * (sy * hh);
      r.d = fwd;
      return r;
    }
    if (C.projection == 2) {
      float phi = sx * PI, theta = sy * PI * 0.5f;
      V3 d = fwd * (std::cos(theta) * std::cos(phi)) + right * (std::cos(theta) * std::sin(phi)) + up * std::sin(theta);
      r.o = pos; r.d = normalize(d);
      return r;
    }
    if (C.projection == 3) {
      float rr = std::sqrt(sx * sx * aspect * aspect + sy * sy);
      float fov = C.fov_y > 0 ? C.fov_y : PI;
      float theta = rr * fov * 0.5f;
      float phi = std::atan2(sy, sx * aspect);
      V3 d = fwd * std::cos(theta) + (right * std::cos(phi) + up * std::sin(phi)) * std::sin(theta);
      r.o = pos; r.d = normalize(d);
      return r;
    }
    float th = std::tan(C.fov_y * 0.5f);
    V3 dcam(sx * th * aspect + C.shift_x * 2 * th, sy * th + C.shift_y * 2 * th, 1);
    V3 d = normalize(right * dcam.x + up * dcam.y + fwd * dcam.z);
    r.o = pos; r.d = d;
    if (C.aperture_radius > 0 && C.focus_distance > 0) {
      float lx, ly;
      if (C.blades >= 3) {
        // sample a regular polygon
        int nb = C.blades;
        float seg = std::min((float)nb - 1, std::floor(lu1 * nb));
        float uu = lu1 * nb - seg;
        float a0 = C.blade_rotation + seg * 2 * PI / nb, a1 = C.blade_rotation + (seg + 1) * 2 * PI / nb;
        float su = std::sqrt(uu);
        float b0 = 1 - su, b1 = lu2 * su;
        lx = std::cos(a0) * b0 + std::cos(a1) * b1; // barycentric in triangle (centre, v0, v1)
        ly = std::sin(a0) * b0 + std::sin(a1) * b1;
        float bc = 1 - b0 - b1;
        (void)bc;
        // triangle with vertices (0,0),(cos a0, sin a0),(cos a1, sin a1): p = b0*v0 + b1*v1 when centre weight is 1-b0-b1
      } else {
        concentric_disk(lu1, lu2, lx, ly);
      }
      float ft = C.focus_distance / std::max(1e-6f, dot(d, fwd));
      V3 focus = pos + d * ft;
      r.o = pos + right * (lx * C.aperture_radius) + up * (ly * C.aperture_radius);
      r.d = normalize(focus - r.o);
    }
    if (C.near_clip > 0) r.o = r.o + r.d * (C.near_clip / std::max(1e-6f, dot(r.d, fwd)));
    return r;
  }
};

// ============================================================================
// Rendering driver
// ============================================================================
struct Accum {
  std::vector<float> rgb, lum2, alpha, albedo, normal, depth;
  std::vector<int32_t> count;
};

static inline float gauss_sample(float u, float sigma) {
  // inverse-CDF approximation via Box-Muller using u and a fixed partner is not stratified;
  // use logistic approximation of the normal quantile (good to ~1%)
  u = clampf(u, 1e-6f, 1 - 1e-6f);
  return sigma * 0.5513288954217921f * std::log(u / (1 - u));  // sqrt(3)/pi
}

AN_EXPORT const char *an_version() { return "animora-native 1.0.0"; }

AN_EXPORT int an_struct_sizes(int32_t *out) {
  out[0] = (int32_t)sizeof(AnTexture);
  out[1] = (int32_t)sizeof(AnMaterial);
  out[2] = (int32_t)sizeof(AnLight);
  out[3] = (int32_t)sizeof(AnCamera);
  out[4] = (int32_t)sizeof(AnScene);
  out[5] = (int32_t)sizeof(AnSettings);
  out[6] = (int32_t)sizeof(AnVolume);
  return 7;
}

AN_EXPORT int an_render(const AnScene *scene, const AnSettings *settings, float *out_rgba, float *out_albedo,
                        float *out_normal, float *out_depth, float *out_variance, int32_t *out_spp, AnProgress progress) {
  if (!scene || !settings || !out_rgba) return -1;
  Engine eng;
  eng.S = scene;
  eng.ST = settings;
  eng.setup();
  const int W = settings->width, H = settings->height;
  if (W <= 0 || H <= 0) return -2;
  int rx0 = 0, ry0 = 0, rx1 = W, ry1 = H;
  if (settings->region_x1 > settings->region_x0 && settings->region_y1 > settings->region_y0) {
    rx0 = std::max(0, settings->region_x0); ry0 = std::max(0, settings->region_y0);
    rx1 = std::min(W, settings->region_x1); ry1 = std::min(H, settings->region_y1);
  }
  const size_t N = (size_t)W * H;
  Accum A;
  A.rgb.assign(N * 3, 0.f); A.lum2.assign(N, 0.f); A.alpha.assign(N, 0.f);
  A.albedo.assign(N * 3, 0.f); A.normal.assign(N * 3, 0.f); A.depth.assign(N, 0.f);
  A.count.assign(N, 0);
  const int TS = settings->tile_size > 0 ? settings->tile_size : 16;
  int tx = (rx1 - rx0 + TS - 1) / TS, ty = (ry1 - ry0 + TS - 1) / TS;
  int ntiles = tx * ty;
  std::vector<uint8_t> tile_active(ntiles, 1);
  int spp = std::max(1, settings->spp);
  bool adaptive = settings->adaptive_threshold > 0 && spp > 8;
  int min_spp = adaptive ? std::max(4, std::min(spp, settings->min_spp > 0 ? settings->min_spp : std::max(8, spp / 8))) : spp;
  int threads = settings->threads > 0 ? settings->threads : (int)std::max(1u, std::thread::hardware_concurrency());
  std::atomic<bool> cancel(false);
  float filter_sigma = std::max(0.f, settings->filter_width) * 0.35f;
  const bool primary_only = settings->integrator >= INT_NORMALS && settings->integrator <= INT_UV;

  // pass schedule
  std::vector<int> passes;
  {
    int done = 0;
    int first = adaptive ? min_spp : std::max(1, std::min(spp, std::max(1, spp / 8)));
    passes.push_back(first); done = first;
    int step = std::max(1, adaptive ? std::max(4, spp / 16) : std::max(1, spp / 8));
    while (done < spp) { int s = std::min(step, spp - done); passes.push_back(s); done += s; }
  }
  long long total_work = (long long)spp * ntiles, work_done = 0;

  for (size_t pi = 0; pi < passes.size() && !cancel.load(); ++pi) {
    int pass_spp = passes[pi];
    std::vector<int> active;
    for (int t = 0; t < ntiles; ++t) if (tile_active[t]) active.push_back(t);
    if (active.empty()) break;
    std::atomic<int> next(0), finished(0);
    auto worker = [&]() {
      for (;;) {
        if (cancel.load()) return;
        int ai = next.fetch_add(1);
        if (ai >= (int)active.size()) return;
        int tile = active[ai];
        int x0 = rx0 + (tile % tx) * TS, y0 = ry0 + (tile / tx) * TS;
        int x1 = std::min(rx1, x0 + TS), y1 = std::min(ry1, y0 + TS);
        for (int y = y0; y < y1; ++y)
          for (int x = x0; x < x1; ++x) {
            size_t pidx = (size_t)y * W + x;
            uint32_t ph = hash_combine(hash_combine(settings->seed, (uint32_t)x), (uint32_t)y);
            PCG rng(((uint64_t)ph << 32) ^ (uint64_t)A.count[pidx] * 0x9E3779B97F4A7C15ULL);
            for (int s = 0; s < pass_spp; ++s) {
              uint32_t si = (uint32_t)A.count[pidx];
              Sampler smp;
              smp.start(ph, si);
              float fx, fy; smp.get2(fx, fy);
              float lu1, lu2; smp.get2(lu1, lu2);
              float tu = smp.get1();
              float ox = 0.5f, oy = 0.5f;
              if (filter_sigma > 0 && !primary_only) { ox += gauss_sample(fx, filter_sigma); oy += gauss_sample(fy, filter_sigma); }
              else if (!primary_only) { ox = fx; oy = fy; }
              Ray r = eng.camera_ray(x + ox, y + oy, lu1, lu2, scene->has_motion ? tu : 0.f);
              Engine::Aux aux;
              V3 c = eng.Li(r, smp, rng, aux);
              if (!isfinite3(c)) c = V3(0);
              float l = lum(c);
              A.rgb[pidx * 3] += c.x; A.rgb[pidx * 3 + 1] += c.y; A.rgb[pidx * 3 + 2] += c.z;
              A.lum2[pidx] += l * l;
              A.alpha[pidx] += aux.alpha;
              A.albedo[pidx * 3] += aux.albedo.x; A.albedo[pidx * 3 + 1] += aux.albedo.y; A.albedo[pidx * 3 + 2] += aux.albedo.z;
              A.normal[pidx * 3] += aux.normal.x; A.normal[pidx * 3 + 1] += aux.normal.y; A.normal[pidx * 3 + 2] += aux.normal.z;
              A.depth[pidx] += aux.depth;
              A.count[pidx]++;
            }
          }
        finished.fetch_add(1);
      }
    };
    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
    if (progress) {
      while (finished.load() < (int)active.size()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        float f = (float)(work_done + (long long)finished.load() * pass_spp) / (float)total_work;
        if (progress(std::min(1.f, f))) { cancel.store(true); break; }
        bool alive = false;
        // detect if all workers ended (e.g. cancelled)
        if (next.load() >= (int)active.size() && finished.load() >= (int)active.size()) break;
        (void)alive;
      }
    }
    for (auto &t : pool) t.join();
    work_done += (long long)active.size() * pass_spp;
    // skipped tiles count as done for progress purposes
    work_done += (long long)(ntiles - (int)active.size()) * pass_spp;

    // adaptive convergence test
    if (adaptive && pi + 1 < passes.size()) {
      for (int t = 0; t < ntiles; ++t) {
        if (!tile_active[t]) continue;
        int x0 = rx0 + (t % tx) * TS, y0 = ry0 + (t / tx) * TS;
        int x1 = std::min(rx1, x0 + TS), y1 = std::min(ry1, y0 + TS);
        float worst = 0;
        for (int y = y0; y < y1; ++y)
          for (int x = x0; x < x1; ++x) {
            size_t p = (size_t)y * W + x;
            int n = A.count[p];
            if (n < 2) { worst = INF; continue; }
            float mean = lum(V3(A.rgb[p * 3], A.rgb[p * 3 + 1], A.rgb[p * 3 + 2])) / n;
            float var = std::max(0.f, (A.lum2[p] / n - mean * mean)) * n / (n - 1);
            float err = std::sqrt(var / n) / (std::sqrt(std::max(mean, 0.f)) + 0.05f);
            worst = std::max(worst, err);
          }
        if (worst < settings->adaptive_threshold) tile_active[t] = 0;
      }
    }
  }
  if (progress) progress(1.f);

  // resolve
  for (size_t p = 0; p < N; ++p) {
    int n = A.count[p];
    float inv = n > 0 ? 1.f / n : 0.f;
    float r = A.rgb[p * 3] * inv, g = A.rgb[p * 3 + 1] * inv, b = A.rgb[p * 3 + 2] * inv;
    out_rgba[p * 4] = r; out_rgba[p * 4 + 1] = g; out_rgba[p * 4 + 2] = b;
    out_rgba[p * 4 + 3] = A.alpha[p] * inv;
    if (out_albedo) for (int k = 0; k < 3; ++k) out_albedo[p * 3 + k] = A.albedo[p * 3 + k] * inv;
    if (out_normal) {
      V3 nn(A.normal[p * 3], A.normal[p * 3 + 1], A.normal[p * 3 + 2]);
      float l = length(nn);
      if (l > 1e-6f) nn = nn / l;
      out_normal[p * 3] = nn.x; out_normal[p * 3 + 1] = nn.y; out_normal[p * 3 + 2] = nn.z;
    }
    if (out_depth) out_depth[p] = A.depth[p] * inv;
    if (out_variance) {
      float mean = lum(V3(r, g, b));
      float var = n > 1 ? std::max(0.f, (A.lum2[p] * inv - mean * mean)) * n / (n - 1) / n : 0.f;
      out_variance[p] = var;
    }
    if (out_spp) out_spp[p] = n;
  }
  return cancel.load() ? 1 : 0;
}

// ============================================================================
// Edge-avoiding a-trous wavelet denoiser (SVGF-style spatial filter)
// ============================================================================
AN_EXPORT int an_denoise(int W, int H, const float *rgb, const float *albedo, const float *normal, const float *depth,
                         const float *variance, float *out, int iterations, float sigma_l, float sigma_n, float sigma_z,
                         int threads) {
  if (W <= 0 || H <= 0 || !rgb || !out) return -1;
  const size_t N = (size_t)W * H;
  if (threads <= 0) threads = (int)std::max(1u, std::thread::hardware_concurrency());
  std::vector<float> col(N * 3), tmp(N * 3), var(N, 0.f), vtmp(N, 0.f), alb(N * 3, 1.f), dgrad(N, 0.f);
  for (size_t p = 0; p < N; ++p) {
    for (int k = 0; k < 3; ++k) {
      float a = albedo ? std::max(albedo[p * 3 + k], 0.02f) : 1.f;
      alb[p * 3 + k] = a;
      col[p * 3 + k] = rgb[p * 3 + k] / a;
    }
    float la = albedo ? std::max(0.02f, lum(V3(alb[p * 3], alb[p * 3 + 1], alb[p * 3 + 2]))) : 1.f;
    var[p] = variance ? variance[p] / (la * la) : 0.f;
  }
  if (depth) {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        size_t p = (size_t)y * W + x;
        float dx = 0, dy = 0;
        if (x > 0 && x < W - 1) dx = 0.5f * std::fabs(depth[p + 1] - depth[p - 1]);
        if (y > 0 && y < H - 1) dy = 0.5f * std::fabs(depth[p + W] - depth[p - W]);
        dgrad[p] = std::max(dx, dy);
      }
  }
  const float kern[3] = {3.f / 8.f, 1.f / 4.f, 1.f / 16.f};
  iterations = std::max(1, std::min(iterations, 8));
  auto run_rows = [&](int step, int y_begin, int y_end) {
    for (int y = y_begin; y < y_end; ++y)
      for (int x = 0; x < W; ++x) {
        size_t p = (size_t)y * W + x;
        V3 cp(col[p * 3], col[p * 3 + 1], col[p * 3 + 2]);
        float lp = lum(cp);
        V3 np = normal ? V3(normal[p * 3], normal[p * 3 + 1], normal[p * 3 + 2]) : V3(0);
        bool np_bg = dot(np, np) < 0.25f;
        float zp = depth ? depth[p] : 0;
        // prefiltered variance (3x3 gaussian)
        float vsum = 0, vw = 0;
        for (int j = -1; j <= 1; ++j)
          for (int i = -1; i <= 1; ++i) {
            int xx = x + i, yy = y + j;
            if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
            float k = (i == 0 ? 0.5f : 0.25f) * (j == 0 ? 0.5f : 0.25f);
            vsum += var[(size_t)yy * W + xx] * k; vw += k;
          }
        float sv = std::sqrt(std::max(0.f, vsum / std::max(vw, 1e-8f)));
        V3 acc(0); float wsum = 0, vacc = 0;
        for (int j = -2; j <= 2; ++j)
          for (int i = -2; i <= 2; ++i) {
            int xx = x + i * step, yy = y + j * step;
            if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
            size_t q = (size_t)yy * W + xx;
            float k = kern[std::abs(i)] * kern[std::abs(j)];
            V3 cq(col[q * 3], col[q * 3 + 1], col[q * 3 + 2]);
            float w = k;
            if (normal) {
              V3 nq(normal[q * 3], normal[q * 3 + 1], normal[q * 3 + 2]);
              bool nq_bg = dot(nq, nq) < 0.25f;
              if (np_bg != nq_bg) w = 0;
              else if (!np_bg) w *= std::pow(std::max(0.f, dot(np, nq)), sigma_n);
            }
            if (depth && w > 0) {
              float dz = std::fabs(zp - depth[q]);
              float denom = sigma_z * dgrad[p] * std::sqrt((float)(i * i + j * j)) * step + 1e-3f * std::max(zp, 1e-3f);
              w *= std::exp(-dz / denom);
            }
            if (w > 0) {
              float dl = std::fabs(lp - lum(cq));
              w *= std::exp(-dl / (sigma_l * sv + 1e-4f));
            }
            acc += cq * w; wsum += w; vacc += w * w * var[q];
          }
        if (wsum > 0) {
          tmp[p * 3] = acc.x / wsum; tmp[p * 3 + 1] = acc.y / wsum; tmp[p * 3 + 2] = acc.z / wsum;
          vtmp[p] = vacc / (wsum * wsum);
        } else {
          tmp[p * 3] = cp.x; tmp[p * 3 + 1] = cp.y; tmp[p * 3 + 2] = cp.z;
          vtmp[p] = var[p];
        }
      }
  };
  for (int it = 0; it < iterations; ++it) {
    int step = 1 << it;
    std::vector<std::thread> pool;
    int rows = (H + threads - 1) / threads;
    for (int t = 0; t < threads; ++t) {
      int yb = t * rows, ye = std::min(H, yb + rows);
      if (yb >= ye) break;
      pool.emplace_back(run_rows, step, yb, ye);
    }
    for (auto &th : pool) th.join();
    std::swap(col, tmp);
    std::swap(var, vtmp);
  }
  for (size_t p = 0; p < N; ++p)
    for (int k = 0; k < 3; ++k) out[p * 3 + k] = col[p * 3 + k] * alb[p * 3 + k];
  return 0;
}

// =============================================================================================
// Geometry kernels: mesh signed distance (BVH closest point + fast winding number sign)
// =============================================================================================
namespace geomk {
struct G3 { double x, y, z; };
static inline G3 gsub(G3 a, G3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline G3 gadd(G3 a, G3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline G3 gmul(G3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
static inline double gdot(G3 a, G3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline G3 gcross(G3 a, G3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline double glen(G3 a) { return std::sqrt(gdot(a, a)); }

struct GNode { double lo[3], hi[3]; G3 c, N; double r; int left, right, start, count; };

struct GBVH {
  std::vector<G3> V;
  std::vector<int> F;   // 3 per tri
  std::vector<int> tri; // permutation
  std::vector<GNode> nodes;
  std::vector<G3> cent;

  void build(const float *v, int nv, const int *f, int nf) {
    V.resize(nv);
    for (int i = 0; i < nv; i++) V[i] = {v[3 * i], v[3 * i + 1], v[3 * i + 2]};
    F.assign(f, f + 3 * nf);
    tri.resize(nf);
    cent.resize(nf);
    for (int i = 0; i < nf; i++) {
      tri[i] = i;
      cent[i] = gmul(gadd(gadd(V[F[3 * i]], V[F[3 * i + 1]]), V[F[3 * i + 2]]), 1.0 / 3.0);
    }
    nodes.reserve(2 * nf + 1);
    if (nf) rec(0, nf);
  }
  int rec(int s, int e) {
    int id = (int)nodes.size();
    nodes.push_back(GNode());
    GNode nd;
    for (int k = 0; k < 3; k++) { nd.lo[k] = 1e300; nd.hi[k] = -1e300; }
    G3 N{0, 0, 0}, c{0, 0, 0};
    double A = 0;
    for (int i = s; i < e; i++) {
      int t = tri[i];
      G3 a = V[F[3 * t]], b = V[F[3 * t + 1]], d = V[F[3 * t + 2]];
      for (G3 p : {a, b, d}) {
        nd.lo[0] = std::min(nd.lo[0], p.x); nd.hi[0] = std::max(nd.hi[0], p.x);
        nd.lo[1] = std::min(nd.lo[1], p.y); nd.hi[1] = std::max(nd.hi[1], p.y);
        nd.lo[2] = std::min(nd.lo[2], p.z); nd.hi[2] = std::max(nd.hi[2], p.z);
      }
      G3 n = gmul(gcross(gsub(b, a), gsub(d, a)), 0.5);
      double ar = glen(n);
      N = gadd(N, n);
      c = gadd(c, gmul(cent[t], ar));
      A += ar;
    }
    nd.N = N;
    nd.c = A > 0 ? gmul(c, 1.0 / A) : G3{(nd.lo[0] + nd.hi[0]) / 2, (nd.lo[1] + nd.hi[1]) / 2, (nd.lo[2] + nd.hi[2]) / 2};
    double r = 0;
    for (int i = s; i < e; i++) {
      int t = tri[i];
      for (int k = 0; k < 3; k++) r = std::max(r, glen(gsub(V[F[3 * t + k]], nd.c)));
    }
    nd.r = r;
    nd.start = s; nd.count = e - s; nd.left = nd.right = -1;
    if (e - s > 4) {
      int ax = 0;
      double ext = -1;
      for (int k = 0; k < 3; k++) if (nd.hi[k] - nd.lo[k] > ext) { ext = nd.hi[k] - nd.lo[k]; ax = k; }
      int m = (s + e) / 2;
      auto key = [&](int t) { const G3 &q = cent[t]; return ax == 0 ? q.x : ax == 1 ? q.y : q.z; };
      std::nth_element(tri.begin() + s, tri.begin() + m, tri.begin() + e, [&](int a, int b) { return key(a) < key(b); });
      nd.left = rec(s, m);
      nd.right = rec(m, e);
    }
    nodes[id] = nd;
    return id;
  }
  static G3 closest_tri(G3 p, G3 a, G3 b, G3 c) {  // Ericson, Real-Time Collision Detection
    G3 ab = gsub(b, a), ac = gsub(c, a), ap = gsub(p, a);
    double d1 = gdot(ab, ap), d2 = gdot(ac, ap);
    if (d1 <= 0 && d2 <= 0) return a;
    G3 bp = gsub(p, b);
    double d3 = gdot(ab, bp), d4 = gdot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return b;
    double vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return gadd(a, gmul(ab, d1 / (d1 - d3)));
    G3 cp = gsub(p, c);
    double d5 = gdot(ab, cp), d6 = gdot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return c;
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return gadd(a, gmul(ac, d2 / (d2 - d6)));
    double va = d3 * d6 - d5 * d4;
    if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0)
      return gadd(b, gmul(gsub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6))));
    double den = 1.0 / (va + vb + vc);
    return gadd(a, gadd(gmul(ab, vb * den), gmul(ac, vc * den)));
  }
  static double box_d2(const GNode &n, G3 p) {
    double d = 0, q[3] = {p.x, p.y, p.z};
    for (int k = 0; k < 3; k++) {
      double v = std::max(std::max(n.lo[k] - q[k], 0.0), q[k] - n.hi[k]);
      d += v * v;
    }
    return d;
  }
  double closest(G3 p, G3 &best_pt, int &best_tri) const {
    double best = 1e300;
    int stack[128], sp = 0;
    stack[sp++] = 0;
    while (sp) {
      const GNode &n = nodes[stack[--sp]];
      if (box_d2(n, p) >= best) continue;
      if (n.left < 0) {
        for (int i = n.start; i < n.start + n.count; i++) {
          int t = tri[i];
          G3 q = closest_tri(p, V[F[3 * t]], V[F[3 * t + 1]], V[F[3 * t + 2]]);
          G3 dv = gsub(q, p);
          double d2 = gdot(dv, dv);
          if (d2 < best) { best = d2; best_pt = q; best_tri = t; }
        }
      } else {
        const GNode &L = nodes[n.left], &R = nodes[n.right];
        double dl = box_d2(L, p), dr = box_d2(R, p);
        if (sp + 2 >= 128) continue;
        if (dl < dr) { stack[sp++] = n.right; stack[sp++] = n.left; }
        else { stack[sp++] = n.left; stack[sp++] = n.right; }
      }
    }
    return std::sqrt(best);
  }
  static double solid_angle(G3 p, G3 a, G3 b, G3 c) {  // van Oosterom & Strackee
    a = gsub(a, p); b = gsub(b, p); c = gsub(c, p);
    double la = glen(a), lb = glen(b), lc = glen(c);
    double num = gdot(a, gcross(b, c));
    double den = la * lb * lc + gdot(a, b) * lc + gdot(b, c) * la + gdot(c, a) * lb;
    return 2.0 * std::atan2(num, den);
  }
  double winding(G3 p, double beta) const {
    double w = 0;
    int stack[128], sp = 0;
    stack[sp++] = 0;
    while (sp) {
      const GNode &n = nodes[stack[--sp]];
      G3 d = gsub(n.c, p);
      double dist = glen(d);
      if (n.left >= 0 && dist > beta * n.r) {
        w += gdot(d, n.N) / (dist * dist * dist);  // dipole far-field approximation
        continue;
      }
      if (n.left < 0) {
        for (int i = n.start; i < n.start + n.count; i++) {
          int t = tri[i];
          w += solid_angle(p, V[F[3 * t]], V[F[3 * t + 1]], V[F[3 * t + 2]]);
        }
      } else if (sp + 2 < 128) {
        stack[sp++] = n.left;
        stack[sp++] = n.right;
      }
    }
    return w / (4.0 * M_PI);
  }
};
}  // namespace geomk

// Signed distance from points to a triangle mesh (negative inside, sign from the generalised
// winding number so slightly open meshes still work). Optional outputs: winding number and
// closest surface point. Returns 0 on success.
AN_EXPORT int an_mesh_sdf(const float *verts, int nv, const int *tris, int nt, const float *pts, int np,
                          float *out_dist, float *out_wind, float *out_closest, int threads) {
  if (nt <= 0 || nv <= 0) return 1;
  geomk::GBVH bvh;
  bvh.build(verts, nv, tris, nt);
  if (threads <= 0) threads = (int)std::max(1u, std::thread::hardware_concurrency());
  std::atomic<int> next(0);
  auto work = [&]() {
    for (;;) {
      int s = next.fetch_add(256);
      if (s >= np) break;
      int e = std::min(np, s + 256);
      for (int i = s; i < e; i++) {
        geomk::G3 p{pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]}, q{0, 0, 0};
        int t = -1;
        double d = bvh.closest(p, q, t);
        double w = bvh.winding(p, 2.0);
        if (out_wind) out_wind[i] = (float)w;
        if (out_closest) { out_closest[3 * i] = (float)q.x; out_closest[3 * i + 1] = (float)q.y; out_closest[3 * i + 2] = (float)q.z; }
        out_dist[i] = (float)(w > 0.5 ? -d : d);
      }
    }
  };
  std::vector<std::thread> pool;
  for (int k = 0; k < threads - 1; k++) pool.emplace_back(work);
  work();
  for (auto &th : pool) th.join();
  return 0;
}

// =====================================================================================
// Position-Based Fluids (Macklin & Mueller 2013) + point self-repulsion for cloth
// =====================================================================================
namespace pbfk {
struct P3 { float x, y, z; };
static inline P3 operator+(P3 a, P3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline P3 operator-(P3 a, P3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline P3 operator*(P3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static inline float pdot(P3 a, P3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline P3 pcross(P3 a, P3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }

template <class F> static void pfor(int n, int threads, F fn) {
  if (threads <= 0) threads = (int)std::max(1u, std::thread::hardware_concurrency());
  if (n < 2048 || threads == 1) { fn(0, n); return; }
  std::vector<std::thread> ts;
  int chunk = (n + threads - 1) / threads;
  for (int t = 0; t < threads; t++) {
    int a = t * chunk, b = std::min(n, a + chunk);
    if (a >= b) break;
    ts.emplace_back([=]() { fn(a, b); });
  }
  for (auto &t : ts) t.join();
}

struct HashGrid {
  float lo[3], cell;
  int dims[3];
  std::vector<int> start, sorted;
  void build(const P3 *p, int n, const float *lo_, const float *hi_, float h) {
    cell = h;
    for (int a = 0; a < 3; a++) {
      lo[a] = lo_[a] - h;
      dims[a] = std::max(1, (int)std::ceil((hi_[a] - lo_[a] + 2 * h) / h));
    }
    long nc = (long)dims[0] * dims[1] * dims[2];
    if (nc > 64L * 1024 * 1024) {  // fall back to a coarser grid on huge domains
      float f = std::cbrt((float)nc / (64.f * 1024 * 1024));
      cell *= f;
      for (int a = 0; a < 3; a++) dims[a] = std::max(1, (int)std::ceil((hi_[a] - lo_[a] + 2 * h) / cell));
      nc = (long)dims[0] * dims[1] * dims[2];
    }
    start.assign(nc + 1, 0);
    sorted.resize(n);
    std::vector<int> cid(n);
    for (int i = 0; i < n; i++) {
      cid[i] = index(p[i]);
      start[cid[i] + 1]++;
    }
    for (long c = 0; c < nc; c++) start[c + 1] += start[c];
    std::vector<int> fill(start.begin(), start.end() - 1);
    for (int i = 0; i < n; i++) sorted[fill[cid[i]]++] = i;
  }
  inline void coords(const P3 &q, int &x, int &y, int &z) const {
    x = std::min(dims[0] - 1, std::max(0, (int)((q.x - lo[0]) / cell)));
    y = std::min(dims[1] - 1, std::max(0, (int)((q.y - lo[1]) / cell)));
    z = std::min(dims[2] - 1, std::max(0, (int)((q.z - lo[2]) / cell)));
  }
  inline int index(const P3 &q) const {
    int x, y, z;
    coords(q, x, y, z);
    return (z * dims[1] + y) * dims[0] + x;
  }
  template <class F> void query(const P3 &q, F fn) const {
    int cx, cy, cz;
    coords(q, cx, cy, cz);
    for (int z = std::max(0, cz - 1); z <= std::min(dims[2] - 1, cz + 1); z++)
      for (int y = std::max(0, cy - 1); y <= std::min(dims[1] - 1, cy + 1); y++) {
        int row = (z * dims[1] + y) * dims[0];
        int x0 = std::max(0, cx - 1), x1 = std::min(dims[0] - 1, cx + 1);
        for (int k = start[row + x0]; k < start[row + x1 + 1]; k++) fn(sorted[k]);
      }
  }
};

struct SDFGrid {
  const float *d = nullptr;
  int nx = 0, ny = 0, nz = 0;
  float lo[3] = {0, 0, 0}, cell = 1;
  float sample(P3 p) const {
    float x = (p.x - lo[0]) / cell, y = (p.y - lo[1]) / cell, z = (p.z - lo[2]) / cell;
    if (x < 0 || y < 0 || z < 0 || x > nx - 1.001f || y > ny - 1.001f || z > nz - 1.001f) return 1e30f;
    int x0 = (int)x, y0 = (int)y, z0 = (int)z;
    float fx = x - x0, fy = y - y0, fz = z - z0;
    auto g = [&](int i, int j, int k) { return d[((long)(z0 + k) * ny + (y0 + j)) * nx + (x0 + i)]; };
    float c00 = g(0, 0, 0) * (1 - fx) + g(1, 0, 0) * fx, c10 = g(0, 1, 0) * (1 - fx) + g(1, 1, 0) * fx;
    float c01 = g(0, 0, 1) * (1 - fx) + g(1, 0, 1) * fx, c11 = g(0, 1, 1) * (1 - fx) + g(1, 1, 1) * fx;
    return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
  }
  P3 grad(P3 p) const {
    float e = cell * 0.5f;
    P3 g = {sample({p.x + e, p.y, p.z}) - sample({p.x - e, p.y, p.z}),
            sample({p.x, p.y + e, p.z}) - sample({p.x, p.y - e, p.z}),
            sample({p.x, p.y, p.z + e}) - sample({p.x, p.y, p.z - e})};
    float l = std::sqrt(pdot(g, g));
    return l > 1e-12f && l < 1e20f ? g * (1.f / l) : P3{0, 1, 0};
  }
};
}  // namespace pbfk

// prm: 0 dt, 1 h, 2 rho0, 3 iterations, 4-6 gravity, 7-9 domain lo, 10-12 domain hi, 13 xsph viscosity,
//      14 vorticity eps, 15 scorr k, 16 particle radius, 17 cfm eps, 18 friction, 19 open top (0/1),
//      20 damping (per second), 21 max speed (0 = off), 22 max correction per iteration (in h)
// sdf_info: nx, ny, nz (int) ; sdf_xf: lo xyz, cell. sdf may be null.
AN_EXPORT int an_pbf_step(float *pos, float *vel, int n, const float *prm, const float *sdf, const int *sdf_info,
                          const float *sdf_xf, int threads) {
  using namespace pbfk;
  if (n <= 0) return 0;
  const float dt = prm[0], h = prm[1], rho0 = prm[2];
  const int iters = std::max(1, (int)prm[3]);
  const P3 g = {prm[4], prm[5], prm[6]};
  const float lo[3] = {prm[7], prm[8], prm[9]}, hi[3] = {prm[10], prm[11], prm[12]};
  const float xsph = prm[13], vort = prm[14], scorr_k = prm[15], rad = prm[16], cfm = prm[17], fric = prm[18];
  const bool open_top = prm[19] > 0.5f;
  const float damp = std::exp(-std::max(0.f, prm[20]) * dt);
  const float vmax_clamp = prm[21];
  P3 *X = reinterpret_cast<P3 *>(pos);
  P3 *Vv = reinterpret_cast<P3 *>(vel);
  std::vector<P3> Pp(n), dP(n);
  std::vector<float> lam(n), rho(n);
  SDFGrid sg;
  if (sdf && sdf_info) {
    sg.d = sdf;
    sg.nx = sdf_info[0]; sg.ny = sdf_info[1]; sg.nz = sdf_info[2];
    sg.lo[0] = sdf_xf[0]; sg.lo[1] = sdf_xf[1]; sg.lo[2] = sdf_xf[2]; sg.cell = sdf_xf[3];
  }
  const float h2 = h * h;
  const float poly6 = 315.f / (64.f * PI * std::pow(h, 9.f));
  const float spiky = -45.f / (PI * std::pow(h, 6.f));
  auto W = [&](float r2) { float d = h2 - r2; return d > 0 ? poly6 * d * d * d : 0.f; };
  auto gradW = [&](P3 r, float rl) { if (rl <= 1e-9f || rl >= h) return P3{0, 0, 0}; float c = spiky * (h - rl) * (h - rl) / rl; return r * c; };
  const float wq = W(0.04f * h2);  // s_corr reference (|dq| = 0.2h)

  // 1. predict
  pfor(n, threads, [&](int a, int b) {
    for (int i = a; i < b; i++) {
      Vv[i] = (Vv[i] + g * dt) * damp;
      Pp[i] = X[i] + Vv[i] * dt;
    }
  });
  auto collide = [&](int a, int b) {
    for (int i = a; i < b; i++) {
      P3 &p = Pp[i];
      float *pc = &p.x;
      for (int k = 0; k < 3; k++) {
        if (pc[k] < lo[k] + rad) pc[k] = lo[k] + rad;
        if (pc[k] > hi[k] - rad && !(open_top && k == 1)) pc[k] = hi[k] - rad;
      }
      if (sg.d) {
        float d = sg.sample(p);
        if (d < rad) {
          P3 nrm = sg.grad(p);
          p = p + nrm * (rad - d);
          if (fric > 0) {  // kill part of the tangential displacement this step
            P3 disp = p - X[i];
            P3 tang = disp - nrm * pdot(disp, nrm);
            p = p - tang * std::min(1.f, fric);
          }
        }
      }
    }
  };
  pfor(n, threads, collide);
  // 2. neighbours
  HashGrid grid;
  grid.build(Pp.data(), n, lo, hi, h);
  const int K = 80;
  std::vector<int> nb((size_t)n * K), nc(n);
  pfor(n, threads, [&](int a, int b) {
    for (int i = a; i < b; i++) {
      int c = 0;
      P3 pi = Pp[i];
      grid.query(pi, [&](int j) {
        if (j == i || c >= K) return;
        P3 r = pi - Pp[j];
        if (pdot(r, r) < h2) nb[(size_t)i * K + c++] = j;
      });
      nc[i] = c;
    }
  });
  // 3. density constraint iterations
  for (int it = 0; it < iters; it++) {
    pfor(n, threads, [&](int a, int b) {
      for (int i = a; i < b; i++) {
        P3 pi = Pp[i];
        float d = W(0.f);
        P3 gi = {0, 0, 0};
        float sg2 = 0;
        for (int k = 0; k < nc[i]; k++) {
          int j = nb[(size_t)i * K + k];
          P3 r = pi - Pp[j];
          float r2 = pdot(r, r);
          d += W(r2);
          P3 gw = gradW(r, std::sqrt(r2)) * (1.f / rho0);
          gi = gi + gw;
          sg2 += pdot(gw, gw);
        }
        sg2 += pdot(gi, gi);
        rho[i] = d;
        float C = std::max(d / rho0 - 1.f, 0.f);
        lam[i] = -C / (sg2 + cfm);
      }
    });
    pfor(n, threads, [&](int a, int b) {
      for (int i = a; i < b; i++) {
        P3 pi = Pp[i];
        P3 acc = {0, 0, 0};
        for (int k = 0; k < nc[i]; k++) {
          int j = nb[(size_t)i * K + k];
          P3 r = pi - Pp[j];
          float r2 = pdot(r, r);
          float s = 0;
          if (scorr_k > 0) { float q = W(r2) / wq; s = -scorr_k * q * q * q * q; }
          acc = acc + gradW(r, std::sqrt(r2)) * (lam[i] + lam[j] + s);
        }
        dP[i] = acc * (1.f / rho0);
      }
    });
    pfor(n, threads, [&](int a, int b) {
      for (int i = a; i < b; i++) {
        P3 d = dP[i];
        float l = std::sqrt(pdot(d, d)), mx = h * prm[22];  // limit per-iteration correction
        if (l > mx) d = d * (mx / l);
        Pp[i] = Pp[i] + d;
      }
    });
    pfor(n, threads, collide);
  }
  // 4. velocities, vorticity confinement, XSPH viscosity
  std::vector<P3> Vn(n), omega(n);
  pfor(n, threads, [&](int a, int b) {
    for (int i = a; i < b; i++) Vn[i] = (Pp[i] - X[i]) * (1.f / dt);
  });
  if (vort > 0) {
    pfor(n, threads, [&](int a, int b) {
      for (int i = a; i < b; i++) {
        P3 w = {0, 0, 0};
        for (int k = 0; k < nc[i]; k++) {
          int j = nb[(size_t)i * K + k];
          P3 r = Pp[i] - Pp[j];
          w = w + pcross(Vn[j] - Vn[i], gradW(r, std::sqrt(pdot(r, r))) * -1.f);
        }
        omega[i] = w * (1.f / rho0);
      }
    });
  }
  std::vector<P3> Vout(n);
  pfor(n, threads, [&](int a, int b) {
    for (int i = a; i < b; i++) {
      P3 v = Vn[i], acc = {0, 0, 0}, eta = {0, 0, 0};
      float wo = std::sqrt(pdot(omega[i], omega[i]));
      for (int k = 0; k < nc[i]; k++) {
        int j = nb[(size_t)i * K + k];
        P3 r = Pp[i] - Pp[j];
        float r2 = pdot(r, r);
        acc = acc + (Vn[j] - v) * W(r2);
        if (vort > 0) {
          float wj = std::sqrt(pdot(omega[j], omega[j]));
          eta = eta + gradW(r, std::sqrt(r2)) * (wj - wo) * -1.f;
        }
      }
      v = v + acc * (xsph / rho0);
      if (vort > 0) {
        float el = std::sqrt(pdot(eta, eta));
        if (el > 1e-9f) v = v + pcross(eta * (1.f / el), omega[i]) * (vort * dt);
      }
      if (vmax_clamp > 0) {
        float sp = std::sqrt(pdot(v, v));
        if (sp > vmax_clamp) v = v * (vmax_clamp / sp);
      }
      Vout[i] = v;
      X[i] = Pp[i];
    }
  });
  for (int i = 0; i < n; i++) Vv[i] = Vout[i];
  return 0;
}

// Splat particles into a density grid (nx, ny, nz; x fastest) with the poly6-shaped kernel
// (1 - r^2/R^2)^3 of radius R. Used for liquid surface reconstruction.
AN_EXPORT int an_splat(const float *pos, int n, float *grid, const int *dims, const float *xf, float R, int threads) {
  using namespace pbfk;
  const int nx = dims[0], ny = dims[1], nz = dims[2];
  const float lx = xf[0], ly = xf[1], lz = xf[2], cell = xf[3];
  if (threads <= 0) threads = (int)std::max(1u, std::thread::hardware_concurrency());
  // split along z into slabs so threads never write the same voxel
  std::vector<std::vector<int>> buckets(nz);
  int rc = (int)std::ceil(R / cell);
  for (int i = 0; i < n; i++) {
    int z = (int)std::floor((pos[3 * i + 2] - lz) / cell);
    for (int k = std::max(0, z - rc); k <= std::min(nz - 1, z + rc + 1); k++) buckets[k].push_back(i);
  }
  const float R2 = R * R;
  pfor(nz, nz >= threads ? threads : 1, [&](int za, int zb) {
    for (int z = za; z < zb; z++) {
      float pz = lz + z * cell;
      for (int i : buckets[z]) {
        float dz = pos[3 * i + 2] - pz;
        if (dz * dz >= R2) continue;
        float px = pos[3 * i], py = pos[3 * i + 1];
        int x0 = std::max(0, (int)std::floor((px - R - lx) / cell)), x1 = std::min(nx - 1, (int)std::ceil((px + R - lx) / cell));
        int y0 = std::max(0, (int)std::floor((py - R - ly) / cell)), y1 = std::min(ny - 1, (int)std::ceil((py + R - ly) / cell));
        for (int y = y0; y <= y1; y++) {
          float dy = py - (ly + y * cell);
          float r2yz = dy * dy + dz * dz;
          if (r2yz >= R2) continue;
          float *row = grid + ((long)z * ny + y) * nx;
          for (int x = x0; x <= x1; x++) {
            float dx = px - (lx + x * cell);
            float r2 = dx * dx + r2yz;
            if (r2 < R2) { float q = 1.f - r2 / R2; row[x] += q * q * q; }
          }
        }
      }
    }
  });
  return 0;
}

// Cloth self-collision: push apart vertex pairs closer than ``thickness`` whose rest-pose
// distance is larger than ``rest_ignore`` (so neighbouring vertices are not affected).
AN_EXPORT int an_self_collide(float *pos, const float *prev, const float *rest, int n, float thickness,
                              float rest_ignore, int iters, int threads) {
  using namespace pbfk;
  if (n <= 1) return 0;
  P3 *X = reinterpret_cast<P3 *>(pos);
  const P3 *R = reinterpret_cast<const P3 *>(rest);
  const P3 *Pv = reinterpret_cast<const P3 *>(prev);
  const float t2 = thickness * thickness, ri2 = rest_ignore * rest_ignore;
  std::vector<P3> d(n);
  std::vector<int> cnt(n);
  int total = 0;
  for (int it = 0; it < iters; it++) {
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = 0; i < n; i++) {
      const float *p = &X[i].x;
      for (int a = 0; a < 3; a++) { lo[a] = std::min(lo[a], p[a]); hi[a] = std::max(hi[a], p[a]); }
    }
    HashGrid grid;
    grid.build(X, n, lo, hi, thickness);
    std::atomic<int> hits{0};
    pfor(n, threads, [&](int a, int b) {
      int local = 0;
      for (int i = a; i < b; i++) {
        P3 acc = {0, 0, 0};
        int c = 0;
        P3 pi = X[i];
        grid.query(pi, [&](int j) {
          if (j == i) return;
          P3 r = pi - X[j];
          float r2 = pdot(r, r);
          if (r2 >= t2) return;
          P3 rr = R[i] - R[j];
          if (pdot(rr, rr) <= ri2) return;
          float l = std::sqrt(r2);
          P3 dir;
          if (l > 1e-7f) dir = r * (1.f / l);
          else {  // coincident: separate along the previous-frame offset
            P3 pr = Pv[i] - Pv[j];
            float pl = std::sqrt(pdot(pr, pr));
            dir = pl > 1e-9f ? pr * (1.f / pl) : P3{0, 1, 0};
          }
          // prefer the side the pair came from (prevents passing through)
          P3 pr = Pv[i] - Pv[j];
          if (pdot(pr, pr) > 1e-12f && pdot(pr, dir) < 0 && pdot(pr, pr) < t2 * 4) {
            float pl = std::sqrt(pdot(pr, pr));
            dir = pr * (1.f / pl);
            l = -l;
          }
          acc = acc + dir * (0.5f * (thickness - l));
          c++;
        });
        d[i] = acc;
        cnt[i] = c;
        local += c;
      }
      hits += local;
    });
    if (hits.load() == 0) break;
    total += hits.load();
    for (int i = 0; i < n; i++)
      if (cnt[i]) X[i] = X[i] + d[i] * (1.f / std::max(1, cnt[i]) * std::min(cnt[i], 4) * 0.5f);
  }
  return total;
}

// =====================================================================================
// Eulerian smoke kernels (collocated grid, grid-unit velocities). All arrays are [z][y][x].
// =====================================================================================
namespace fluidk {
struct Grid {
  int nx, ny, nz;
  inline size_t id(int x, int y, int z) const { return ((size_t)z * ny + y) * nx + x; }
  size_t n() const { return (size_t)nx * ny * nz; }
};

// run fn(z0, z1) over z-slabs with a fixed worker count (no small-size threshold)
template <class F> static void zfor(int nz, int threads, F fn) {
  if (threads <= 0) threads = (int)std::max(1u, std::thread::hardware_concurrency());
  threads = std::max(1, std::min(threads, nz));
  if (threads == 1) { fn(0, nz); return; }
  std::vector<std::thread> ts;
  int chunk = (nz + threads - 1) / threads;
  for (int t = 0; t < threads; t++) {
    int a = t * chunk, b = std::min(nz, a + chunk);
    if (a >= b) break;
    ts.emplace_back([=]() { fn(a, b); });
  }
  for (auto &t : ts) t.join();
}

static inline float samp(const float *f, const Grid &g, float x, float y, float z) {
  x = std::min(std::max(x, 0.f), g.nx - 1.001f);
  y = std::min(std::max(y, 0.f), g.ny - 1.001f);
  z = std::min(std::max(z, 0.f), g.nz - 1.001f);
  int x0 = (int)x, y0 = (int)y, z0 = (int)z;
  float fx = x - x0, fy = y - y0, fz = z - z0;
  size_t b = g.id(x0, y0, z0), sy = g.nx, sz = (size_t)g.nx * g.ny;
  float c00 = f[b] * (1 - fx) + f[b + 1] * fx;
  float c10 = f[b + sy] * (1 - fx) + f[b + sy + 1] * fx;
  float c01 = f[b + sz] * (1 - fx) + f[b + sz + 1] * fx;
  float c11 = f[b + sz + sy] * (1 - fx) + f[b + sz + sy + 1] * fx;
  return (c00 * (1 - fy) + c10 * fy) * (1 - fz) + (c01 * (1 - fy) + c11 * fy) * fz;
}

static inline void samp_minmax(const float *f, const Grid &g, float x, float y, float z, float &lo, float &hi) {
  int x0 = std::min(std::max((int)std::floor(x), 0), g.nx - 2);
  int y0 = std::min(std::max((int)std::floor(y), 0), g.ny - 2);
  int z0 = std::min(std::max((int)std::floor(z), 0), g.nz - 2);
  size_t b = g.id(x0, y0, z0), sy = g.nx, sz = (size_t)g.nx * g.ny;
  size_t o[8] = {0, 1, sy, sy + 1, sz, sz + 1, sz + sy, sz + sy + 1};
  lo = hi = f[b];
  for (int k = 1; k < 8; k++) { float v = f[b + o[k]]; lo = std::min(lo, v); hi = std::max(hi, v); }
}

// simple reusable spin barrier for persistent worker regions
struct Barrier {
  std::atomic<int> count{0}, gen{0};
  int n;
  explicit Barrier(int n_) : n(n_) {}
  void wait() {
    int g = gen.load();
    if (count.fetch_add(1) == n - 1) { count.store(0); gen.fetch_add(1); return; }
    int spins = 0;
    while (gen.load() == g) { if (++spins > 64) { std::this_thread::yield(); spins = 0; } }
  }
};
}  // namespace fluidk

// Advect nf scalar fields (packed nf*n in src) through the velocity field (grid units per step,
// scaled by dt). mode 0: semi-Lagrangian, 1: MacCormack with min/max limiter. Cells flagged in
// ``solid`` (may be null) keep zero. Returns 0.
AN_EXPORT int an_fluid_advect(const float *u, const float *v, const float *w, const int *dims, float dt,
                              const float *src, float *dst, int nf, int mode, const uint8_t *solid, int threads) {
  using namespace fluidk;
  Grid g{dims[0], dims[1], dims[2]};
  size_t n = g.n();
  std::vector<float> fwd;
  if (mode == 1) fwd.resize(n * nf);
  float *out1 = mode == 1 ? fwd.data() : dst;
  zfor(g.nz, threads, [&](int z0, int z1) {
    for (int z = z0; z < z1; z++)
      for (int y = 0; y < g.ny; y++)
        for (int x = 0; x < g.nx; x++) {
          size_t i = g.id(x, y, z);
          float px = x - dt * u[i], py = y - dt * v[i], pz = z - dt * w[i];
          for (int f = 0; f < nf; f++) out1[f * n + i] = samp(src + f * n, g, px, py, pz);
        }
  });
  if (mode == 1) {
    zfor(g.nz, threads, [&](int z0, int z1) {
      for (int z = z0; z < z1; z++)
        for (int y = 0; y < g.ny; y++)
          for (int x = 0; x < g.nx; x++) {
            size_t i = g.id(x, y, z);
            float px = x - dt * u[i], py = y - dt * v[i], pz = z - dt * w[i];
            float qx = x + dt * u[i], qy = y + dt * v[i], qz = z + dt * w[i];
            for (int f = 0; f < nf; f++) {
              const float *S = src + f * n, *Fw = fwd.data() + f * n;
              float back = samp(Fw, g, qx, qy, qz);
              float corr = Fw[i] + 0.5f * (S[i] - back);
              float lo, hi;
              samp_minmax(S, g, px, py, pz, lo, hi);
              dst[f * n + i] = std::min(std::max(corr, lo), hi);
            }
          }
    });
  }
  if (solid)
    for (size_t i = 0; i < n; i++)
      if (solid[i])
        for (int f = 0; f < nf; f++) dst[f * n + i] = 0.f;
  return 0;
}

// Vorticity confinement (in place, grid units): u += eps * (N x omega), N = grad|omega| / |.|
AN_EXPORT int an_fluid_vorticity(float *u, float *v, float *w, const int *dims, float eps, int threads) {
  using namespace fluidk;
  Grid g{dims[0], dims[1], dims[2]};
  size_t n = g.n();
  std::vector<float> om(3 * n, 0.f), mag(n, 0.f);
  size_t sy = g.nx, sz = (size_t)g.nx * g.ny;
  zfor(g.nz, threads, [&](int z0, int z1) {
    for (int z = z0; z < z1; z++)
      for (int y = 0; y < g.ny; y++)
        for (int x = 0; x < g.nx; x++) {
          size_t i = g.id(x, y, z);
          // central differences, zero on the faces the numpy _grad leaves at zero
          float uy = (y > 0 && y < g.ny - 1) ? 0.5f * (u[i + sy] - u[i - sy]) : 0.f;
          float uz = (z > 0 && z < g.nz - 1) ? 0.5f * (u[i + sz] - u[i - sz]) : 0.f;
          float vx = (x > 0 && x < g.nx - 1) ? 0.5f * (v[i + 1] - v[i - 1]) : 0.f;
          float vz = (z > 0 && z < g.nz - 1) ? 0.5f * (v[i + sz] - v[i - sz]) : 0.f;
          float wx = (x > 0 && x < g.nx - 1) ? 0.5f * (w[i + 1] - w[i - 1]) : 0.f;
          float wy = (y > 0 && y < g.ny - 1) ? 0.5f * (w[i + sy] - w[i - sy]) : 0.f;
          float ox = wy - vz, oy = uz - wx, oz = vx - uy;
          om[i] = ox; om[n + i] = oy; om[2 * n + i] = oz;
          mag[i] = std::sqrt(ox * ox + oy * oy + oz * oz);
        }
  });
  zfor(g.nz, threads, [&](int z0, int z1) {
    for (int z = z0; z < z1; z++)
      for (int y = 0; y < g.ny; y++)
        for (int x = 0; x < g.nx; x++) {
          size_t i = g.id(x, y, z);
          float gx = (x > 0 && x < g.nx - 1) ? 0.5f * (mag[i + 1] - mag[i - 1]) : 0.f;
          float gy = (y > 0 && y < g.ny - 1) ? 0.5f * (mag[i + sy] - mag[i - sy]) : 0.f;
          float gz = (z > 0 && z < g.nz - 1) ? 0.5f * (mag[i + sz] - mag[i - sz]) : 0.f;
          float gl = std::sqrt(gx * gx + gy * gy + gz * gz) + 1e-6f;
          float Nx = gx / gl, Ny = gy / gl, Nz = gz / gl;
          float ox = om[i], oy = om[n + i], oz = om[2 * n + i];
          u[i] += eps * (Ny * oz - Nz * oy);
          v[i] += eps * (Nz * ox - Nx * oz);
          w[i] += eps * (Nx * oy - Ny * ox);
        }
  });
  return 0;
}

// Pressure projection with solid obstacles (in place). Open (p = 0) domain faces, Neumann
// condition against cells flagged in ``solid`` whose velocity is ``svel`` (3*n, may be null =
// static). Red-black SOR with relaxation ``omega`` for ``iters`` sweeps, warm-started from p.
// ``floor`` != 0 makes the bottom face (y = 0) a wall. Returns 0.
AN_EXPORT int an_fluid_project(float *u, float *v, float *w, float *p, const int *dims, const uint8_t *solid,
                               const float *svel, int iters, float omega, int floor_, int threads) {
  using namespace fluidk;
  Grid g{dims[0], dims[1], dims[2]};
  size_t n = g.n();
  size_t sy = g.nx, sz = (size_t)g.nx * g.ny;
  auto S = [&](size_t i) -> bool { return solid && solid[i]; };
  // solid cells carry the obstacle velocity
  if (solid)
    for (size_t i = 0; i < n; i++)
      if (solid[i]) {
        u[i] = svel ? svel[i] : 0.f;
        v[i] = svel ? svel[n + i] : 0.f;
        w[i] = svel ? svel[2 * n + i] : 0.f;
      }
  std::vector<float> div(n, 0.f);
  std::vector<uint8_t> nb(n, 0);  // number of fluid neighbours (0 = not solved)
  zfor(g.nz, threads, [&](int z0, int z1) {
    for (int z = std::max(z0, 1); z < std::min(z1, g.nz - 1); z++)
      for (int y = 1; y < g.ny - 1; y++)
        for (int x = 1; x < g.nx - 1; x++) {
          size_t i = g.id(x, y, z);
          if (S(i)) continue;
          div[i] = 0.5f * (u[i + 1] - u[i - 1] + v[i + sy] - v[i - sy] + w[i + sz] - w[i - sz]);
          int k = 0;
          size_t o[6] = {i + 1, i - 1, i + sy, i - sy, i + sz, i - sz};
          for (int q = 0; q < 6; q++) k += !S(o[q]);
          nb[i] = (uint8_t)k;
        }
  });
  for (size_t i = 0; i < n; i++) p[i] = nb[i] ? p[i] * 0.9f : 0.f;
  int T = threads <= 0 ? (int)std::max(1u, std::thread::hardware_concurrency()) : threads;
  T = std::max(1, std::min(T, std::max(1, (g.nz - 2) / 2)));
  Barrier bar(T);
  auto worker = [&](int t) {
    int zi0 = 1 + (int)((long)(g.nz - 2) * t / T), zi1 = 1 + (int)((long)(g.nz - 2) * (t + 1) / T);
    for (int it = 0; it < iters; it++)
      for (int color = 0; color < 2; color++) {
        for (int z = zi0; z < zi1; z++)
          for (int y = 1; y < g.ny - 1; y++) {
            int xs = 1 + ((1 + y + z + color) & 1);
            for (int x = xs; x < g.nx - 1; x += 2) {
              size_t i = g.id(x, y, z);
              int k = nb[i];
              if (!k) continue;
              float s;
              if (k == 6) s = p[i + 1] + p[i - 1] + p[i + sy] + p[i - sy] + p[i + sz] + p[i - sz];
              else {
                s = 0;
                size_t o[6] = {i + 1, i - 1, i + sy, i - sy, i + sz, i - sz};
                for (int q = 0; q < 6; q++) if (!S(o[q])) s += p[o[q]];
              }
              float pn = (s - div[i]) / k;
              p[i] += omega * (pn - p[i]);
            }
          }
        if (T > 1) bar.wait();
      }
  };
  if (T == 1) worker(0);
  else {
    std::vector<std::thread> ts;
    for (int t = 1; t < T; t++) ts.emplace_back(worker, t);
    worker(0);
    for (auto &th : ts) th.join();
  }
  // subtract the pressure gradient (solid neighbours mirror the centre pressure)
  zfor(g.nz, threads, [&](int z0, int z1) {
    for (int z = z0; z < z1; z++)
      for (int y = 0; y < g.ny; y++)
        for (int x = 0; x < g.nx; x++) {
          size_t i = g.id(x, y, z);
          if (S(i)) continue;
          float pc = p[i];
          auto P = [&](size_t j) { return S(j) ? pc : p[j]; };
          if (x > 0 && x < g.nx - 1) u[i] -= 0.5f * (P(i + 1) - P(i - 1));
          if (y > 0 && y < g.ny - 1) v[i] -= 0.5f * (P(i + sy) - P(i - sy));
          if (z > 0 && z < g.nz - 1) w[i] -= 0.5f * (P(i + sz) - P(i - sz));
        }
  });
  if (floor_)
    for (int z = 0; z < g.nz; z++)
      for (int x = 0; x < g.nx; x++) { float &vv = v[g.id(x, 0, z)]; vv = std::max(vv, 0.f); }
  return 0;
}

// Signed distance + interpolated surface velocity of points against a moving triangle mesh.
// vvel (3*nv) is the per-vertex velocity; out_vel (3*np) receives the barycentric
// interpolation at the closest surface point. ``band`` is reserved (pass 0). Returns 0.
AN_EXPORT int an_mesh_sdf_vel(const float *verts, int nv, const int *tris, int nt, const float *vvel,
                              const float *pts, int np, float band, float *out_dist, float *out_vel, int threads) {
  if (nt <= 0 || nv <= 0) return 1;
  geomk::GBVH bvh;
  bvh.build(verts, nv, tris, nt);
  if (threads <= 0) threads = (int)std::max(1u, std::thread::hardware_concurrency());
  std::atomic<int> next(0);
  auto work = [&]() {
    using namespace geomk;
    for (;;) {
      int s = next.fetch_add(256);
      if (s >= np) break;
      int e = std::min(np, s + 256);
      for (int i = s; i < e; i++) {
        G3 p{pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]}, q{0, 0, 0};
        int t = -1;
        double d = bvh.closest(p, q, t);
        double wnd = bvh.winding(p, 2.0);
        out_dist[i] = (float)(wnd > 0.5 ? -d : d);
        float vx = 0, vy = 0, vz = 0;
        if (vvel && t >= 0) {
          int ia = tris[3 * t], ib = tris[3 * t + 1], ic = tris[3 * t + 2];
          G3 a = bvh.V[ia], b = bvh.V[ib], c = bvh.V[ic];
          G3 v0 = gsub(b, a), v1 = gsub(c, a), v2 = gsub(q, a);
          double d00 = gdot(v0, v0), d01 = gdot(v0, v1), d11 = gdot(v1, v1), d20 = gdot(v2, v0), d21 = gdot(v2, v1);
          double den = d00 * d11 - d01 * d01;
          double bv = 0, bw = 0;
          if (std::fabs(den) > 1e-30) { bv = (d11 * d20 - d01 * d21) / den; bw = (d00 * d21 - d01 * d20) / den; }
          double bu = 1 - bv - bw;
          vx = (float)(bu * vvel[3 * ia] + bv * vvel[3 * ib] + bw * vvel[3 * ic]);
          vy = (float)(bu * vvel[3 * ia + 1] + bv * vvel[3 * ib + 1] + bw * vvel[3 * ic + 1]);
          vz = (float)(bu * vvel[3 * ia + 2] + bv * vvel[3 * ib + 2] + bw * vvel[3 * ic + 2]);
        }
        if (out_vel) { out_vel[3 * i] = vx; out_vel[3 * i + 1] = vy; out_vel[3 * i + 2] = vz; }
      }
    }
  };
  (void)band;
  std::vector<std::thread> pool;
  for (int k = 0; k < threads - 1; k++) pool.emplace_back(work);
  work();
  for (auto &th : pool) th.join();
  return 0;
}

// Trilinear resample of nf coarse grids (cdims) stretched over a fine grid (fdims), scaled by
// ``scale`` and added to (add != 0) or written into ``dst`` (nf * fine cells). Returns 0.
AN_EXPORT int an_fluid_resample(const float *src, const int *cdims, int nf, float *dst, const int *fdims, float scale,
                                int add, int threads) {
  using namespace fluidk;
  Grid c{cdims[0], cdims[1], cdims[2]}, f{fdims[0], fdims[1], fdims[2]};
  size_t cn = c.n(), fn = f.n();
  float sx = (float)(c.nx - 1) / std::max(f.nx - 1, 1), sy = (float)(c.ny - 1) / std::max(f.ny - 1, 1),
        sz = (float)(c.nz - 1) / std::max(f.nz - 1, 1);
  zfor(f.nz, threads, [&](int z0, int z1) {
    for (int z = z0; z < z1; z++)
      for (int y = 0; y < f.ny; y++)
        for (int x = 0; x < f.nx; x++) {
          size_t i = f.id(x, y, z);
          for (int k = 0; k < nf; k++) {
            float v = scale * samp(src + k * cn, c, x * sx, y * sy, z * sz);
            dst[k * fn + i] = add ? dst[k * fn + i] + v : v;
          }
        }
  });
  return 0;
}
