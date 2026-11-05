// Animora GPU renderer host (Apple Metal).
//
// Builds the scene acceleration structures and light tables with the exact same CPU Engine code
// (pathtracer.cpp is compiled into this library), uploads them into shared-memory Metal buffers and
// runs the integrator kernel in gpu_kernel.msl.  The pass schedule, adaptive sampling test and
// AOV resolve mirror an_render() so CPU and GPU renders converge to the same image.
//
// Exports:
//   int         an_gpu_available()         1 when a Metal device exists
//   const char *an_gpu_device_name()
//   int         an_gpu_compile()           0 ok, -3 no device, -4 kernel compile failure
//   const char *an_gpu_last_error()
//   int         an_gpu_render(...)         same signature/semantics as an_render; returns
//                                          -3 when the scene needs the CPU (volumes / no device),
//                                          -4 when the kernel failed to compile or run.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "pathtracer.cpp"

#include <chrono>
#include <mutex>
#include <string>

static const char *kSrc =
#include "gpu_kernel.msl"
    ;

// Must mirror `struct Params` in gpu_kernel.msl field-for-field (all 4-byte scalars).
struct GParams {
  int32_t width, height, max_depth, integrator;
  int32_t caustics, rr_depth, primary_only, pass_spp;
  float clamp_indirect, ao_distance, wire_width, filter_sigma;
  uint32_t seed; int32_t rx0, ry0, rw;
  int32_t rh, num_tris, motion, cam_motion;
  int32_t has_N0, has_N1, has_UV, has_PREF;
  int32_t has_tri_mat, num_materials, num_textures, num_lights;
  int32_t env_tex, env_visible, num_entries, env_entry;
  int32_t env_w, env_h, env_valid, has_texdata;
  float env_strength, env_rotation, env_integral, scene_radius;
  float env_bg[3]; float fog_density;
  float fog_color[3]; float fog_anisotropy;
  float fog_height_min, fog_height_max, pad_a, pad_b;
  AnCamera camera;
};

static std::mutex g_mu;
static int g_state = 0;  // 0 = not initialised, 1 = ready, -3 = no device, -4 = compile failure
static id<MTLDevice> g_dev = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLComputePipelineState> g_pso = nil;
static std::string g_err;
static std::string g_name;

static int gpu_init_locked() {
  if (g_state != 0) return g_state;
  @autoreleasepool {
    g_dev = MTLCreateSystemDefaultDevice();
    if (!g_dev) { g_err = "no Metal device"; g_state = -3; return g_state; }
    g_name = [[g_dev name] UTF8String];
    MTLCompileOptions *opt = [MTLCompileOptions new];
    const char *fm = getenv("ANIMORA_GPU_FASTMATH");
    bool fast = fm && fm[0] == '1';
    if (@available(macOS 15.0, *)) opt.mathMode = fast ? MTLMathModeRelaxed : MTLMathModeSafe;
    else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      opt.fastMathEnabled = fast ? YES : NO;
#pragma clang diagnostic pop
    }
    NSError *err = nil;
    id<MTLLibrary> lib = [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:opt error:&err];
    if (!lib) {
      g_err = err ? [[err localizedDescription] UTF8String] : "unknown Metal compile error";
      g_state = -4;
      return g_state;
    }
    id<MTLFunction> fn = [lib newFunctionWithName:@"render"];
    if (!fn) { g_err = "kernel 'render' not found"; g_state = -4; return g_state; }
    g_pso = [g_dev newComputePipelineStateWithFunction:fn error:&err];
    if (!g_pso) {
      g_err = err ? [[err localizedDescription] UTF8String] : "pipeline creation failed";
      g_state = -4;
      return g_state;
    }
    g_queue = [g_dev newCommandQueue];
    if (getenv("ANIMORA_GPU_DEBUG"))
      fprintf(stderr, "[animora-gpu] %s: simd width %lu, max threads/group %lu\n", g_name.c_str(),
              (unsigned long)[g_pso threadExecutionWidth], (unsigned long)[g_pso maxTotalThreadsPerThreadgroup]);
    g_state = 1;
  }
  return g_state;
}

AN_EXPORT int an_gpu_available() {
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    return d ? 1 : 0;
  }
}

AN_EXPORT const char *an_gpu_device_name() {
  std::lock_guard<std::mutex> lk(g_mu);
  if (g_name.empty()) {
    @autoreleasepool {
      id<MTLDevice> d = MTLCreateSystemDefaultDevice();
      if (d) g_name = [[d name] UTF8String];
    }
  }
  return g_name.c_str();
}

AN_EXPORT int an_gpu_compile() {
  std::lock_guard<std::mutex> lk(g_mu);
  return gpu_init_locked();
}

AN_EXPORT const char *an_gpu_last_error() { return g_err.c_str(); }

static id<MTLBuffer> make_buf(const void *p, size_t bytes) {
  static const uint8_t zeros[256] = {0};
  if (!p || bytes == 0) return [g_dev newBufferWithBytes:zeros length:sizeof(zeros) options:MTLResourceStorageModeShared];
  return [g_dev newBufferWithBytes:p length:bytes options:MTLResourceStorageModeShared];
}

static size_t texdata_floats(const AnScene *S) {
  size_t n = 0;
  for (int i = 0; i < S->num_textures; ++i) {
    const AnTexture &t = S->textures[i];
    if (t.type != TEX_IMAGE || t.width <= 0 || t.height <= 0 || t.offset < 0) continue;
    size_t end = (size_t)t.offset + (size_t)t.width * t.height * std::max(1, t.channels);
    n = std::max(n, end);
  }
  return n;
}

AN_EXPORT int an_gpu_render(const AnScene *scene, const AnSettings *settings, float *out_rgba, float *out_albedo,
                            float *out_normal, float *out_depth, float *out_variance, int32_t *out_spp,
                            AnProgress progress) {
  if (!scene || !settings || !out_rgba) return -1;
  if (scene->num_volumes > 0) { g_err = "volumes are rendered on the CPU"; return -3; }
  std::lock_guard<std::mutex> lk(g_mu);
  int st = gpu_init_locked();
  if (st != 1) return st;

  const int W = settings->width, H = settings->height;
  if (W <= 0 || H <= 0) return -2;
  int rx0 = 0, ry0 = 0, rx1 = W, ry1 = H;
  if (settings->region_x1 > settings->region_x0 && settings->region_y1 > settings->region_y0) {
    rx0 = std::max(0, settings->region_x0); ry0 = std::max(0, settings->region_y0);
    rx1 = std::min(W, settings->region_x1); ry1 = std::min(H, settings->region_y1);
  }
  const int rw = rx1 - rx0, rh = ry1 - ry0;
  if (rw <= 0 || rh <= 0) return -2;
  const size_t RN = (size_t)rw * rh;

  Engine eng;
  eng.S = scene;
  eng.ST = settings;
  eng.setup();

  int ret = 0;
  @autoreleasepool {
    const int nt = eng.nt;
    // ---------------------------------------------------------------- scene buffers
    id<MTLBuffer> bNodes = make_buf(eng.nodes.data(), eng.nodes.size() * sizeof(BVHNode));
    id<MTLBuffer> bIdx = make_buf(eng.idx.data(), eng.idx.size() * sizeof(int32_t));
    id<MTLBuffer> bP0 = make_buf(scene->P0, (size_t)nt * 9 * sizeof(float));
    id<MTLBuffer> bP1 = make_buf(eng.motion ? scene->P1 : nullptr, (size_t)nt * 9 * sizeof(float));
    id<MTLBuffer> bN0 = make_buf(scene->N0, (size_t)nt * 9 * sizeof(float));
    id<MTLBuffer> bN1 = make_buf(scene->N1, (size_t)nt * 9 * sizeof(float));
    id<MTLBuffer> bUV = make_buf(scene->UV, (size_t)nt * 6 * sizeof(float));
    id<MTLBuffer> bPREF = make_buf(scene->PREF, (size_t)nt * 9 * sizeof(float));
    id<MTLBuffer> bTriMat = make_buf(scene->tri_mat, (size_t)nt * sizeof(int32_t));
    id<MTLBuffer> bOpaque = make_buf(eng.tri_opaque.data(), eng.tri_opaque.size());
    std::vector<AnMaterial> mats(scene->materials, scene->materials + std::max(0, scene->num_materials));
    mats.push_back(eng.default_mat);
    id<MTLBuffer> bMats = make_buf(mats.data(), mats.size() * sizeof(AnMaterial));
    id<MTLBuffer> bTexs = make_buf(scene->textures, (size_t)std::max(0, scene->num_textures) * sizeof(AnTexture));
    size_t ntd = texdata_floats(scene);
    id<MTLBuffer> bTexdata = make_buf(ntd ? scene->texdata : nullptr, ntd * sizeof(float));
    id<MTLBuffer> bLights = make_buf(scene->lights, (size_t)std::max(0, scene->num_lights) * sizeof(AnLight));
    id<MTLBuffer> bEntries = make_buf(eng.lights.data(), eng.lights.size() * sizeof(LightEntry));
    id<MTLBuffer> bCdf = make_buf(eng.light_cdf.data(), eng.light_cdf.size() * sizeof(float));
    id<MTLBuffer> bProb = make_buf(eng.light_prob.data(), eng.light_prob.size() * sizeof(float));
    id<MTLBuffer> bTriLight = make_buf(eng.tri_light.data(), eng.tri_light.size() * sizeof(int32_t));
    id<MTLBuffer> bEnvFunc = make_buf(eng.env.func.data(), eng.env.func.size() * sizeof(float));
    id<MTLBuffer> bEnvCond = make_buf(eng.env.cond.data(), eng.env.cond.size() * sizeof(float));
    id<MTLBuffer> bEnvMarg = make_buf(eng.env.marg.data(), eng.env.marg.size() * sizeof(float));
    id<MTLBuffer> bPerm = make_buf(perlin_table().p, 512 * sizeof(int));
    id<MTLBuffer> bAccum = [g_dev newBufferWithLength:RN * 12 * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bActive = [g_dev newBufferWithLength:std::max<size_t>(RN, 16) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bCounts = [g_dev newBufferWithLength:RN * sizeof(int32_t) options:MTLResourceStorageModeShared];
    if (!bAccum || !bActive || !bCounts) { g_err = "out of GPU memory"; return -4; }
    float *acc = (float *)[bAccum contents];
    uint8_t *act = (uint8_t *)[bActive contents];
    int32_t *cnt = (int32_t *)[bCounts contents];
    std::memset(acc, 0, RN * 12 * sizeof(float));
    std::memset(cnt, 0, RN * sizeof(int32_t));

    // ---------------------------------------------------------------- params
    GParams P;
    std::memset(&P, 0, sizeof(P));
    P.width = W; P.height = H;
    P.max_depth = settings->max_depth;
    P.integrator = settings->integrator;
    P.caustics = settings->caustics;
    P.rr_depth = settings->rr_depth;
    P.primary_only = (settings->integrator >= INT_NORMALS && settings->integrator <= INT_UV) ? 1 : 0;
    P.clamp_indirect = settings->clamp_indirect;
    P.ao_distance = settings->ao_distance;
    P.wire_width = settings->wire_width;
    P.filter_sigma = std::max(0.f, settings->filter_width) * 0.35f;
    P.seed = (uint32_t)settings->seed;
    P.rx0 = rx0; P.ry0 = ry0; P.rw = rw; P.rh = rh;
    P.num_tris = nt;
    P.motion = eng.motion ? 1 : 0;
    P.cam_motion = scene->has_motion ? 1 : 0;
    P.has_N0 = scene->N0 != nullptr; P.has_N1 = scene->N1 != nullptr;
    P.has_UV = scene->UV != nullptr; P.has_PREF = scene->PREF != nullptr;
    P.has_tri_mat = scene->tri_mat != nullptr;
    P.num_materials = std::max(0, scene->num_materials);
    P.num_textures = std::max(0, scene->num_textures);
    P.num_lights = std::max(0, scene->num_lights);
    P.env_tex = scene->env_tex; P.env_visible = scene->env_visible;
    P.num_entries = (int32_t)eng.lights.size();
    P.env_entry = eng.env_entry;
    P.env_w = eng.env.w; P.env_h = eng.env.h; P.env_valid = eng.env.valid ? 1 : 0;
    P.has_texdata = ntd > 0;
    P.env_strength = scene->env_strength; P.env_rotation = scene->env_rotation;
    P.env_integral = eng.env.integral; P.scene_radius = eng.scene_radius;
    for (int k = 0; k < 3; ++k) { P.env_bg[k] = scene->env_bg[k]; P.fog_color[k] = scene->fog_color[k]; }
    P.fog_density = scene->fog_density; P.fog_anisotropy = scene->fog_anisotropy;
    P.fog_height_min = scene->fog_height_min; P.fog_height_max = scene->fog_height_max;
    P.camera = scene->camera;

    // ---------------------------------------------------------------- schedule (mirrors an_render)
    const int TS = settings->tile_size > 0 ? settings->tile_size : 16;
    const int tx = (rw + TS - 1) / TS, ty = (rh + TS - 1) / TS, ntiles = tx * ty;
    std::vector<uint8_t> tile_active(ntiles, 1);
    const int spp = std::max(1, settings->spp);
    const bool adaptive = settings->adaptive_threshold > 0 && spp > 8;
    const int min_spp = adaptive ? std::max(4, std::min(spp, settings->min_spp > 0 ? settings->min_spp : std::max(8, spp / 8))) : spp;
    std::vector<int> passes;
    {
      int first = adaptive ? min_spp : std::max(1, std::min(spp, std::max(1, spp / 8)));
      passes.push_back(first);
      int done = first;
      int step = std::max(1, adaptive ? std::max(4, spp / 16) : std::max(1, spp / 8));
      while (done < spp) { int s = std::min(step, spp - done); passes.push_back(s); done += s; }
    }

    const NSUInteger tw = [g_pso threadExecutionWidth];
    const NSUInteger th = std::max<NSUInteger>(1, [g_pso maxTotalThreadsPerThreadgroup] / tw);
    NSUInteger tgy = std::min<NSUInteger>(th, 8);
    if (const char *e = getenv("ANIMORA_GPU_TGY")) tgy = std::max(1, std::min((int)th, atoi(e)));
    const MTLSize tg = MTLSizeMake(tw, tgy, 1);
    const double target = getenv("ANIMORA_GPU_TARGET") ? atof(getenv("ANIMORA_GPU_TARGET")) : 0.08;  // seconds per command buffer (keeps the display responsive / no watchdog)
    double sec_per_ps = -1;      // measured seconds per pixel-sample
    bool cancelled = false, failed = false;
    const double total_work = (double)spp * RN;
    double work_done = 0;

    auto dispatch = [&](int y0, int bh, int nspp) -> double {
      GParams Q = P;
      Q.ry0 = ry0 + y0; Q.rh = bh; Q.pass_spp = nspp;
      id<MTLCommandBuffer> cb = [g_queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:g_pso];
      [enc setBytes:&Q length:sizeof(Q) atIndex:0];
      id<MTLBuffer> bufs[22] = {bNodes, bIdx, bP0, bP1, bN0, bN1, bUV, bPREF, bTriMat, bOpaque, bMats,
                                bTexs, bTexdata, bLights, bEntries, bCdf, bProb, bTriLight, bEnvFunc, bEnvCond,
                                bEnvMarg, bPerm};
      for (int i = 0; i < 22; ++i) [enc setBuffer:bufs[i] offset:0 atIndex:i + 1];
      size_t off = (size_t)y0 * rw;
      [enc setBuffer:bAccum offset:off * 12 * sizeof(float) atIndex:23];
      [enc setBuffer:bActive offset:off atIndex:24];
      [enc setBuffer:bCounts offset:off * sizeof(int32_t) atIndex:25];
      [enc dispatchThreads:MTLSizeMake(rw, bh, 1) threadsPerThreadgroup:tg];
      [enc endEncoding];
      auto t0 = std::chrono::steady_clock::now();
      [cb commit];
      [cb waitUntilCompleted];
      double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if ([cb status] == MTLCommandBufferStatusError) {
        NSError *e = [cb error];
        g_err = e ? [[e localizedDescription] UTF8String] : "GPU command buffer failed";
        failed = true;
      }
      return dt;
    };

    for (size_t pi = 0; pi < passes.size() && !cancelled && !failed; ++pi) {
      const int pass_spp = passes[pi];
      // active mask
      int nact = 0;
      for (int y = 0; y < rh; ++y)
        for (int x = 0; x < rw; ++x) {
          uint8_t a = tile_active[(y / TS) * tx + (x / TS)];
          act[(size_t)y * rw + x] = a;
          nact += a;
        }
      if (nact == 0) break;
      // work out band height / spp chunk from the measured throughput
      int y = 0;
      while (y < rh && !cancelled && !failed) {
        int bh, chunk;
        if (sec_per_ps <= 0) {
          bh = std::min(rh, std::max(4, (int)(262144 / std::max(1, rw)) & ~3));
          chunk = 1;
        } else {
          double budget = target / sec_per_ps;  // pixel-samples per command buffer
          double rows_pix = (double)rw * (rh - y);
          if (budget >= rows_pix) {
            bh = rh - y;
            chunk = std::max(1, std::min(pass_spp, (int)(budget / rows_pix)));
          } else {
            bh = std::max(4, (int)(budget / rw) & ~3);
            bh = std::min(bh, rh - y);
            chunk = 1;
          }
        }
        bh = std::min(bh, rh - y);
        // all samples of this pass for the band
        int done = 0;
        while (done < pass_spp && !cancelled && !failed) {
          int n = std::min(chunk, pass_spp - done);
          double dt = dispatch(y, bh, n);
          double ps = (double)rw * bh * n;
          double rate = dt / std::max(1.0, ps);
          sec_per_ps = sec_per_ps <= 0 ? rate : 0.5 * sec_per_ps + 0.5 * rate;
          done += n;
          work_done += ps;
          if (progress && progress((float)std::min(1.0, work_done / total_work))) cancelled = true;
          // re-plan chunk inside the band
          if (bh == rh - y) chunk = std::max(1, std::min(pass_spp, (int)(target / sec_per_ps / ((double)rw * bh))));
        }
        y += bh;
      }
      // inactive pixels count as done for progress
      work_done += (double)(RN - nact) * pass_spp;

      // adaptive convergence test (same metric as an_render)
      if (adaptive && pi + 1 < passes.size() && !cancelled && !failed) {
        for (int t = 0; t < ntiles; ++t) {
          if (!tile_active[t]) continue;
          int x0 = (t % tx) * TS, y0 = (t / tx) * TS;
          int x1 = std::min(rw, x0 + TS), y1 = std::min(rh, y0 + TS);
          float worst = 0;
          for (int yy = y0; yy < y1; ++yy)
            for (int xx = x0; xx < x1; ++xx) {
              size_t p = (size_t)yy * rw + xx;
              int n = cnt[p];
              if (n < 2) { worst = INF; continue; }
              const float *a = acc + p * 12;
              float mean = lum(V3(a[0], a[1], a[2])) / n;
              float var = std::max(0.f, (a[3] / n - mean * mean)) * n / (n - 1);
              float err = std::sqrt(var / n) / (std::sqrt(std::max(mean, 0.f)) + 0.05f);
              worst = std::max(worst, err);
            }
          if (worst < settings->adaptive_threshold) tile_active[t] = 0;
        }
      }
    }
    if (failed) return -4;
    if (progress) progress(1.f);

    // ---------------------------------------------------------------- resolve (mirrors an_render)
    const size_t N = (size_t)W * H;
    std::memset(out_rgba, 0, N * 4 * sizeof(float));
    if (out_albedo) std::memset(out_albedo, 0, N * 3 * sizeof(float));
    if (out_normal) std::memset(out_normal, 0, N * 3 * sizeof(float));
    if (out_depth) std::memset(out_depth, 0, N * sizeof(float));
    if (out_variance) std::memset(out_variance, 0, N * sizeof(float));
    if (out_spp) std::memset(out_spp, 0, N * sizeof(int32_t));
    for (int yy = 0; yy < rh; ++yy)
      for (int xx = 0; xx < rw; ++xx) {
        size_t lp = (size_t)yy * rw + xx;
        size_t p = (size_t)(ry0 + yy) * W + (rx0 + xx);
        const float *a = acc + lp * 12;
        int n = cnt[lp];
        float inv = n > 0 ? 1.f / n : 0.f;
        float r = a[0] * inv, g = a[1] * inv, b = a[2] * inv;
        out_rgba[p * 4] = r; out_rgba[p * 4 + 1] = g; out_rgba[p * 4 + 2] = b;
        out_rgba[p * 4 + 3] = a[4] * inv;
        if (out_albedo) for (int k = 0; k < 3; ++k) out_albedo[p * 3 + k] = a[5 + k] * inv;
        if (out_normal) {
          V3 nn(a[8], a[9], a[10]);
          float l = length(nn);
          if (l > 1e-6f) nn = nn / l;
          out_normal[p * 3] = nn.x; out_normal[p * 3 + 1] = nn.y; out_normal[p * 3 + 2] = nn.z;
        }
        if (out_depth) out_depth[p] = a[11] * inv;
        if (out_variance) {
          float mean = lum(V3(r, g, b));
          out_variance[p] = n > 1 ? std::max(0.f, (a[3] * inv - mean * mean)) * n / (n - 1) / n : 0.f;
        }
        if (out_spp) out_spp[p] = n;
      }
    ret = cancelled ? 1 : 0;
  }
  return ret;
}
