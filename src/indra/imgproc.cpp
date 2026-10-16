// OpenCV-free image ops for the lite (Indra) pipeline. Each op reproduces the arithmetic of the
// OpenCV 5.0 code path the reference pipeline used (rounding, fixed point, FMA placement), so the
// native pipeline gives the same page results without cv2:
//   resize u8 C1: INTER_AREA (generic float tables / integer-factor fast path) and INTER_LINEAR
//                 (11-bit fixed point, VResizeLinearVec_32s8u rounding), incl. AREA-with-upscale.
//   warpAffine C1 linear: f32 (BORDER_CONSTANT) and u8 (BORDER_REPLICATE / CONSTANT); OpenCV 5
//                 float kernels: coords fma(M0,x,M_x), lerp fma(a,b-a,a), round-half-even.
//   connected components 4-connectivity with stats, labels in raster order of first pixel.
// Compiled with -ffp-contract=off: every FMA below is explicit.
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define IMG_API extern "C" __declspec(dllexport)
#else
#define IMG_API extern "C" __attribute__((visibility("default")))
#endif

namespace {
inline int rne(float v) { return (int)std::nearbyint(v); }  // default FE_TONEAREST: half to even
inline uint8_t sat_u8(int v) { return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
inline uint8_t sat_u8f(float v) { return sat_u8(rne(v)); }
inline int cv_floor(double v) { int i = (int)v; return i - (i > v); }
inline int cv_ceil(double v) { int i = (int)v; return i + (i < v); }
inline int sat_short(float v) { int i = rne(v); return i < -32768 ? -32768 : i > 32767 ? 32767 : i; }

#if defined(__aarch64__) || defined(_M_ARM64)
// 1: OpenCV built with clang: scalar a*b+c contracted to fma; 2: carotene HAL linear resize
constexpr int kDefFlags = 1 | 2 | 8 | 16 | 64 | 128 | 256;  // 8/16: carotene column/row positions fused; 64: KleidiCV 2x/4x; 128: KleidiCV generic downsize; 256: no left-edge weight clamp (OpenCV 5.0.0 carotene)
#else
constexpr int kDefFlags = 0;
#endif
// warp flags: 4 = M_x/M_y unfused, (1<<3) = scalar tail coords fma(x,M0,y*M1)+M2, 32 = tail lerp unfused
#if defined(__aarch64__) || defined(_M_ARM64)
constexpr int kWarpFlags = 8;   // clang build: M_x/M_y fused too
#else
constexpr int kWarpFlags = 12;
#endif

// ---------------------------------------------------------------- resize: area, generic
struct Dec { int di, si; float a; };
int area_tab(int ssize, int dsize, double scale, Dec* tab) {
  int k = 0;
  for (int dx = 0; dx < dsize; dx++) {
    double fsx1 = dx * scale, fsx2 = fsx1 + scale;
    double cw = std::min(scale, ssize - fsx1);
    int sx1 = cv_ceil(fsx1), sx2 = cv_floor(fsx2);
    sx2 = std::min(sx2, ssize - 1);
    sx1 = std::min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) tab[k++] = {dx, sx1 - 1, (float)((sx1 - fsx1) / cw)};
    for (int sx = sx1; sx < sx2; sx++) tab[k++] = {dx, sx, float(1.0 / cw)};
    if (fsx2 - sx2 > 1e-3) tab[k++] = {dx, sx2, (float)(std::min(std::min(fsx2 - sx2, 1.), cw) / cw)};
  }
  return k;
}

template <typename RowsFn>
void resize_area(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, double sx, double sy, int fl, RowsFn rows) {
  const bool fused = fl & 1;
  std::vector<Dec> xt(sw * 2 + 2), yt(sh * 2 + 2);
  int nx = area_tab(sw, dw, sx, xt.data()), ny = area_tab(sh, dh, sy, yt.data());
  std::vector<int> tabofs(dh + 1, ny);
  for (int k = 0, d = 0; k < ny; k++) if (k == 0 || yt[k].di != yt[k - 1].di) tabofs[d++] = k;
  // inter_area::mul/muladd: SIMD (4 lanes, unfused) while dx + 4 < width, scalar tail after
  int vend = 0;
  while (vend + 4 < dw) vend += 4;
  // horizontal taps re-laid tap-major and padded to the same count per output column (weight 0 adds exactly
  // nothing), so the per-column accumulation keeps its order while the loop runs across columns (SIMD)
  int T = 0;
  for (int k = 0, c = 0; k < nx; k++) { c = (k && xt[k].di == xt[k - 1].di) ? c + 1 : 1; T = std::max(T, c); }
  std::vector<int> tsi((size_t)T * dw); std::vector<float> ta((size_t)T * dw, 0.f);
  for (int k = 0, c = 0; k < nx; k++) {
    c = (k && xt[k].di == xt[k - 1].di) ? c + 1 : 0;
    if (c == 0) for (int t = 0; t < T; t++) tsi[(size_t)t * dw + xt[k].di] = xt[k].si;
    tsi[(size_t)c * dw + xt[k].di] = xt[k].si; ta[(size_t)c * dw + xt[k].di] = xt[k].a;
  }
  rows(dh, [&](int d0, int d1) {
    std::vector<float> buf(dw), sum(dw, 0.f);
    auto store = [&](int dy) {
      uint8_t* D = dst + (size_t)dy * dw;
      for (int i = 0; i < dw; i++) D[i] = sat_u8f(sum[i]);
    };
    int j0 = tabofs[d0], j1 = tabofs[d1];
    if (j0 >= j1) return;
    int prev = yt[j0].di;
    for (int j = j0; j < j1; j++) {
      float beta = yt[j].a; int dy = yt[j].di; const uint8_t* S = src + (size_t)yt[j].si * sw;
      float* B = buf.data();
      {
        const int* I0 = tsi.data(); const float* A0 = ta.data();
        for (int i = 0; i < dw; i++) B[i] = (float)S[I0[i]] * A0[i];  // = 0 + S*a, fused or not
      }
      for (int t = 1; t < T; t++) {
        const int* It = tsi.data() + (size_t)t * dw; const float* At = ta.data() + (size_t)t * dw;
        if (fused) for (int i = 0; i < dw; i++) B[i] = std::fma((float)S[It[i]], At[i], B[i]);
        else for (int i = 0; i < dw; i++) B[i] = B[i] + (float)S[It[i]] * At[i];
      }
      if (dy != prev) {
        store(prev);
        for (int i = 0; i < dw; i++) sum[i] = beta * buf[i];
        prev = dy;
      } else {
        int i = 0;
        for (; i < vend; i++) sum[i] = sum[i] + beta * buf[i];
        if (fused) for (; i < dw; i++) sum[i] = std::fma(beta, buf[i], sum[i]);
        else for (; i < dw; i++) sum[i] = sum[i] + beta * buf[i];
      }
    }
    store(prev);
  });
}

// integer factors (resizeAreaFast_); factor 2 uses (sum + 2) >> 2
void resize_area_fast(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, int isx, int isy) {
  int area = isx * isy; float scale = 1.f / area;
  int dwidth1 = sw / isx;
  bool two = isx == 2 && isy == 2;
  for (int dy = 0; dy < dh; dy++) {
    uint8_t* D = dst + (size_t)dy * dw;
    int sy0 = dy * isy;
    int w = sy0 + isy <= sh ? dwidth1 : 0;
    if (sy0 >= sh) { std::memset(D, 0, dw); continue; }
    int dx = 0;
    for (; dx < w; dx++) {
      int s = 0;
      for (int a = 0; a < isy; a++) for (int b = 0; b < isx; b++) s += src[(size_t)(sy0 + a) * sw + dx * isx + b];
      D[dx] = two ? (uint8_t)((s + 2) >> 2) : sat_u8f(s * scale);
    }
    for (; dx < dw; dx++) {
      int s = 0, cnt = 0, sx0 = dx * isx;
      for (int a = 0; a < isy && sy0 + a < sh; a++)
        for (int b = 0; b < isx && sx0 + b < sw; b++) { s += src[(size_t)(sy0 + a) * sw + sx0 + b]; cnt++; }
      D[dx] = cnt ? sat_u8f((float)s / cnt) : 0;
    }
  }
}

// ---------------------------------------------------------------- resize: linear, 8U fixed point
template <typename RowsFn>
void resize_linear(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, double inv_x, double inv_y, bool area_mode, int fl, RowsFn rows) {
  const bool fused = fl & 1;  // clang build: (dx+0.5)*scale-0.5 and (dx+1)-(sx+1)*inv contracted
  const int ONE = 2048;
  double scale_x = 1. / inv_x, scale_y = 1. / inv_y;
  std::vector<int> xofs(dw), yofs(dh);
  std::vector<short> ia(dw * 2), ib(dh * 2);
  int xmax = dw;
  for (int dx = 0; dx < dw; dx++) {
    float fx; int sx;
    if (!area_mode) { fx = (float)(fused ? std::fma(dx + 0.5, scale_x, -0.5) : (dx + 0.5) * scale_x - 0.5); sx = cv_floor(fx); fx -= sx; }
    else { sx = cv_floor(dx * scale_x); fx = (float)(fused ? std::fma(-(double)(sx + 1), inv_x, (double)(dx + 1)) : (dx + 1) - (sx + 1) * inv_x); fx = fx <= 0 ? 0.f : fx - cv_floor(fx); }
    if (sx < 0) fx = 0, sx = 0;
    if (sx + 1 >= sw) { xmax = std::min(xmax, dx); if (sx >= sw - 1) fx = 0, sx = sw - 1; }
    xofs[dx] = sx;
    ia[dx * 2] = (short)sat_short((1.f - fx) * ONE); ia[dx * 2 + 1] = (short)sat_short(fx * ONE);
  }
  for (int dy = 0; dy < dh; dy++) {
    float fy; int sy;
    if (!area_mode) { fy = (float)(fused ? std::fma(dy + 0.5, scale_y, -0.5) : (dy + 0.5) * scale_y - 0.5); sy = cv_floor(fy); fy -= sy; }
    else { sy = cv_floor(dy * scale_y); fy = (float)(fused ? std::fma(-(double)(sy + 1), inv_y, (double)(dy + 1)) : (dy + 1) - (sy + 1) * inv_y); fy = fy <= 0 ? 0.f : fy - cv_floor(fy); }
    yofs[dy] = sy;
    ib[dy * 2] = (short)sat_short((1.f - fy) * ONE); ib[dy * 2 + 1] = (short)sat_short(fy * ONE);
  }
  auto clip = [](int x, int a, int b) { return x >= a ? (x < b ? x : b - 1) : a; };
  rows(dh, [&](int d0, int d1) {
    std::vector<int> r0(dw), r1(dw);
    int p0 = -1, p1 = -1;
    auto hres = [&](int sy, int* D) {
      const uint8_t* S = src + (size_t)sy * sw;
      int dx = 0;
      for (; dx < xmax; dx++) D[dx] = S[xofs[dx]] * ia[dx * 2] + S[xofs[dx] + 1] * ia[dx * 2 + 1];
      for (; dx < dw; dx++) D[dx] = S[xofs[dx]] * ONE;
    };
    for (int dy = d0; dy < d1; dy++) {
      int s0 = clip(yofs[dy], 0, sh), s1 = clip(yofs[dy] + 1, 0, sh);
      if (s0 != p0) { if (s0 == p1) r0.swap(r1), std::swap(p0, p1); else hres(s0, r0.data()), p0 = s0; }
      if (s1 != p1) { hres(s1, r1.data()); p1 = s1; }
      int b0 = ib[dy * 2], b1 = ib[dy * 2 + 1];
      uint8_t* D = dst + (size_t)dy * dw;
      for (int x = 0; x < dw; x++)
        D[x] = sat_u8((((b0 * (r0[x] >> 4)) >> 16) + ((b1 * (r1[x] >> 4)) >> 16) + 2) >> 2);
    }
  });
}

// KleidiCV (outer ARM HAL) INTER_LINEAR u8 for exact 2x / 4x upscales: exact bilinear sum, round half up.
template <typename RowsFn>
void resize_linear_kleidicv(const uint8_t* src, int sw, int sh, uint8_t* dst, int f, RowsFn rows) {
  int dw = sw * f, dh = sh * f, den = 2 * f;  // tap weights are k / (2f)
  auto taps = [&](int n, int N, std::vector<int>& t0, std::vector<int>& t1, std::vector<int>& w1) {
    t0.resize(N); t1.resize(N); w1.resize(N);
    for (int i = 0; i < N; i++) {
      int num = 2 * i + 1 - f;                  // fx * 2f = (i + 0.5) * 2 - f
      int s0 = num >= 0 ? num / den : -((-num + den - 1) / den);
      int fr = num - s0 * den;
      t0[i] = std::min(std::max(s0, 0), n - 1); t1[i] = std::min(std::max(s0 + 1, 0), n - 1); w1[i] = fr;
    }
  };
  std::vector<int> x0, x1, wx, y0, y1, wy;
  taps(sw, dw, x0, x1, wx); taps(sh, dh, y0, y1, wy);
  const int D = den * den;
  rows(dh, [&](int d0, int d1) {
    for (int y = d0; y < d1; y++) {
      const uint8_t *a = src + (size_t)y0[y] * sw, *b = src + (size_t)y1[y] * sw;
      int vy1 = wy[y], vy0 = den - vy1;
      uint8_t* o = dst + (size_t)y * dw;
      for (int x = 0; x < dw; x++) {
        int vx1 = wx[x], vx0 = den - vx1;
        int sum = vy0 * (vx0 * a[x0[x]] + vx1 * a[x1[x]]) + vy1 * (vx0 * b[x0[x]] + vx1 * b[x1[x]]);
        o[x] = (uint8_t)((sum + D / 2) / D);
      }
    }
  });
}

// aarch64 OpenCV build, INTER_LINEAR u8 with 1 < scale <= 2 (src width >= 16): fitted on M4 against cv2 (exact on
// probes): double-precision source coords, 8-bit weights rounded half up, rows first (rounded to u8), then columns.
static void down_coords(int ns, int nd, std::vector<int>& x0, std::vector<int>& x1, std::vector<int>& w) {
  x0.resize(nd); x1.resize(nd); w.resize(nd);
  for (int i = 0; i < nd; i++) {
    double x = (i + 0.5) * ns / nd - 0.5;
    long a = (long)std::floor(x); int ww = (int)std::floor((x - a) * 256 + 0.5);
    if (a < 0) { a = 0; ww = 0; }
    if (a >= ns - 1) { a = ns - 1; ww = 0; }
    x0[i] = (int)a; x1[i] = (int)std::min<long>(a + 1, ns - 1); w[i] = ww;
  }
}
template <typename RowsFn>
void resize_down_carotene(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, float, float, int, RowsFn rows) {
  std::vector<int> cx0, cx1, cw, ry0, ry1, rw;
  down_coords(sw, dw, cx0, cx1, cw); down_coords(sh, dh, ry0, ry1, rw);
  rows(dh, [&](int d0, int d1) {
    std::vector<uint8_t> t(sw);
    for (int r = d0; r < d1; r++) {
      const uint8_t *A = src + (size_t)ry0[r] * sw, *B = src + (size_t)ry1[r] * sw;
      int wy = rw[r];
      for (int c = 0; c < sw; c++) t[c] = (uint8_t)((A[c] * (256 - wy) + B[c] * wy + 128) >> 8);
      uint8_t* o = dst + (size_t)r * dw;
      for (int i = 0; i < dw; i++) o[i] = (uint8_t)((t[cx0[i]] * (256 - cw[i]) + t[cx1[i]] * cw[i] + 128) >> 8);
    }
  });
}


// KleidiCV 26.03 (Apache-2.0, Arm) generic u8 linear downsize, 1 channel, horizontal ratio 1/3..1 with both
// axes shrinking: what OpenCV 5.0's aarch64 build (KleidiCV HAL ahead of carotene) runs for those sizes.
// Scalar transcription of kleidicv/src/resize/resize_linear_generic_u8_neon.h (neon backend, 16-byte
// vectors): 16.16 fixed-point source coordinates with periodic re-calibration, 8-bit fractions, vqtbl
// lookups (out-of-table index -> 0), vraddhn rounding, and the same pull-back of the last vectors.
namespace kcv {
inline uint64_t rdiv(uint64_t n, uint64_t d) { return (n + d / 2) / d; }
inline uint64_t aligned_scale(uint64_t x, uint64_t nom, uint64_t den) {
  return rdiv(((x << 16) + 32768) * nom, den) - 32768 + (1 << 7);
}
inline uint8_t lerp(uint8_t a, uint8_t b, unsigned w) {  // vraddhn_u16(a << 8, (b - a) * w)
  unsigned t = ((unsigned)a << 8) + (((unsigned)(uint16_t)(b - a) * w) & 0xFFFF);
  return (uint8_t)(((t & 0xFFFF) + 128) >> 8);
}
struct Vec { int n; uint8_t idx[16]; uint16_t frac[16]; int64_t src; int64_t dst; };
}  // namespace kcv

bool kleidicv_down_supported(int sw, int sh, int dw, int dh) {
  if (!(dw * 3 >= sw && dw < sw && dh < sh) || dw < 8) return false;
  if (sw == 2 * dw && sh == 2 * dh) return false;  // resize_to_quarter (same result as INTER_AREA 2x)
  return dw * 2 >= sw ? sw >= 16 : sw >= 32;
}

template <typename RowsFn>
void resize_down_kleidicv(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, RowsFn rows) {
  using namespace kcv;
  const int ratio = dw * 2 >= sw ? 2 : 3;
  const uint64_t SW = (uint64_t)sw, DW = (uint64_t)dw;
  uint32_t lane0[16];
  for (int k = 0; k < 16; k++) lane0[k] = (uint32_t)rdiv((uint64_t)k * SW << 16, DW);
  std::vector<Vec> vecs;
  auto fill = [&](int n, uint64_t sx_base, uint64_t sx_fixp, int64_t dstpos) {
    Vec v{}; v.n = n;
    uint32_t xf0 = (uint32_t)(sx_fixp - (sx_base << 16));
    for (int k = 0; k < n; k++) {
      uint32_t d = lane0[k] + xf0;
      v.idx[k] = (uint8_t)(d >> 16); v.frac[k] = (uint16_t)((d >> 8) & 0xFF);
    }
    v.src = (int64_t)sx_base; v.dst = dstpos;
    vecs.push_back(v);
  };
  uint64_t two_x = SW >= (uint64_t)(16 * ratio) ? DW / 32 : 0;
  uint64_t dx = 0, sx = 0;
  const uint64_t max_sx_2x = SW - 16 * ratio, step2 = rdiv((SW * 16) << 16, DW);
  for (uint64_t i = 0; i < two_x; i++) {
    if ((i & 255) == 0) sx = aligned_scale(dx, SW, DW);
    for (int h = 0; h < 2; h++) {
      fill(16, std::min(max_sx_2x, sx >> 16), sx, (int64_t)dx);
      sx += step2; dx += 16;
    }
  }
  uint64_t rem = DW - two_x * 32, half = (rem + 7) / 8;
  sx = aligned_scale(dx, SW, DW);
  const uint64_t one = rdiv(SW << 16, DW), max_sx_half = SW - 16 * (ratio - 1), max_dx_half = DW - 8,
                 steph = rdiv((SW * 8) << 16, DW);
  for (uint64_t i = 0; i < half; i++) {
    uint64_t dxp = std::min(dx, max_dx_half);
    sx -= (dx - dxp) * one; dx = dxp;
    fill(8, std::min(max_sx_half, sx >> 16), sx, (int64_t)dx);
    dx += 8; sx += steph;
  }
  rows(dh, [&](int d0, int d1) {
    for (int dy = d0; dy < d1; dy++) {
      uint64_t sy_fixp = aligned_scale((uint64_t)dy, (uint64_t)sh, (uint64_t)dh);
      int64_t sy = (int64_t)(sy_fixp >> 16);
      unsigned yf = (unsigned)((sy_fixp - ((uint64_t)sy << 16)) >> 8);
      const uint8_t *top = src + (size_t)sy * sw, *bot = src + (size_t)(sy + 1) * sw;
      uint8_t* out = dst + (size_t)dy * dw;
      for (const Vec& v : vecs) {
        // table bytes available to vqtbl: full vectors 16*ratio, half vectors 16*(ratio-1)
        int tbl = v.n == 16 ? 16 * ratio : 16 * (ratio - 1);
        auto at = [&](const uint8_t* r, unsigned i) -> uint8_t { return i < (unsigned)tbl ? r[v.src + i] : 0; };
        for (int k = 0; k < v.n; k++) {
          unsigned i0 = v.idx[k], i1 = (uint8_t)(i0 + 1);
          uint8_t a = at(top, i0), b = at(top, i1), c = at(bot, i0), d = at(bot, i1);
          out[v.dst + k] = lerp(lerp(a, c, yf), lerp(b, d, yf), v.frac[k]);
        }
      }
    }
  });
}


// carotene downsample_bilinear_8uc1 (OpenCV hal/carotene/src/resize.cpp, BSD): taken for INTER_LINEAR u8 when
// not both scales <= 1, both <= 2 and src width >= 16 (and KleidiCV declined). 11-bit weights, "ugly
// version matching OpenCV's SSE": >>4 narrow, vqdmulh by the row weight, halve, add, vqrshrun by 2.
template <typename RowsFn>
void resize_downsample_carotene(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, float wr, float hr, int fl, RowsFn rows) {
  const bool fuse = (fl & 1) && !(fl & 512), clamp_left = !(fl & 256);
  float ox = 0.5f * wr - 0.5f, oy = 0.5f * hr - 0.5f;
  auto s16 = [](int64_t x) { return (int16_t)(uint16_t)(uint64_t)x; };
  auto qdmulh = [](int a, int b) { int64_t p = (2 * (int64_t)a * b) >> 16; return (int)std::min<int64_t>(32767, std::max<int64_t>(-32768, p)); };
  std::vector<int> c1(dw), c2(dw); std::vector<int16_t> cw(dw);
  for (int i = 0; i < dw; i++) {
    float c = fuse ? std::fma((float)i, wr, ox) : (float)i * wr + ox;
    long a = (long)c, b = a + 1;
    int16_t w = (int16_t)std::floor(((float)b - c) * 2048.0f + 0.5f);
    if (clamp_left && c < 0.f) w = 2048;
    if (a < 0) a = 0;
    if (b >= sw) b = sw - 1;
    c1[i] = (int)a; c2[i] = (int)b; cw[i] = w;
  }
  rows(dh, [&](int d0, int d1) {
    for (int r = d0; r < d1; r++) {
      float rr = fuse ? std::fma((float)r, hr, oy) : (float)r * hr + oy;
      long s0 = (long)std::floor(rr), s1 = s0 + 1;
      int rw = (int)std::floor(((float)s1 - rr) * 2048.0f + 0.5f);
      const uint8_t* A = src + (size_t)std::max<long>(0, s0) * sw;
      const uint8_t* B = src + (size_t)std::min<long>(sh - 1, s1) * sw;
      int vrw = s16(rw), vrW = s16(2048 - rw);
      uint8_t* o = dst + (size_t)r * dw;
      for (int i = 0; i < dw; i++) {
        auto part = [&](const uint8_t* R) {
          int l = R[c1[i]], h = R[c2[i]];
          int32_t v = (int32_t)((uint32_t)h << 11) + (int32_t)s16(l - h) * cw[i];
          return (int)s16(v >> 4);
        };
        int v1 = qdmulh(part(A), vrw), v2 = qdmulh(part(B), vrW);
        int sum = s16((v1 >> 1) + (v2 >> 1));
        int q = (sum + 2) >> 2;
        o[i] = (uint8_t)std::min(255, std::max(0, q));
      }
    }
  });
}

// carotene (ARM HAL of OpenCV builds for aarch64) resizeLinearOpenCVchan<1>: 7-bit weights, rows first (rounded
// to u8), then columns. fl & 8: column positions vmlaq_f32 fused; fl & 16: row positions fused.
template <typename RowsFn>
void resize_linear_carotene(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, float wr, float hr, int fl, RowsFn rows) {
  float offx = 0.5f * wr - 0.5f, offy = 0.5f * hr - 0.5f;
  std::vector<int> cl(dw), ch(dw); std::vector<uint8_t> cwt(dw);
  for (int i = 0; i < dw; i++) {
    float fi = (float)i;
    float vw = (fl & 8) ? std::fma(wr, fi, offx) : offx + wr * fi;
    int vwi = (int)vw;                         // vcvtq_s32_f32: toward zero
    int hi = vwi - (((float)vwi < vw) ? -1 : 0);
    int lo = hi - 1;
    float w2 = ((float)hi - vw) * 128.0f;
    uint32_t wu = w2 <= 0.f ? 0u : (uint32_t)w2;  // vcvtq_u32_f32: toward zero, saturating
    cwt[i] = (uint8_t)(uint16_t)wu;               // vmovn twice
    hi = std::min(hi, sw - 1); lo = std::min(std::max(lo, 0), sw - 1);
    ch[i] = hi; cl[i] = lo;
  }
  rows(dh, [&](int d0, int d1) {
    std::vector<uint8_t> t(sw);
    for (int r = d0; r < d1; r++) {
      float fr = (float)r;
      float w = (fl & 16) ? std::fma(fr, hr, offy) : fr * hr + offy;
      long r0 = (long)std::floor(w), r1 = r0 + 1;
      uint8_t rw = (uint8_t)(int)(((float)r1 - w) * 128);
      if (r0 < 0) r0 = 0;
      if (r1 >= sh) r1 = sh - 1;
      const uint8_t *a = src + (size_t)r0 * sw, *b = src + (size_t)r1 * sw;
      for (int c = 0; c < sw; c++) t[c] = (uint8_t)((a[c] * rw + b[c] * (128 - rw) + 64) >> 7);
      uint8_t* D = dst + (size_t)r * dw;
      for (int i = 0; i < dw; i++) D[i] = (uint8_t)((t[cl[i]] * cwt[i] + t[ch[i]] * (128 - cwt[i]) + 64) >> 7);
    }
  });
}

}  // namespace

#define IMG_NS img_base
#include "imgproc_warp.inc"
#undef IMG_NS
#if defined(__x86_64__) && !defined(_MSC_VER)
namespace img_avx2 {
void warp_f32(const float*, int, int, float*, int, int, int, int, const double*, bool, float, int, int);
void warp_u8(const uint8_t*, int, int, uint8_t*, int, int, int, int, const double*, bool, float, int, int);
}
#endif
namespace {
int g_threads = 1;
thread_local int t_threads = 0;  // > 0: per-thread override (callers that already run ops in parallel)
// rows [0, dh) split over up to g_threads threads (row results do not depend on the split)
template <typename F>
void for_rows(int dh, size_t work, F f) {
  int nt = std::min(t_threads > 0 ? t_threads : g_threads, std::max(1, (int)(work / 65536)));
  nt = std::min(nt, dh);
  if (nt <= 1) { f(0, dh); return; }
  std::vector<std::thread> th;
  for (int i = 1; i < nt; i++) th.emplace_back(f, (int)((int64_t)dh * i / nt), (int)((int64_t)dh * (i + 1) / nt));
  f(0, dh / nt);
  for (auto& t : th) t.join();
}
}  // namespace

// Image-op threads: at least the engine's thread count; by default all cores up to 8 (as OpenCV did for
// the reference pipeline). PPOCR_IMG_THREADS overrides.
IMG_API void ppocr_img_set_threads(int n) {
  int hw = (int)std::thread::hardware_concurrency();
  int t = std::max(n, std::min(8, hw > 0 ? hw : 1));
  if (const char* e = std::getenv("PPOCR_IMG_THREADS")) t = std::atoi(e);
  g_threads = t < 1 ? 1 : t;
}

// Thread count for image ops issued from the calling thread only (0 = use ppocr_img_set_threads).
IMG_API void ppocr_img_set_local_threads(int n) { t_threads = n < 0 ? 0 : n; }
IMG_API int ppocr_img_get_threads(void) { return g_threads; }

// interp: 1 = INTER_LINEAR, 3 = INTER_AREA (cv2 constants). flags: -1 = platform default.
IMG_API int ppocr_img_resize_u8(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, int interp, int flags) {
  if (!src || !dst || sw < 1 || sh < 1 || dw < 1 || dh < 1) return -1;
  if (flags < 0) flags = kDefFlags;
  if (sw == dw && sh == dh) { std::memcpy(dst, src, (size_t)sw * sh); return 0; }
  double inv_x = (double)dw / sw, inv_y = (double)dh / sh;
  size_t work = (size_t)sw * sh + (size_t)dw * dh;
  auto rows = [&](int n, const std::function<void(int, int)>& f) { for_rows(n, work, f); };
  // OpenCV calls the HALs first: on aarch64 builds KleidiCV (exact 2x / 4x), then carotene take INTER_LINEAR u8
  if (interp == 1 && (flags & 64) && sh > 0 && ((dw == 2 * sw && dh == 2 * sh) || (dw == 4 * sw && dh == 4 * sh))) {
    resize_linear_kleidicv(src, sw, sh, dst, dw / sw, rows);
    return 0;
  }
  // KleidiCV HAL (flags & 128) takes both-axes linear downsizes with horizontal ratio 1/3..1 first
  if (interp == 1 && (flags & 128) && kleidicv_down_supported(sw, sh, dw, dh)) {
    resize_down_kleidicv(src, sw, sh, dst, dw, dh, rows);
    return 0;
  }
  if (interp == 1 && (flags & 2) && sw >= 8 && dw >= 8 && dh >= 8 &&
      (dw - 0.5) / inv_x - 0.5 < sw && (dh - 0.5) / inv_y - 0.5 < sh &&
      (dw + 0.5) / inv_x + 0.5 >= sw && (dh + 0.5) / inv_y + 0.5 >= sh &&
      std::abs(dw / inv_x - sw) < 0.1 && std::abs(dh / inv_y - sh) < 0.1) {
    float wr = (float)(1.0 / inv_x), hr = (float)(1.0 / inv_y);
    bool small = wr <= 2.f && hr <= 2.f && sw >= 16;
    if ((wr <= 1.f && hr <= 1.f) || !small) { resize_linear_carotene(src, sw, sh, dst, dw, dh, wr, hr, flags, rows); return 0; }
    if (flags & 128) { resize_downsample_carotene(src, sw, sh, dst, dw, dh, wr, hr, flags, rows); return 0; }
    if (wr > 1.f && hr > 1.f) { resize_down_carotene(src, sw, sh, dst, dw, dh, wr, hr, flags, rows); return 0; }
    // mixed (one axis up, the other 1-2x down): falls through to OpenCV's generic path (measured on M4)
  }
  double scale_x = 1. / inv_x, scale_y = 1. / inv_y;
  int isx = (int)std::lrint(scale_x), isy = (int)std::lrint(scale_y);
  bool fast = std::abs(scale_x - isx) < 2.220446049250313e-16 && std::abs(scale_y - isy) < 2.220446049250313e-16;
  if (interp == 1 && fast && isx == 2 && isy == 2) interp = 3;
  if (interp == 3 && scale_x >= 1 && scale_y >= 1) {
    if (fast) resize_area_fast(src, sw, sh, dst, dw, dh, isx, isy);
    else resize_area(src, sw, sh, dst, dw, dh, scale_x, scale_y, flags, rows);
    return 0;
  }
  if (interp != 1 && interp != 3) return -2;
  resize_linear(src, sw, sh, dst, dw, dh, inv_x, inv_y, interp == 3, flags, rows);
  return 0;
}

// M: forward 2x3 matrix (as passed to cv2.warpAffine without WARP_INVERSE_MAP); border 0 = constant, 1 = replicate
IMG_API int ppocr_img_warp_affine(const void* src, int sw, int sh, void* dst, int dw, int dh, const double* M0,
                                  int is_float, int border, double bval, int uf, int flags) {
  if (!src || !dst || sw < 1 || sh < 1) return -1;
  if (flags < 0) flags = kWarpFlags;
  if (uf <= 0) {
#if defined(__aarch64__) || defined(_M_ARM64)
    uf = 8;  // NEON: 2 x 4 floats per block
#elif defined(__x86_64__) || defined(__i386__)
    // OpenCV dispatches the warp kernels to AVX2 (16 floats per block) when the CPU has it, else SSE4.1 (8)
    uf = (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? 16 : 8;
#else
    uf = 8;
#endif
    if (const char* e = std::getenv("PPOCR_IMG_WARP_UF")) uf = std::atoi(e);
  }
  double M[6]; std::memcpy(M, M0, sizeof M);
  double D = M[0] * M[4] - M[1] * M[3];
  D = D != 0 ? 1. / D : 0;
  double A11 = M[4] * D, A22 = M[0] * D;
  M[0] = A11; M[1] *= -D; M[3] *= -D; M[4] = A22;
  double b1 = -M[0] * M[2] - M[1] * M[5], b2 = -M[3] * M[2] - M[4] * M[5];
  M[2] = b1; M[5] = b2;
  bool rep = border == 1;
  float bv = is_float ? (float)bval : (float)sat_u8f((float)bval);
  auto wf = img_base::warp_f32; auto wu = img_base::warp_u8;
#if defined(__x86_64__) && !defined(_MSC_VER)
  if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && !std::getenv("PPOCR_IMG_NO_AVX2"))
    wf = img_avx2::warp_f32, wu = img_avx2::warp_u8;
#endif
  for_rows(dh, (size_t)dw * dh, [&](int y0, int y1) {
    if (is_float) wf((const float*)src, sw, sh, (float*)dst, dw, dh, y0, y1, M, rep, bv, uf, flags);
    else wu((const uint8_t*)src, sw, sh, (uint8_t*)dst, dw, dh, y0, y1, M, rep, bv, uf, flags);
  });
  return 0;
}

// cv2.getRotationMatrix2D (Point2f center); flags & 1: clang build contracts the translation terms
IMG_API void ppocr_img_rotation_matrix(double cx, double cy, double angle, double scale, int flags, double* M) {
  if (flags < 0) flags = kDefFlags;
  float fcx = (float)cx, fcy = (float)cy;
  angle *= 3.14159265358979323846 / 180;
  double alpha = std::cos(angle) * scale, beta = std::sin(angle) * scale;
  M[0] = alpha; M[1] = beta; M[3] = -beta; M[4] = alpha;
  if (flags & 1) {
    M[2] = std::fma(1 - alpha, (double)fcx, -(beta * fcy));
    M[5] = std::fma(beta, (double)fcx, (1 - alpha) * fcy);
  } else {
    M[2] = (1 - alpha) * fcx - beta * fcy;
    M[5] = beta * fcx + (1 - alpha) * fcy;
  }
}

// cv2.cvtColor(COLOR_RGB2GRAY) for packed uint8 RGB (15-bit fixed point)
IMG_API void ppocr_img_rgb2gray(const uint8_t* rgb, int w, int h, int stride, uint8_t* dst) {
  for_rows(h, (size_t)w * h, [&](int y0, int y1) {
    for (int y = y0; y < y1; y++) {
      const uint8_t* s = rgb + (size_t)y * stride; uint8_t* d = dst + (size_t)y * w;
      for (int x = 0; x < w; x++) d[x] = (uint8_t)((s[3 * x] * 9798 + s[3 * x + 1] * 19235 + s[3 * x + 2] * 3735 + 16384) >> 15);
    }
  });
}

// 4-connectivity components of a 0/1 mask. labels [h*w] int32; stats [n*5] (left, top, width, height, area),
// written for up to max_n labels (incl. background 0). Returns n (labels incl. background).
IMG_API int ppocr_img_cc4(const uint8_t* mask, int w, int h, int32_t* labels, int32_t* stats, int max_n) {
  std::vector<int32_t> par(1, 0);
  auto find = [&](int32_t a) { while (par[a] != a) { par[a] = par[par[a]]; a = par[a]; } return a; };
  for (int y = 0; y < h; y++) {
    const uint8_t* m = mask + (size_t)y * w; int32_t* L = labels + (size_t)y * w;
    const int32_t* U = y ? L - w : nullptr;
    for (int x = 0; x < w; x++) {
      if (!m[x]) { L[x] = 0; continue; }
      int32_t a = x ? L[x - 1] : 0, b = U ? U[x] : 0;
      if (!a && !b) { L[x] = (int32_t)par.size(); par.push_back(L[x]); }
      else if (a && b) {
        int32_t ra = find(a), rb = find(b);
        if (ra != rb) { if (ra < rb) par[rb] = ra; else par[ra] = rb; }
        L[x] = std::min(ra, rb);
      } else L[x] = a ? a : b;
    }
  }
  // roots are the smallest provisional label of each set = first raster pixel; number them in order
  std::vector<int32_t> fin(par.size(), 0);
  int n = 1;
  for (size_t i = 1; i < par.size(); i++) { int32_t r = find((int32_t)i); fin[i] = r == (int32_t)i ? n++ : fin[r]; }
  std::vector<int32_t> st((size_t)n * 5);
  for (int i = 0; i < n; i++) { st[i * 5] = w; st[i * 5 + 1] = h; st[i * 5 + 2] = -1; st[i * 5 + 3] = -1; st[i * 5 + 4] = 0; }
  for (int y = 0; y < h; y++) {
    int32_t* L = labels + (size_t)y * w;
    for (int x = 0; x < w; x++) {
      int32_t l = L[x] = fin[L[x]];
      int32_t* s = &st[(size_t)l * 5];
      s[0] = std::min(s[0], x); s[1] = std::min(s[1], y); s[2] = std::max(s[2], x); s[3] = std::max(s[3], y); s[4]++;
    }
  }
  if (stats) for (int i = 0; i < std::min(n, max_n); i++) {
    int32_t* s = &st[(size_t)i * 5];
    stats[i * 5] = s[0]; stats[i * 5 + 1] = s[1]; stats[i * 5 + 2] = s[2] - s[0] + 1; stats[i * 5 + 3] = s[3] - s[1] + 1; stats[i * 5 + 4] = s[4];
  }
  return n;
}
