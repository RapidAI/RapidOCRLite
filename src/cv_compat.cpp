// See cv_compat.hpp. This translation unit is compiled with the strict FP
// policy of db_geometry.cpp (no fast-math; FMA contraction exactly as the
// OpenCV wheels: none on x86, "on" for Clang/arm64), because the coordinate
// math below must round exactly like OpenCV's own build.
#include "cv_compat.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#define PPOCR_CVC_NEON_FP16 1
#endif

#if defined(PPOCR_HAS_AVX2_KERNELS)
#include "kernels.hpp"
#endif

namespace rpocr::detail::cvc {
namespace {

inline int Floor(double v) {
  const int i = static_cast<int>(v);
  return i - (static_cast<double>(i) > v);
}
inline int FloorF(float v) {
  const int i = static_cast<int>(v);
  return i - (static_cast<float>(i) > v);
}
// saturate_cast<short>(float): cvRound (nearest-even) then clamp.
inline short SatShort(float v) {
  const int i = static_cast<int>(std::nearbyint(v));
  return static_cast<short>(std::clamp(i, -32768, 32767));
}

void ResizeAreaFast2x(const std::uint8_t* src, std::size_t sstep, int cn, std::uint8_t* dst,
                      int dw, int dh, std::size_t dstep) {
  // resizeAreaFast_<uchar,int>: (a + b + c + d + 2) >> 2 over each 2x2 cell.
  for (int y = 0; y < dh; ++y) {
    const std::uint8_t* s0 = src + std::size_t(2 * y) * sstep;
    const std::uint8_t* s1 = s0 + sstep;
    std::uint8_t* d = dst + std::size_t(y) * dstep;
    for (int x = 0; x < dw; ++x) {
      for (int c = 0; c < cn; ++c) {
        const int i = 2 * x * cn + c;
        d[x * cn + c] =
            static_cast<std::uint8_t>((s0[i] + s0[i + cn] + s1[i] + s1[i + cn] + 2) >> 2);
      }
    }
  }
}

}  // namespace

#if defined(__APPLE__) && defined(__aarch64__)
// macOS arm64 OpenCV wheels route 3-channel downscales (ratio up to 3 in x,
// both axes shrinking) through the KleidiCV 26.03 HAL instead of OpenCV's own
// resize. Its NEON kernel (resize_linear_generic_u8_neon.h) uses 16.16 fixed
// point source coordinates, built per 16/8-lane vector block, and an 8-bit
// fraction lerp. Reproduce the exact coordinate schedule and arithmetic.
namespace {
inline std::uint64_t RDiv(std::uint64_t n, std::uint64_t d) { return (n + d / 2) / d; }

bool KleidiResize3(const std::uint8_t* src, int sw_, int sh_, std::size_t sstep,
                   std::uint8_t* dst, int dw_, int dh_, std::size_t dstep) {
  const std::uint64_t sw = sw_, sh = sh_, dw = dw_, dh = dh_;
  constexpr std::uint64_t C = 3;
  if (!(dw * 3 >= sw && dw < sw && dh < sh)) return false;
  if (dw * C < 8 || sw * C < 32) return false;
  const double inverse_scale = double(sw) / double(dw);
  const std::uint64_t R = inverse_scale < 1.8 ? 2 : 3;
  const auto scale_x = [&](std::uint64_t k) {
    return static_cast<std::uint32_t>(RDiv((k * sw) << 16, dw));
  };
  const auto aligned = [](std::uint64_t x, std::uint64_t nom, std::uint64_t den) {
    return RDiv(((x << 16) + 32768) * nom, den) - 32768 + 128;
  };
  const auto to_src_x = [&](std::uint64_t dx) { return aligned(dx, sw, dw); };
  const std::uint64_t one = RDiv(sw << 16, dw);
  const std::uint64_t total = dw * C;
  std::vector<std::uint64_t> X(total);
  const std::uint64_t two_x = (sw * C >= 16 * R) ? total / 32 : 0;
  const std::uint64_t rem = total - two_x * 32;
  const std::uint64_t half = (rem + 7) / 8;
  const std::uint64_t nfull = two_x * 2;
  auto fill = [&](std::uint64_t elem, unsigned n, std::uint64_t s) {
    const unsigned inp = static_cast<unsigned>(elem % C);
    unsigned j = 0;
    for (; j < C - inp; ++j) X[elem + j] = s;
    s += one;
    while (j < n) {
      for (unsigned k = 0; j < n && k < C; ++j, ++k) X[elem + j] = s;
      s += one;
    }
  };
  std::uint64_t elem = 0;
  if (nfull > 0) {
    std::uint64_t handled = 0;
    if (nfull > 3) {
      auto needs_pullback = [&](std::uint64_t e) {
        const std::uint64_t sx = to_src_x(e / C);
        return (sx >> 16) * C + e % C + 16 * R > sw * C;
      };
      std::uint64_t cand = nfull - 1;
      do {
        if (!needs_pullback(cand * 16)) break;
        --cand;
      } while (cand > 0);
      std::uint64_t nwp = cand + 1;
      if (cand == 0 && needs_pullback(0)) nwp = 0;
      static constexpr int off[3][16] = {
          {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5},
          {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5},
          {0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5}};
      const std::uint64_t five = RDiv((sw * 5) << 16, dw), six = RDiv((sw * 6) << 16, dw);
      std::uint64_t sx = to_src_x(0);
      unsigned cnt = 0;
      for (std::uint64_t t = 0; t < nwp / 3; ++t) {
        if (cnt == 170) {
          sx = to_src_x(elem / 3);
          cnt = 0;
        } else {
          ++cnt;
        }
        for (int b = 0; b < 3; ++b) {
          const std::uint64_t frac = sx & 0xffff, base = sx - frac;
          for (int j = 0; j < 16; ++j)
            X[elem + b * 16 + j] = base + std::uint32_t(scale_x(off[b][j]) + std::uint32_t(frac));
          sx += b == 2 ? six : five;
        }
        handled += 3;
        elem += 48;
      }
    }
    for (; handled < nfull; ++handled, elem += 16) fill(elem, 16, to_src_x(elem / C));
  }
  const std::uint64_t max_half = total - 8;
  for (std::uint64_t i = 0; i < half; ++i, elem += 8) {
    elem = std::min(elem, max_half);
    fill(elem, 8, to_src_x(elem / C));
  }
  std::vector<int> idx(total), xf(total);
  for (std::uint64_t e = 0; e < total; ++e) {
    idx[e] = static_cast<int>((X[e] >> 16) * C + e % C);
    xf[e] = static_cast<int>((X[e] >> 8) & 0xff);
  }
  for (std::uint64_t dy = 0; dy < dh; ++dy) {
    const std::uint64_t sy_fixp = aligned(dy, sh, dh);
    const std::uint64_t sy = sy_fixp >> 16;
    const int yf = static_cast<int>((sy_fixp - (sy << 16)) >> 8);
    const std::uint8_t* top = src + sy * sstep;
    const std::uint8_t* bot = top + sstep;
    std::uint8_t* d = dst + dy * dstep;
    for (std::uint64_t e = 0; e < total; ++e) {
      const int i = idx[e];
      const int a = top[i], b = top[i + 3], c = bot[i], dd = bot[i + 3];
      const int left = (a * 256 + (c - a) * yf + 128) >> 8;
      const int right = (b * 256 + (dd - b) * yf + 128) >> 8;
      d[e] = static_cast<std::uint8_t>((left * 256 + (right - left) * xf[e] + 128) >> 8);
    }
  }
  return true;
}
}  // namespace
#endif

void ResizeLinear8u(const std::uint8_t* src, int sw, int sh, std::size_t sstep, int cn,
                    std::uint8_t* dst, int dw, int dh, std::size_t dstep) {
  if (sw == dw && sh == dh) {
    for (int y = 0; y < sh; ++y)
      std::memcpy(dst + std::size_t(y) * dstep, src + std::size_t(y) * sstep,
                  std::size_t(sw) * cn);
    return;
  }
#if defined(__APPLE__) && defined(__aarch64__)
  if (cn == 3 && KleidiResize3(src, sw, sh, sstep, dst, dw, dh, dstep)) return;
#endif
  const double inv_scale_x = static_cast<double>(dw) / sw;
  const double inv_scale_y = static_cast<double>(dh) / sh;
  const double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;
  const int iscale_x = static_cast<int>(std::nearbyint(scale_x));
  const int iscale_y = static_cast<int>(std::nearbyint(scale_y));
  const bool area_fast = std::abs(scale_x - iscale_x) < 2.220446049250313e-16 &&
                         std::abs(scale_y - iscale_y) < 2.220446049250313e-16;
  if (area_fast && iscale_x == 2 && iscale_y == 2) {
    ResizeAreaFast2x(src, sstep, cn, dst, dw, dh, dstep);
    return;
  }
  constexpr int kOne = 2048;  // INTER_RESIZE_COEF_SCALE
  int xmax = dw;
  std::vector<int> xofs(static_cast<std::size_t>(dw));
  std::vector<short> ialpha(static_cast<std::size_t>(dw) * 2);
  for (int dx = 0; dx < dw; ++dx) {
    float fx = static_cast<float>((dx + 0.5) * scale_x - 0.5);
    int sx = FloorF(fx);
    fx -= sx;
    if (sx < 0) fx = 0, sx = 0;
    if (sx + 1 >= sw) {
      xmax = std::min(xmax, dx);
      if (sx >= sw - 1) fx = 0, sx = sw - 1;
    }
    xofs[dx] = sx * cn;
    ialpha[dx * 2] = SatShort((1.f - fx) * kOne);
    ialpha[dx * 2 + 1] = SatShort(fx * kOne);
  }
  const int width = dw * cn;
  std::vector<int> rows(std::size_t(width) * 2);
  int* buf[2] = {rows.data(), rows.data() + width};
  int tag[2] = {-1, -1};
  auto hresize = [&](int sy, int* d) {
    const std::uint8_t* s = src + std::size_t(sy) * sstep;
    for (int dx = 0; dx < xmax; ++dx) {
      const int sx = xofs[dx];
      const int a0 = ialpha[dx * 2], a1 = ialpha[dx * 2 + 1];
      for (int c = 0; c < cn; ++c) d[dx * cn + c] = s[sx + c] * a0 + s[sx + c + cn] * a1;
    }
    for (int dx = xmax; dx < dw; ++dx) {
      const int sx = xofs[dx];
      for (int c = 0; c < cn; ++c) d[dx * cn + c] = s[sx + c] * kOne;
    }
  };
  auto row = [&](int sy, const int* keep) -> const int* {
    for (int b = 0; b < 2; ++b)
      if (tag[b] == sy) return buf[b];
    const int b = buf[0] == keep ? 1 : 0;
    hresize(sy, buf[b]);
    tag[b] = sy;
    return buf[b];
  };
  const auto clip = [sh](int v) { return v >= 0 ? (v < sh ? v : sh - 1) : 0; };
  for (int dy = 0; dy < dh; ++dy) {
    float fy = static_cast<float>((dy + 0.5) * scale_y - 0.5);
    const int sy = FloorF(fy);
    fy -= sy;
    const int b0 = SatShort((1.f - fy) * kOne), b1 = SatShort(fy * kOne);
    const int* r0 = row(clip(sy), nullptr);
    const int* r1 = row(clip(sy + 1), r0);
    std::uint8_t* d = dst + std::size_t(dy) * dstep;
    for (int x = 0; x < width; ++x)
      d[x] = static_cast<std::uint8_t>(
          (((b0 * (r0[x] >> 4)) >> 16) + ((b1 * (r1[x] >> 4)) >> 16) + 2) >> 2);
  }
}

namespace {

// hal::LU64f (LUImpl<double>) on an 8x8 system with one right-hand side.
bool Lu8(double a[8][8], double b[8]) {
  constexpr double eps = 2.220446049250313e-16 * 100;
  for (int i = 0; i < 8; ++i) {
    int k = i;
    for (int j = i + 1; j < 8; ++j)
      if (std::abs(a[j][i]) > std::abs(a[k][i])) k = j;
    if (std::abs(a[k][i]) < eps) return false;
    if (k != i) {
      for (int j = i; j < 8; ++j) std::swap(a[i][j], a[k][j]);
      std::swap(b[i], b[k]);
    }
    const double d = -1 / a[i][i];
    for (int j = i + 1; j < 8; ++j) {
      const double alpha = a[j][i] * d;
      for (int c = i + 1; c < 8; ++c) a[j][c] += alpha * a[i][c];
      b[j] += alpha * b[i];
    }
  }
  for (int i = 7; i >= 0; --i) {
    double s = b[i];
    for (int c = i + 1; c < 8; ++c) s -= a[i][c] * b[c];
    b[i] = s / a[i][i];
  }
  return true;
}

// cv::getPerspectiveTransform(src, dst, DECOMP_LU). The SVD fallback for a
// failed LU residual check is never reached for text boxes (measured
// residual <= 2e-12 against the 1e-8 limit), so the LU result is used.
void PerspectiveTransform(const float src[8], const float dst[8], double m[9]) {
  double a[8][8], b[8];
  for (int i = 0; i < 4; ++i) {
    const float sx = src[2 * i], sy = src[2 * i + 1];
    const float dx = dst[2 * i], dy = dst[2 * i + 1];
    a[i][0] = a[i + 4][3] = sx;
    a[i][1] = a[i + 4][4] = sy;
    a[i][2] = a[i + 4][5] = 1;
    a[i][3] = a[i][4] = a[i][5] = a[i + 4][0] = a[i + 4][1] = a[i + 4][2] = 0;
    a[i][6] = -sx * dx;
    a[i][7] = -sy * dx;
    a[i + 4][6] = -sx * dy;
    a[i + 4][7] = -sy * dy;
    b[i] = dx;
    b[i + 4] = dy;
  }
  if (!Lu8(a, b)) {
    for (int i = 0; i < 9; ++i) m[i] = i % 4 == 0 ? 1 : 0;
    return;
  }
  for (int i = 0; i < 8; ++i) m[i] = b[i];
  m[8] = 1.;
}

// cv::invert(DECOMP_LU) closed form for 3x3 doubles.
bool Invert3(const double s[9], double d[9]) {
#define S(y, x) s[(y) * 3 + (x)]
  double det = S(0, 0) * (S(1, 1) * S(2, 2) - S(1, 2) * S(2, 1)) -
               S(0, 1) * (S(1, 0) * S(2, 2) - S(1, 2) * S(2, 0)) +
               S(0, 2) * (S(1, 0) * S(2, 1) - S(1, 1) * S(2, 0));
  if (det == 0.) return false;
  det = 1. / det;
  double t[9];
  t[0] = (S(1, 1) * S(2, 2) - S(1, 2) * S(2, 1)) * det;
  t[1] = (S(0, 2) * S(2, 1) - S(0, 1) * S(2, 2)) * det;
  t[2] = (S(0, 1) * S(1, 2) - S(0, 2) * S(1, 1)) * det;
  t[3] = (S(1, 2) * S(2, 0) - S(1, 0) * S(2, 2)) * det;
  t[4] = (S(0, 0) * S(2, 2) - S(0, 2) * S(2, 0)) * det;
  t[5] = (S(0, 2) * S(1, 0) - S(0, 0) * S(1, 2)) * det;
  t[6] = (S(1, 0) * S(2, 1) - S(1, 1) * S(2, 0)) * det;
  t[7] = (S(0, 1) * S(2, 0) - S(0, 0) * S(2, 1)) * det;
  t[8] = (S(0, 0) * S(1, 1) - S(0, 1) * S(1, 0)) * det;
#undef S
  std::memcpy(d, t, sizeof t);
  return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// bicubic8uC3 (warp_kernels.simd.hpp). Every output pixel depends only on its
// own source coordinate; batching, the inlier fast path and the tail overlap
// in OpenCV do not change any value under BORDER_REPLICATE. Arithmetic:
//  * x86 wheels dispatch the AVX2/AVX-512 v_float32 kernel: float32 with FMA.
//  * Apple arm64 wheels have CV_SIMD_FP16: weights and accumulation in fp16.
// ---------------------------------------------------------------------------
namespace {

struct Taps {
  int x[4], y[4];
};
inline void TapsFor(const CubicRow& s, float xs, float ys, Taps& t, float& alpha, float& beta) {
  const int bw = std::max(s.sw, 16), bh = std::max(s.sh, 16);
  const float vx = std::min(std::max(xs, float(-bw)), float(bw * 2));
  const float vy = std::min(std::max(ys, float(-bh)), float(bh * 2));
  const int ix = FloorF(vx), iy = FloorF(vy);
  alpha = vx - float(ix);
  beta = vy - float(iy);
  for (int k = 0; k < 4; ++k) {
    t.x[k] = std::clamp(ix - 1 + k, 0, s.sw - 1) * 3;
    t.y[k] = std::clamp(iy - 1 + k, 0, s.sh - 1);
  }
}

inline void WeightsF32(float a, float w[4]) {
  constexpr float A = -0.75f;
  const float a2 = a * a, b = 1.f - a, b2 = b * b;
  w[0] = A * (a * b2);
  w[3] = A * (a2 * b);
  w[1] = std::fma(a2, std::fma(A + 2.f, a, -(A + 3.f)), 1.f);
  w[2] = ((1.f - w[0]) - w[1]) - w[3];
}

inline std::uint8_t SatU8(float v) {
  const int i = static_cast<int>(std::nearbyint(v));
  return static_cast<std::uint8_t>(std::clamp(i, 0, 255));
}

}  // namespace

#if defined(PPOCR_CVC_NEON_FP16)
void BicubicRowScalar(const CubicRow& s, const float* xs, const float* ys, int len,
                      std::uint8_t* dst) {
  const float16x8_t vA = vdupq_n_f16(-0.75f), vAp2 = vdupq_n_f16(1.25f),
                    vAp3 = vdupq_n_f16(-2.25f), v1 = vdupq_n_f16(1.f);
  for (int i = 0; i < len; i += 8) {
    const int n = std::min(8, len - i);
    float al[8] = {}, be[8] = {};
    Taps taps[8];
    for (int j = 0; j < n; ++j) TapsFor(s, xs[i + j], ys[i + j], taps[j], al[j], be[j]);
    for (int j = n; j < 8; ++j) taps[j] = taps[0];
    const float16x8_t alpha =
        vcombine_f16(vcvt_f16_f32(vld1q_f32(al)), vcvt_f16_f32(vld1q_f32(al + 4)));
    const float16x8_t beta =
        vcombine_f16(vcvt_f16_f32(vld1q_f32(be)), vcvt_f16_f32(vld1q_f32(be + 4)));
    float16x8_t wx[4], wy[4];
    auto weights = [&](float16x8_t a, float16x8_t* w) {
      const float16x8_t a2 = vmulq_f16(a, a), b = vsubq_f16(v1, a), b2 = vmulq_f16(b, b);
      w[0] = vmulq_f16(vA, vmulq_f16(a, b2));
      w[3] = vmulq_f16(vA, vmulq_f16(a2, b));
      w[1] = vfmaq_f16(v1, a2, vfmaq_f16(vAp3, vAp2, a));
      w[2] = vsubq_f16(vsubq_f16(vsubq_f16(v1, w[0]), w[1]), w[3]);
    };
    weights(alpha, wx);
    weights(beta, wy);
    std::uint8_t out[3][8];
    for (int c = 0; c < 3; ++c) {
      float16x8_t acc = vdupq_n_f16(0.f);
      for (int r = 0; r < 4; ++r) {
        std::int16_t px[4][8];
        for (int j = 0; j < 8; ++j) {
          const std::uint8_t* row = s.src + std::size_t(taps[j].y[r]) * s.sstep + c;
          for (int k = 0; k < 4; ++k) px[k][j] = row[taps[j].x[k]];
        }
        const float16x8_t p0 = vcvtq_f16_s16(vld1q_s16(px[0]));
        const float16x8_t p1 = vcvtq_f16_s16(vld1q_s16(px[1]));
        const float16x8_t p2 = vcvtq_f16_s16(vld1q_s16(px[2]));
        const float16x8_t p3 = vcvtq_f16_s16(vld1q_s16(px[3]));
        float16x8_t sum = vfmaq_f16(vmulq_f16(p0, wx[0]), p1, wx[1]);
        sum = vfmaq_f16(sum, p2, wx[2]);
        sum = vfmaq_f16(sum, p3, wx[3]);
        acc = vfmaq_f16(acc, sum, wy[r]);
      }
      vst1_u8(out[c], vqmovun_s16(vcvtnq_s16_f16(acc)));
    }
    for (int j = 0; j < n; ++j)
      for (int c = 0; c < 3; ++c) dst[(i + j) * 3 + c] = out[c][j];
  }
}
#else
void BicubicRowScalar(const CubicRow& s, const float* xs, const float* ys, int len,
                      std::uint8_t* dst) {
  for (int i = 0; i < len; ++i) {
    Taps t;
    float alpha, beta, wx[4], wy[4];
    TapsFor(s, xs[i], ys[i], t, alpha, beta);
    WeightsF32(alpha, wx);
    WeightsF32(beta, wy);
    for (int c = 0; c < 3; ++c) {
      float acc = 0.f;
      for (int r = 0; r < 4; ++r) {
        const std::uint8_t* row = s.src + std::size_t(t.y[r]) * s.sstep + c;
        float sum = float(row[t.x[0]]) * wx[0];
        sum = std::fma(float(row[t.x[1]]), wx[1], sum);
        sum = std::fma(float(row[t.x[2]]), wx[2], sum);
        sum = std::fma(float(row[t.x[3]]), wx[3], sum);
        acc = std::fma(sum, wy[r], acc);
      }
      dst[i * 3 + c] = SatU8(acc);
    }
  }
}
#endif

void RotateCropImage(const std::uint8_t* src, int sw, int sh, const float pts[8],
                     std::vector<std::uint8_t>& out, int& out_w, int& out_h) {
  // np.linalg.norm on float32 rows: sqrt of the float32 dot product.
  const auto norm = [&](int a, int b) {
    const float dx = pts[2 * a] - pts[2 * b], dy = pts[2 * a + 1] - pts[2 * b + 1];
    return std::sqrt(dx * dx + dy * dy);
  };
  int w = static_cast<int>(std::max(norm(0, 1), norm(2, 3)));
  int h = static_cast<int>(std::max(norm(0, 3), norm(1, 2)));
  const float std_pts[8] = {0.f, 0.f, float(w), 0.f, float(w), float(h), 0.f, float(h)};
  double m[9], inv[9];
  PerspectiveTransform(pts, std_pts, m);
  if (!Invert3(m, inv)) std::fill(inv, inv + 9, 0.);
  if (w <= 0 || h <= 0) w = sw, h = sh;  // cv::warpPerspective: empty dsize = src size
  float M[9];
  for (int i = 0; i < 9; ++i) M[i] = static_cast<float>(inv[i]);
  std::vector<std::uint8_t> warped(std::size_t(w) * h * 3);
  const CubicRow row{src, sw, sh, std::size_t(sw) * 3};
  auto kernel = &BicubicRowScalar;
#if defined(PPOCR_HAS_AVX2_KERNELS)
  // AVX2+FMA availability (also honours a forced scalar ISA).
  static const bool avx2 = rpocr::detail::kernels::HostHasAvxDetStem();
  if (avx2) kernel = &BicubicRowAvx2;
#endif
  constexpr int kBlock = 128;
  float xbuf[kBlock], ybuf[kBlock];
  const float M_xx = M[0], M_yx = M[3], M_zx = M[6];
  for (int y = 0; y < h; ++y) {
    const float M_x = float(y) * M[1] + M[2];
    const float M_y = float(y) * M[4] + M[5];
    const float M_z = float(y) * M[7] + M[8];
    for (int x = 0; x < w; x += kBlock) {
      const int n = std::min(kBlock, w - x);
      for (int dx = 0; dx < n; ++dx) {
        const double xf = double(x + dx);
        const double invz = 1. / (M_z + M_zx * xf);
        xbuf[dx] = float((M_x + M_xx * xf) * invz);
        ybuf[dx] = float((M_y + M_yx * xf) * invz);
      }
      kernel(row, xbuf, ybuf, n, warped.data() + (std::size_t(y) * w + x) * 3);
    }
  }
  if (h * 1.0 / w >= 1.5) {
    // np.rot90 (counter-clockwise): out[i][j] = in[j][w - 1 - i].
    out.resize(warped.size());
    for (int i = 0; i < w; ++i)
      for (int j = 0; j < h; ++j)
        std::memcpy(&out[(std::size_t(i) * h + j) * 3],
                    &warped[(std::size_t(j) * w + (w - 1 - i)) * 3], 3);
    out_w = h;
    out_h = w;
  } else {
    out.swap(warped);
    out_w = w;
    out_h = h;
  }
}

int RecResizedWidth(int w, int h, int rec_h, int batch_width) {
  const double ratio = w / static_cast<double>(h);
  const int natural = static_cast<int>(std::ceil(rec_h * ratio));
  return natural > batch_width ? batch_width : natural;
}

int RecBatchWidth(const int* widths, const int* heights, std::size_t count, int rec_h,
                  int rec_w) {
  double max_ratio = rec_w / static_cast<double>(rec_h);
  for (std::size_t i = 0; i < count; ++i)
    max_ratio = std::max(max_ratio, widths[i] * 1.0 / heights[i]);
  return static_cast<int>(rec_h * max_ratio);
}

std::array<float, 256> RecNormalizeTable(float mean, float std) {
  std::array<float, 256> t{};
  for (int v = 0; v < 256; ++v) {
    float x = static_cast<float>(v) / 255.f;
    x -= mean;
    x /= std;
    t[v] = x;
  }
  return t;
}

std::array<float, 256> DetNormalizeTable(float mean, float std) {
  std::array<float, 256> t{};
  const float scale = static_cast<float>(1 / 255.0);
  for (int v = 0; v < 256; ++v) {
    const float scaled = static_cast<float>(v) * scale;
    t[v] = static_cast<float>((static_cast<double>(scaled) - mean) / std);
  }
  return t;
}

}  // namespace rpocr::detail::cvc
