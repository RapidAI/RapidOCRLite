// AVX2/FMA build of the OpenCV bicubic8uC3 kernel (v_float32, 8 lanes). See
// cv_compat.cpp. Compiled without fast-math or contraction: every rounding
// step is an explicit intrinsic, matching warp_kernels.simd.hpp.
#include "cv_compat.hpp"

#include <immintrin.h>

#include <algorithm>

namespace rpocr::detail::cvc {

void BicubicRowAvx2(const CubicRow& s, const float* xs, const float* ys, int len,
                    std::uint8_t* dst) {
  const int bw = std::max(s.sw, 16), bh = std::max(s.sh, 16);
  const __m256 minx = _mm256_set1_ps(float(-bw)), maxx = _mm256_set1_ps(float(bw * 2));
  const __m256 miny = _mm256_set1_ps(float(-bh)), maxy = _mm256_set1_ps(float(bh * 2));
  const __m256 vA = _mm256_set1_ps(-0.75f), vAp2 = _mm256_set1_ps(1.25f),
               vAp3 = _mm256_set1_ps(-2.25f), v1 = _mm256_set1_ps(1.f);
  auto weights = [&](__m256 a, __m256* w) {
    const __m256 a2 = _mm256_mul_ps(a, a), b = _mm256_sub_ps(v1, a), b2 = _mm256_mul_ps(b, b);
    w[0] = _mm256_mul_ps(vA, _mm256_mul_ps(a, b2));
    w[3] = _mm256_mul_ps(vA, _mm256_mul_ps(a2, b));
    w[1] = _mm256_fmadd_ps(a2, _mm256_fmadd_ps(vAp2, a, vAp3), v1);
    w[2] = _mm256_sub_ps(_mm256_sub_ps(_mm256_sub_ps(v1, w[0]), w[1]), w[3]);
  };
  for (int i = 0; i < len; i += 8) {
    const int n = std::min(8, len - i);
    alignas(32) float xb[8] = {}, yb[8] = {};
    for (int j = 0; j < n; ++j) xb[j] = xs[i + j], yb[j] = ys[i + j];
    const __m256 vx = _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(xb), minx), maxx);
    const __m256 vy = _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(yb), miny), maxy);
    const __m256 fxf = _mm256_floor_ps(vx), fyf = _mm256_floor_ps(vy);
    const __m256i ixv = _mm256_cvttps_epi32(fxf), iyv = _mm256_cvttps_epi32(fyf);
    __m256 wx[4], wy[4];
    weights(_mm256_sub_ps(vx, _mm256_cvtepi32_ps(ixv)), wx);
    weights(_mm256_sub_ps(vy, _mm256_cvtepi32_ps(iyv)), wy);
    alignas(32) int ix[8], iy[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(ix), ixv);
    _mm256_store_si256(reinterpret_cast<__m256i*>(iy), iyv);
    int tx[8][4], ty[8][4];
    for (int j = 0; j < 8; ++j)
      for (int k = 0; k < 4; ++k) {
        tx[j][k] = std::clamp(ix[j] - 1 + k, 0, s.sw - 1) * 3;
        ty[j][k] = std::clamp(iy[j] - 1 + k, 0, s.sh - 1);
      }
    alignas(32) int res[3][8];
    for (int c = 0; c < 3; ++c) {
      __m256 acc = _mm256_setzero_ps();
      for (int r = 0; r < 4; ++r) {
        alignas(32) float px[4][8];
        for (int j = 0; j < 8; ++j) {
          const std::uint8_t* row = s.src + std::size_t(ty[j][r]) * s.sstep + c;
          for (int k = 0; k < 4; ++k) px[k][j] = float(row[tx[j][k]]);
        }
        __m256 sum = _mm256_fmadd_ps(_mm256_load_ps(px[1]), wx[1],
                                     _mm256_mul_ps(_mm256_load_ps(px[0]), wx[0]));
        sum = _mm256_fmadd_ps(_mm256_load_ps(px[2]), wx[2], sum);
        sum = _mm256_fmadd_ps(_mm256_load_ps(px[3]), wx[3], sum);
        acc = _mm256_fmadd_ps(sum, wy[r], acc);
      }
      _mm256_store_si256(reinterpret_cast<__m256i*>(res[c]), _mm256_cvtps_epi32(acc));
    }
    for (int j = 0; j < n; ++j)
      for (int c = 0; c < 3; ++c)
        dst[(i + j) * 3 + c] = static_cast<std::uint8_t>(std::clamp(res[c][j], 0, 255));
  }
}

}  // namespace rpocr::detail::cvc
