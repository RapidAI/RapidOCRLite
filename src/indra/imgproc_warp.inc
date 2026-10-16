// warpAffine linear kernel (included by imgproc.cpp, and by imgproc_avx2.cpp for an AVX2+FMA build).
// Requires -ffp-contract=off: all FMAs are explicit, so every build gives the same bits.
namespace IMG_NS {
inline int rne_w(float v) { return (int)std::nearbyint(v); }
inline uint8_t sat_u8f_w(float v) { int i = rne_w(v); return (uint8_t)(i < 0 ? 0 : i > 255 ? 255 : i); }
// M: inverse map (dst -> src) as OpenCV passes to the kernel; uf = SIMD block (2 x f32 lanes).
template <typename T>
void warp_linear(const T* src, int sw, int sh, T* dst, int dw, int dh, int y0, int y1, const double* dM, bool replicate,
                 float bval, int uf, int fl) {
  (void)dh;
  float M[6];
  for (int i = 0; i < 6; i++) M[i] = (float)dM[i];
  const bool mxfma = !(fl & 4);       // M_x = float(y*M1 + M2): contracted in the dispatched TU
  const int tail = (fl >> 3) & 3;     // scalar tail coords: 0 plain, 1 fma(x,M0,y*M1)+M2, 2 fma(y,M1,x*M0)+M2
  const bool tail_fused = !(fl & 32); // scalar lerp contracted
  auto px = [&](int ix, int iy) -> float {
    if (replicate) { ix = std::min(std::max(ix, 0), sw - 1); iy = std::min(std::max(iy, 0), sh - 1); }
    else if ((unsigned)ix >= (unsigned)sw || (unsigned)iy >= (unsigned)sh) return bval;
    return (float)src[(size_t)iy * sw + ix];
  };
  auto put = [&](T* d, float v) {
    if constexpr (sizeof(T) == 1) *d = sat_u8f_w(v); else *d = v;
  };
  for (int y = y0; y < y1; y++) {
    T* D = dst + (size_t)y * dw;
    float yf = (float)y;
    float Mx = mxfma ? std::fma(yf, M[1], M[2]) : yf * M[1] + M[2];
    float My = mxfma ? std::fma(yf, M[4], M[5]) : yf * M[4] + M[5];
    int x = 0, vend = dw / uf * uf;
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (sizeof(T) == 4) {
      const __m256 vM0 = _mm256_set1_ps(M[0]), vM3 = _mm256_set1_ps(M[3]), vMx = _mm256_set1_ps(Mx), vMy = _mm256_set1_ps(My);
      const __m256i lim_x = _mm256_set1_epi32(sw - 1), lim_y = _mm256_set1_epi32(sh - 1), one = _mm256_set1_epi32(1);
      const __m256i vsw = _mm256_set1_epi32(sw);
      for (; x + 8 <= vend; x += 8) {
        __m256 xf = _mm256_add_ps(_mm256_set1_ps((float)x), _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7));
        __m256 sx = _mm256_fmadd_ps(vM0, xf, vMx), sy = _mm256_fmadd_ps(vM3, xf, vMy);
        __m256 fx = _mm256_floor_ps(sx), fy = _mm256_floor_ps(sy);
        __m256i ix = _mm256_cvttps_epi32(fx), iy = _mm256_cvttps_epi32(fy);
        // all 4 taps inside: 0 <= ix < sw-1 and 0 <= iy < sh-1 (unsigned compare via max trick)
        __m256i okx = _mm256_and_si256(_mm256_cmpgt_epi32(lim_x, ix), _mm256_cmpgt_epi32(ix, _mm256_set1_epi32(-1)));
        __m256i oky = _mm256_and_si256(_mm256_cmpgt_epi32(lim_y, iy), _mm256_cmpgt_epi32(iy, _mm256_set1_epi32(-1)));
        if (_mm256_movemask_epi8(_mm256_and_si256(okx, oky)) != -1) {
          // mixed block: per pixel (same arithmetic)
          for (int k = 0; k < 8; k++) {
            int xx = x + k; float xf1 = (float)xx;
            float sx1 = std::fma(M[0], xf1, Mx), sy1 = std::fma(M[3], xf1, My);
            int jx = (int)std::floor(sx1), jy = (int)std::floor(sy1);
            float a = sx1 - (float)jx, b = sy1 - (float)jy;
            float p00 = px(jx, jy), p01 = px(jx + 1, jy), p10 = px(jx, jy + 1), p11 = px(jx + 1, jy + 1);
            float v0 = std::fma(a, p01 - p00, p00), v1 = std::fma(a, p11 - p10, p10);
            put(D + xx, std::fma(b, v1 - v0, v0));
          }
          continue;
        }
        __m256 a = _mm256_sub_ps(sx, fx), b = _mm256_sub_ps(sy, fy);
        __m256i idx = _mm256_add_epi32(_mm256_mullo_epi32(iy, vsw), ix);
        const float* sp = (const float*)src;
        __m256 p00 = _mm256_i32gather_ps(sp, idx, 4), p01 = _mm256_i32gather_ps(sp, _mm256_add_epi32(idx, one), 4);
        __m256i idx2 = _mm256_add_epi32(idx, vsw);
        __m256 p10 = _mm256_i32gather_ps(sp, idx2, 4), p11 = _mm256_i32gather_ps(sp, _mm256_add_epi32(idx2, one), 4);
        __m256 v0 = _mm256_fmadd_ps(a, _mm256_sub_ps(p01, p00), p00), v1 = _mm256_fmadd_ps(a, _mm256_sub_ps(p11, p10), p10);
        _mm256_storeu_ps((float*)(D + x), _mm256_fmadd_ps(b, _mm256_sub_ps(v1, v0), v0));
      }
    }
#endif
    for (; x < vend; x++) {
      float xf = (float)x;
      float sx = std::fma(M[0], xf, Mx), sy = std::fma(M[3], xf, My);
      int ix = (int)std::floor(sx), iy = (int)std::floor(sy);
      float a = sx - (float)ix, b = sy - (float)iy;
      float p00, p01, p10, p11;
      if ((unsigned)ix < (unsigned)(sw - 1) && (unsigned)iy < (unsigned)(sh - 1)) {
        const T* q = src + (size_t)iy * sw + ix;
        p00 = (float)q[0]; p01 = (float)q[1]; p10 = (float)q[sw]; p11 = (float)q[sw + 1];
      } else {
        p00 = px(ix, iy); p01 = px(ix + 1, iy); p10 = px(ix, iy + 1); p11 = px(ix + 1, iy + 1);
      }
      float v0 = std::fma(a, p01 - p00, p00), v1 = std::fma(a, p11 - p10, p10);
      put(D + x, std::fma(b, v1 - v0, v0));
    }
    for (; x < dw; x++) {
      float xf = (float)x, sx, sy;
      if (tail == 1) { sx = std::fma(xf, M[0], yf * M[1]) + M[2]; sy = std::fma(xf, M[3], yf * M[4]) + M[5]; }
      else if (tail == 2) { sx = std::fma(yf, M[1], xf * M[0]) + M[2]; sy = std::fma(yf, M[4], xf * M[3]) + M[5]; }
      else { sx = xf * M[0] + yf * M[1] + M[2]; sy = xf * M[3] + yf * M[4] + M[5]; }
      int ix = (int)std::floor(sx), iy = (int)std::floor(sy);
      float a = sx - (float)ix, b = sy - (float)iy;
      if (!replicate && ((unsigned)(ix + 1) >= (unsigned)(sw + 1) || (unsigned)(iy + 1) >= (unsigned)(sh + 1))) {
        put(D + x, bval); continue;
      }
      float p00 = px(ix, iy), p01 = px(ix + 1, iy), p10 = px(ix, iy + 1), p11 = px(ix + 1, iy + 1);
      float v0, v1;
      if (tail_fused) { v0 = std::fma(a, p01 - p00, p00); v1 = std::fma(a, p11 - p10, p10); v0 = std::fma(b, v1 - v0, v0); }
      else { v0 = p00 + a * (p01 - p00); v1 = p10 + a * (p11 - p10); v0 += b * (v1 - v0); }
      put(D + x, v0);
    }
  }
}

void warp_f32(const float* s, int sw, int sh, float* d, int dw, int dh, int y0, int y1, const double* M, bool rep, float bv, int uf, int fl) {
  warp_linear(s, sw, sh, d, dw, dh, y0, y1, M, rep, bv, uf, fl);
}
void warp_u8(const uint8_t* s, int sw, int sh, uint8_t* d, int dw, int dh, int y0, int y1, const double* M, bool rep, float bv, int uf, int fl) {
  warp_linear(s, sw, sh, d, dw, dh, y0, y1, M, rep, bv, uf, fl);
}
}  // namespace IMG_NS
