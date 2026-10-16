// Included by kern_<isa>.cpp with VW (floats per vector) and KNAME defined.
#include <cmath>
#include <cstring>
#include <algorithm>
#include "kern.hpp"
namespace ppocr::indra {
namespace {
typedef float vf __attribute__((vector_size(VW * 4)));
typedef int32_t vi __attribute__((vector_size(VW * 4)));
constexpr int NB = 2 * VW;
static inline vf ld(const float* p) { vf v; std::memcpy(&v, p, sizeof v); return v; }
static inline void st(float* p, vf v) { std::memcpy(p, &v, sizeof v); }
static inline vf bc(float s) { vf v = {}; return v + s; }
static inline vf vmax(vf a, vf b) { return a > b ? a : b; }
static inline vf vmin(vf a, vf b) { return a < b ? a : b; }
static inline vf act_v(vf x, int act) {
  if (act == 1) return x * vmax(bc(0.f), vmin(bc(1.f), x * (1.f / 6.f) + 0.5f));
  if (act == 2) return vmax(x, bc(0.f));
  return x;
}
static inline vf vexp(vf x) {  // Cephes-style expf, x <= 0 expected
  x = vmax(x, bc(-87.3f));
  vf fx = x * 1.44269504088896341f + 0.5f;
  vi ti = __builtin_convertvector(fx, vi);
  vf tf = __builtin_convertvector(ti, vf);
  tf = tf + __builtin_convertvector(tf > fx, vf);  // floor
  x = x - tf * 0.693359375f + tf * 2.12194440e-4f;
  vf y = bc(1.9875691500E-4f);
  y = y * x + 1.3981999507E-3f; y = y * x + 8.3334519073E-3f; y = y * x + 4.1665795894E-2f;
  y = y * x + 1.6666665459E-1f; y = y * x + 5.0000001201E-1f;
  y = y * x * x + x + 1.f;
  vi e = (__builtin_convertvector(tf, vi) + 127) << 23;
  vf s; std::memcpy(&s, &e, sizeof s);
  return y * s;
}

static inline vf vlog(vf x) {  // Cephes logf, x > 0
  vi bits; std::memcpy(&bits, &x, sizeof bits);
  vi ei = ((bits >> 23) & 0xff) - 126;
  vi mb = (bits & 0x7fffff) | 0x3f000000;
  vf m; std::memcpy(&m, &mb, sizeof m);
  vf e = __builtin_convertvector(ei, vf);
  auto small = m < 0.707106781186547524f;
  e = small ? e - 1.f : e;
  m = small ? m + m - 1.f : m - 1.f;
  vf z = m * m;
  vf y = bc(7.0376836292E-2f);
  y = y * m - 1.1514610310E-1f; y = y * m + 1.1676998740E-1f; y = y * m - 1.2420140846E-1f;
  y = y * m + 1.4249322787E-1f; y = y * m - 1.6668057665E-1f; y = y * m + 2.0000714765E-1f;
  y = y * m - 2.4999993993E-1f; y = y * m + 3.3333331174E-1f;
  y = y * m * z;
  y += e * -2.12194440e-4f;
  y += z * -0.5f;
  return m + y + e * 0.693359375f;
}

// x[i] = softplus(x[i]) * mul, softplus(z) = max(z, 0) + log1p(exp(-|z|))
void softplus_mul(float* x, size_t n, float mul) {
  size_t i = 0;
  for (; i + VW <= n; i += VW) {
    vf z = ld(x + i);
    vf az = vmax(z, -z);
    vf u = vexp(-az);
    vf w = u + 1.f;
    vf l1 = vlog(w) - ((w - 1.f) - u) / w;  // log1p(u) with the rounding of 1+u corrected
    st(x + i, (vmax(z, bc(0.f)) + l1) * mul);
  }
  for (; i < n; ++i) { float z = x[i]; x[i] = (z > 20.f ? z : std::log1p(std::exp(z))) * mul; }
}

void pack(const float* W, int N, int K, float* out) {
  int nbk = (N + NB - 1) / NB;
  for (int b = 0; b < nbk; ++b)
    for (int k = 0; k < K; ++k)
      for (int j = 0; j < NB; ++j) {
        int n = b * NB + j;
        out[(size_t(b) * K + k) * NB + j] = n < N ? W[size_t(n) * K + k] : 0.f;
      }
}

template <int R, bool ACC = false>
static inline void tile(const float* A, int lda, int K, const float* wp, const float* bias, float* C,
                        int ldc, int act) {
  vf acc0[R], acc1[R];
  if (ACC) {
    for (int r = 0; r < R; ++r) { acc0[r] = ld(C + size_t(r) * ldc); acc1[r] = ld(C + size_t(r) * ldc + VW); }
  } else {
    vf b0 = ld(bias), b1 = ld(bias + VW);
    for (int r = 0; r < R; ++r) { acc0[r] = b0; acc1[r] = b1; }
  }
  for (int k = 0; k < K; ++k) {
    vf w0 = ld(wp + size_t(k) * NB), w1 = ld(wp + size_t(k) * NB + VW);
    for (int r = 0; r < R; ++r) {
      vf a = bc(A[size_t(r) * lda + k]);
      acc0[r] += a * w0;
      acc1[r] += a * w1;
    }
  }
  for (int r = 0; r < R; ++r) {
    st(C + size_t(r) * ldc, act_v(acc0[r], act));
    st(C + size_t(r) * ldc + VW, act_v(acc1[r], act));
  }
}

void gemm(const float* A, int L, int K, int lda, const float* Wp, const float* bias, int Npad, float* C,
          int ldc, int act) {
#ifdef KRT
  constexpr int RT = KRT;
#else
  constexpr int RT = (VW >= 16) ? 8 : (VW >= 8 ? 6 : 4);
#endif
  // block rows so the A panel stays cache-resident across all output-channel blocks
  constexpr int LB = RT * 16;
  for (int l0 = 0; l0 < L; l0 += LB) {
    int l1 = l0 + LB < L ? l0 + LB : L;
    for (int nb = 0; nb < Npad / NB; ++nb) {
      const float* wp = Wp + size_t(nb) * K * NB;
      int l = l0;
      for (; l + RT <= l1; l += RT)
        tile<RT>(A + size_t(l) * lda, lda, K, wp, bias + nb * NB, C + size_t(l) * ldc + nb * NB, ldc, act);
      if (RT > 4 && l + 4 <= l1) { tile<4>(A + size_t(l) * lda, lda, K, wp, bias + nb * NB, C + size_t(l) * ldc + nb * NB, ldc, act); l += 4; }
      if (l + 2 <= l1) { tile<2>(A + size_t(l) * lda, lda, K, wp, bias + nb * NB, C + size_t(l) * ldc + nb * NB, ldc, act); l += 2; }
      for (; l < l1; ++l)
        tile<1>(A + size_t(l) * lda, lda, K, wp, bias + nb * NB, C + size_t(l) * ldc + nb * NB, ldc, act);
    }
  }
}

void gemm_acc(const float* A, int L, int K, int lda, const float* Wp, int Npad, float* C, int ldc, int act) {
#ifdef KRT
  constexpr int RT = KRT;
#else
  constexpr int RT = (VW >= 16) ? 8 : (VW >= 8 ? 6 : 4);
#endif
  constexpr int LB = RT * 16;
  for (int l0 = 0; l0 < L; l0 += LB) {
    int l1 = l0 + LB < L ? l0 + LB : L;
    for (int nb = 0; nb < Npad / NB; ++nb) {
      const float* wp = Wp + size_t(nb) * K * NB;
      int l = l0;
      for (; l + RT <= l1; l += RT)
        tile<RT, true>(A + size_t(l) * lda, lda, K, wp, nullptr, C + size_t(l) * ldc + nb * NB, ldc, act);
      for (; l < l1; ++l)
        tile<1, true>(A + size_t(l) * lda, lda, K, wp, nullptr, C + size_t(l) * ldc + nb * NB, ldc, act);
    }
  }
}

void dw9(const float* in, int L, int C, const float* wT, const float* b, float* out) {
  for (int l = 0; l < L; ++l) {
    int c = 0;
    for (; c + VW <= C; c += VW) {
      vf acc = ld(b + c);
      for (int j = 0; j < 9; ++j) {
        int t = l + j - 4;
        if (t < 0 || t >= L) continue;
        acc += ld(wT + size_t(j) * C + c) * ld(in + size_t(t) * C + c);
      }
      st(out + size_t(l) * C + c, acc);
    }
    for (; c < C; ++c) {
      float acc = b[c];
      for (int j = 0; j < 9; ++j) {
        int t = l + j - 4;
        if (t >= 0 && t < L) acc += wT[size_t(j) * C + c] * in[size_t(t) * C + c];
      }
      out[size_t(l) * C + c] = acc;
    }
  }
}

void rowmax(const float* x, int n, int* arg, float* conf) {
  // n is a multiple of NB (padded columns hold -1e30)
  vf m = bc(-3.0e38f), idx = {}, cur = {};
  for (int j = 0; j < VW; ++j) cur[j] = float(j);
  for (int i = 0; i < n; i += VW) {
    vf v = ld(x + i);
    auto gt = v > m;
    m = gt ? v : m;
    idx = gt ? cur : idx;
    cur += float(VW);
  }
  float best = m[0]; float bi = idx[0];
  for (int j = 1; j < VW; ++j)
    if (m[j] > best || (m[j] == best && idx[j] < bi)) { best = m[j]; bi = idx[j]; }
  vf s = {};
  vf mb = bc(best);
  for (int i = 0; i < n; i += VW) s += vexp(ld(x + i) - mb);
  float sum = 0.f;
  for (int j = 0; j < VW; ++j) sum += s[j];
  *arg = int(bi);
  *conf = 1.f / sum;
}

template <int KS>
static void dw2d_k(const float* x, int H, int W, int Cs, const float* wT, const float* b, int s, int p, float* y,
                   int Wo, int oy0, int oy1, int act) {
  for (int oy = oy0; oy < oy1; ++oy) {
    int iy0 = oy * s - p;
    int ky0 = std::max(0, -iy0), ky1 = std::min(KS, H - iy0);
    bool yin = ky0 == 0 && ky1 == KS;
    for (int ox = 0; ox < Wo; ++ox) {
      int ix0 = ox * s - p;
      float* out = y + (size_t(oy) * Wo + ox) * Cs;
      if (yin && ix0 >= 0 && ix0 + KS <= W) {
        for (int c = 0; c < Cs; c += VW) {
          vf acc = ld(b + c);
#pragma GCC unroll 5
          for (int ky = 0; ky < KS; ++ky) {
            const float* xr = x + (size_t(iy0 + ky) * W + ix0) * Cs + c;
            const float* wr = wT + size_t(ky * KS) * Cs + c;
#pragma GCC unroll 5
            for (int kx = 0; kx < KS; ++kx) acc += ld(wr + size_t(kx) * Cs) * ld(xr + size_t(kx) * Cs);
          }
          st(out + c, act_v(acc, act));
        }
        continue;
      }
      int kx0 = std::max(0, -ix0), kx1 = std::min(KS, W - ix0);
      for (int c = 0; c < Cs; c += VW) {
        vf acc = ld(b + c);
        for (int ky = ky0; ky < ky1; ++ky) {
          const float* xr = x + (size_t(iy0 + ky) * W + ix0) * Cs + c;
          const float* wr = wT + size_t(ky * KS) * Cs + c;
          for (int kx = kx0; kx < kx1; ++kx) acc += ld(wr + size_t(kx) * Cs) * ld(xr + size_t(kx) * Cs);
        }
        st(out + c, act_v(acc, act));
      }
    }
  }
}

void dw2d(const float* x, int H, int W, int Cs, const float* wT, const float* b, int k, int s, int p, float* y,
          int Wo, int oy0, int oy1, int act) {
  if (k == 3) return dw2d_k<3>(x, H, W, Cs, wT, b, s, p, y, Wo, oy0, oy1, act);
  if (k == 5) return dw2d_k<5>(x, H, W, Cs, wT, b, s, p, y, Wo, oy0, oy1, act);
  for (int oy = oy0; oy < oy1; ++oy) {
    int iy0 = oy * s - p;
    int ky0 = std::max(0, -iy0), ky1 = std::min(k, H - iy0);
    for (int ox = 0; ox < Wo; ++ox) {
      int ix0 = ox * s - p;
      int kx0 = std::max(0, -ix0), kx1 = std::min(k, W - ix0);
      float* out = y + (size_t(oy) * Wo + ox) * Cs;
      for (int c = 0; c < Cs; c += VW) {
        vf acc = ld(b + c);
        for (int ky = ky0; ky < ky1; ++ky) {
          const float* xr = x + (size_t(iy0 + ky) * W + ix0) * Cs + c;
          const float* wr = wT + size_t(ky * k) * Cs + c;
          for (int kx = kx0; kx < kx1; ++kx) acc += ld(wr + size_t(kx) * Cs) * ld(xr + size_t(kx) * Cs);
        }
        st(out + c, act_v(acc, act));
      }
    }
  }
}

template <int NV>  // 3x3 conv on a 1-channel image, NV vectors of output channels
static void c1conv3(const float* x, int H, int W, const float* wT, const float* b, int s, int p, float* y, int Wo,
                    int oy0, int oy1, int act) {
  constexpr int Cs = NV * VW;
  vf wv[9][NV], bv[NV];
  for (int t = 0; t < 9; ++t) for (int v = 0; v < NV; ++v) wv[t][v] = ld(wT + size_t(t) * Cs + v * VW);
  for (int v = 0; v < NV; ++v) bv[v] = ld(b + v * VW);
  for (int oy = oy0; oy < oy1; ++oy) {
    int iy0 = oy * s - p;
    bool yin = iy0 >= 0 && iy0 + 3 <= H;
    int ox = 0;
    if (yin && s == 1 && p == 0) {  // interior rows of a pre-padded image: 4 pixels in flight
      constexpr int P = NV == 1 ? 8 : 4;
      for (; ox + P <= Wo; ox += P) {
        vf acc[P][NV];
        for (int q = 0; q < P; ++q) for (int v = 0; v < NV; ++v) acc[q][v] = bv[v];
        const float* r = x + size_t(iy0) * W + ox;
#pragma GCC unroll 9
        for (int t = 0; t < 9; ++t)
          for (int q = 0; q < P; ++q) {
            vf xv = bc(r[(t / 3) * W + t % 3 + q]);
            for (int v = 0; v < NV; ++v) acc[q][v] += xv * wv[t][v];
          }
        for (int q = 0; q < P; ++q)
          for (int v = 0; v < NV; ++v) st(y + (size_t(oy) * Wo + ox + q) * Cs + v * VW, act_v(acc[q][v], act));
      }
    }
    for (; ox < Wo; ++ox) {
      int ix0 = ox * s - p;
      vf acc[NV];
      for (int v = 0; v < NV; ++v) acc[v] = bv[v];
      if (yin && ix0 >= 0 && ix0 + 3 <= W) {
        const float* r = x + size_t(iy0) * W + ix0;
#pragma GCC unroll 9
        for (int t = 0; t < 9; ++t) {
          vf xv = bc(r[(t / 3) * W + t % 3]);
          for (int v = 0; v < NV; ++v) acc[v] += xv * wv[t][v];
        }
      } else {
        for (int ky = 0; ky < 3; ++ky) for (int kx = 0; kx < 3; ++kx) {
          int iy = iy0 + ky, ix = ix0 + kx;
          if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
            vf xv = bc(x[size_t(iy) * W + ix]);
            for (int v = 0; v < NV; ++v) acc[v] += xv * wv[ky * 3 + kx][v];
          }
        }
      }
      for (int v = 0; v < NV; ++v) st(y + (size_t(oy) * Wo + ox) * Cs + v * VW, act_v(acc[v], act));
    }
  }
}

// dense conv on a single-channel image (x is H x W, stride 1): same weight layout as dw2d ([k*k][Cs])
void c1conv(const float* x, int H, int W, int Cs, const float* wT, const float* b, int k, int s, int p, float* y,
            int Wo, int oy0, int oy1, int act) {
  if (k == 3 && Cs == VW) return c1conv3<1>(x, H, W, wT, b, s, p, y, Wo, oy0, oy1, act);
  if (k == 3 && Cs == 2 * VW) return c1conv3<2>(x, H, W, wT, b, s, p, y, Wo, oy0, oy1, act);
  for (int oy = oy0; oy < oy1; ++oy) {
    int iy0 = oy * s - p;
    for (int ox = 0; ox < Wo; ++ox) {
      int ix0 = ox * s - p;
      float xv[64]; int nt = 0;
      for (int ky = 0; ky < k; ++ky) for (int kx = 0; kx < k; ++kx) {
        int iy = iy0 + ky, ix = ix0 + kx;
        xv[nt++] = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? x[size_t(iy) * W + ix] : 0.f;
      }
      float* out = y + (size_t(oy) * Wo + ox) * Cs;
      for (int c = 0; c < Cs; c += VW) {
        vf acc = ld(b + c);
        for (int t = 0; t < nt; ++t) acc += bc(xv[t]) * ld(wT + size_t(t) * Cs + c);
        st(out + c, act_v(acc, act));
      }
    }
  }
}

template <int R, bool HALF = false>
static inline void tile3(const float* const* A, int lda, int K, const float* const* wp, size_t woff, const float* bias,
                         float* C, int ldc, int act) {
  vf acc0[R], acc1[R];
  vf b0 = ld(bias), b1 = HALF ? b0 : ld(bias + VW);
  for (int r = 0; r < R; ++r) { acc0[r] = b0; acc1[r] = b1; }
  for (int t = 0; t < 3; ++t) {
    const float* a_ = A[t]; const float* w_ = wp[t] + woff;
    for (int k = 0; k < K; ++k) {
      vf w0 = ld(w_ + size_t(k) * NB), w1 = HALF ? w0 : ld(w_ + size_t(k) * NB + VW);
      for (int r = 0; r < R; ++r) {
        vf a = bc(a_[size_t(r) * lda + k]);
        acc0[r] += a * w0;
        if (!HALF) acc1[r] += a * w1;
      }
    }
  }
  for (int r = 0; r < R; ++r) {
    st(C + size_t(r) * ldc, act_v(acc0[r], act));
    if (!HALF) st(C + size_t(r) * ldc + VW, act_v(acc1[r], act));
  }
}

void gemm3(const float* const* A, int L, int K, int lda, const float* const* Wp, const float* bias, int Npad,
           float* C, int ldc, int act) {
#ifdef KRT
  constexpr int RT = KRT;
#else
  constexpr int RT = (VW >= 16) ? 8 : (VW >= 8 ? 6 : 4);
#endif
  for (int nb = 0; nb < Npad / NB; ++nb) {
    size_t woff = size_t(nb) * K * NB;
    int l = 0;
    for (; l + RT <= L; l += RT) {
      const float* a[3] = {A[0] + size_t(l) * lda, A[1] + size_t(l) * lda, A[2] + size_t(l) * lda};
      tile3<RT>(a, lda, K, Wp, woff, bias + nb * NB, C + size_t(l) * ldc + nb * NB, ldc, act);
    }
    if (RT > 4 && l + 4 <= L) {
      const float* a[3] = {A[0] + size_t(l) * lda, A[1] + size_t(l) * lda, A[2] + size_t(l) * lda};
      tile3<4>(a, lda, K, Wp, woff, bias + nb * NB, C + size_t(l) * ldc + nb * NB, ldc, act);
      l += 4;
    }
    if (l + 2 <= L) {
      const float* a[3] = {A[0] + size_t(l) * lda, A[1] + size_t(l) * lda, A[2] + size_t(l) * lda};
      tile3<2>(a, lda, K, Wp, woff, bias + nb * NB, C + size_t(l) * ldc + nb * NB, ldc, act);
      l += 2;
    }
    for (; l < L; ++l) {
      const float* a[3] = {A[0] + size_t(l) * lda, A[1] + size_t(l) * lda, A[2] + size_t(l) * lda};
      tile3<1>(a, lda, K, Wp, woff, bias + nb * NB, C + size_t(l) * ldc + nb * NB, ldc, act);
    }
  }
  if (Npad % NB) {  // trailing half block (Npad % NB == VW): one vector of output channels
    int nb = Npad / NB;
    size_t woff = size_t(nb) * K * NB;
    int l = 0;
    constexpr int RH = RT * 2 > 24 ? 24 : RT * 2;
    for (; l + RH <= L; l += RH) {
      const float* a[3] = {A[0] + size_t(l) * lda, A[1] + size_t(l) * lda, A[2] + size_t(l) * lda};
      tile3<RH, true>(a, lda, K, Wp, woff, bias + nb * NB, C + size_t(l) * ldc + nb * NB, ldc, act);
    }
#define HT(n) if (RH > n && l + n <= L) { const float* a[3] = {A[0] + size_t(l) * lda, A[1] + size_t(l) * lda, A[2] + size_t(l) * lda}; \
      tile3<n, true>(a, lda, K, Wp, woff, bias + nb * NB, C + size_t(l) * ldc + nb * NB, ldc, act); l += n; }
    HT(8) HT(4) HT(2)
#undef HT
    for (; l < L; ++l) {
      const float* a[3] = {A[0] + size_t(l) * lda, A[1] + size_t(l) * lda, A[2] + size_t(l) * lda};
      tile3<1, true>(a, lda, K, Wp, woff, bias + nb * NB, C + size_t(l) * ldc + nb * NB, ldc, act);
    }
  }
}

// in-place softmax over n floats (masked entries already -inf)
void softmax_row(float* x, int n) {
  float mx = -INFINITY;
  int i = 0;
  vf m = bc(-INFINITY);
  for (; i + VW <= n; i += VW) m = vmax(m, ld(x + i));
  for (int j = 0; j < VW; ++j) mx = std::max(mx, m[j]);
  for (; i < n; ++i) mx = std::max(mx, x[i]);
  vf sv = bc(0.f); float sum = 0.f;
  for (i = 0; i + VW <= n; i += VW) { vf e = vexp(ld(x + i) - mx); st(x + i, e); sv += e; }
  for (; i < n; ++i) { x[i] = std::exp(x[i] - mx); sum += x[i]; }
  for (int j = 0; j < VW; ++j) sum += sv[j];
  float inv = 1.f / sum;
  for (i = 0; i + VW <= n; i += VW) st(x + i, ld(x + i) * inv);
  for (; i < n; ++i) x[i] *= inv;
}
// in-place exact-erf GELU: 0.5 x (1 + erf(x / sqrt2)); erfc via Numerical Recipes (rel err < 1.2e-7)
void gelu_row(float* x, int n) {
  int i = 0;
  for (; i + VW <= n; i += VW) {
    vf v = ld(x + i), z = v * 0.70710678118654752f, az = vmax(z, -z);
    vf t = 1.f / (az * 0.5f + 1.f);
    vf p = bc(0.17087277f);
    p = p * t - 0.82215223f; p = p * t + 1.48851587f; p = p * t - 1.13520398f; p = p * t + 0.27886807f;
    p = p * t - 0.18628806f; p = p * t + 0.09678418f; p = p * t + 0.37409196f; p = p * t + 1.00002368f;
    p = p * t - 1.26551223f;
    vf erfc = t * vexp(vmin(p - az * az, bc(0.f)));
    vf erf = 1.f - erfc;
    erf = z < 0.f ? -erf : erf;
    st(x + i, (erf + 1.f) * v * 0.5f);
  }
  for (; i < n; ++i) x[i] = (std::erf(x[i] * 0.70710678118654752f) + 1.f) * x[i] * 0.5f;
}
typedef float v4 __attribute__((vector_size(16)));
template <int R>
static inline void tile4(const float* A, int lda, int K, const float* W4, v4 b, float* C, int ldc, int N, int act) {
  v4 acc[R];
  for (int r = 0; r < R; ++r) acc[r] = b;
  for (int k = 0; k < K; ++k) {
    v4 w; std::memcpy(&w, W4 + size_t(k) * 4, 16);
    for (int r = 0; r < R; ++r) acc[r] += A[size_t(r) * lda + k] * w;
  }
  for (int r = 0; r < R; ++r) {
    v4 x = acc[r];
    if (act == 1) { v4 t = x * (1.f / 6.f) + 0.5f; t = t < 1.f ? t : 1.f; t = t > 0.f ? t : 0.f; x = x * t; }
    else if (act == 2) x = x > 0.f ? x : 0.f;
    if (N == 4) std::memcpy(C + size_t(r) * ldc, &x, 16);
    else for (int n = 0; n < N; ++n) C[size_t(r) * ldc + n] = x[n];
  }
}
void gemm_n4(const float* A, int L, int K, int lda, const float* W4, const float* bias4, int N, float* C,
             int ldc, int act) {
  v4 b; std::memcpy(&b, bias4, 16);
  int l = 0;
  for (; l + 8 <= L; l += 8) tile4<8>(A + size_t(l) * lda, lda, K, W4, b, C + size_t(l) * ldc, ldc, N, act);
  for (; l < L; ++l) tile4<1>(A + size_t(l) * lda, lda, K, W4, b, C + size_t(l) * ldc, ldc, N, act);
}
const Kern kKern = {KNAME, NB, pack, gemm, dw9, rowmax, dw2d, c1conv, gemm3, softplus_mul, gemm_n4, gemm_acc, softmax_row, gelu_row};
}  // namespace
}  // namespace ppocr::indra
