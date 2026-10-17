// INT8 GEMM kernels for the lite recognizer (see int8_gemm.hpp). x86: AVX-512 VNNI (vpdpbusd, u8 x s8 with
// a +128 activation offset corrected by the weight row sums) or AVX2 (sign-extend to s16, vpmaddwd);
// aarch64: SDOT when the compiler targets dotprod, else SMULL/SADALP; scalar otherwise.
#include "int8_gemm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#include "cpu_x86.hpp"
#define I8_X86 1
#if defined(_MSC_VER) && !defined(__clang__)
#define I8_TARGET(x)  // MSVC accepts any ISA's intrinsics; callers check the CPU first
#else
#define I8_TARGET(x) __attribute__((target(x)))
#endif
#if defined(__x86_64__) || defined(_M_X64)
#define I8_VNNI 1  // 32-bit builds keep AVX2 + scalar (no AVX-512)
#endif
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define I8_NEON 1
#endif

namespace rpocr::indra {
namespace {
constexpr int kAlign = 64;

inline float Act(float x, int act) {
  if (act == 1) return x * std::max(0.f, std::min(1.f, x * (1.f / 6.f) + 0.5f));
  if (act == 2) return std::max(x, 0.f);
  return x;
}

// Block kernel: rows r < R (R <= 4) of quantized activations (a: s8, au: a + 128 as u8; stride Kp) times one
// packed block of 16 weight rows -> out[r][16] int32 dot products (raw s8 x s8 sums, offset already removed).
using Block16 = void (*)(const int8_t* a, const uint8_t* au, int R, const int8_t* wp, int Kp, const int32_t* rowsum, int32_t* out);

void BlockScalar(const int8_t* a, const uint8_t*, int R, const int8_t* wp, int Kp, const int32_t*, int32_t* out) {
  for (int r = 0; r < R; r++) {
    int32_t acc[16] = {};
    const int8_t* ar = a + size_t(r) * Kp;
    for (int kg = 0; kg < Kp / 4; kg++) {
      const int8_t* w = wp + size_t(kg) * 64;
      for (int j = 0; j < 16; j++)
        for (int g = 0; g < 4; g++) acc[j] += int32_t(ar[kg * 4 + g]) * w[j * 4 + g];
    }
    std::memcpy(out + r * 16, acc, sizeof acc);
  }
}

#ifdef I8_VNNI
I8_TARGET("avx512f,avx512bw,avx512vnni")
void BlockVnni(const int8_t*, const uint8_t* au, int R, const int8_t* wp, int Kp, const int32_t* rowsum, int32_t* out) {
  __m512i acc[4] = {_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};
  for (int kg = 0; kg < Kp / 4; kg++) {
    __m512i w = _mm512_loadu_si512(wp + size_t(kg) * 64);
    for (int r = 0; r < R; r++) {
      int32_t x; std::memcpy(&x, au + size_t(r) * Kp + kg * 4, 4);
      acc[r] = _mm512_dpbusd_epi32(acc[r], _mm512_set1_epi32(x), w);
    }
  }
  __m512i corr = _mm512_slli_epi32(_mm512_loadu_si512(rowsum), 7);  // 128 * rowsum
  for (int r = 0; r < R; r++) _mm512_storeu_si512(out + r * 16, _mm512_sub_epi32(acc[r], corr));
}
#endif
#ifdef I8_X86
I8_TARGET("avx2,fma")
void BlockAvx2(const int8_t* a, const uint8_t*, int R, const int8_t* wp, int Kp, const int32_t*, int32_t* out) {
  // 16 outputs as two 8-lane s32 halves; each k pair (2 activations as s16) x weights widened to s16 via vpmaddwd
  for (int r = 0; r < R; r++) {
    __m256i lo = _mm256_setzero_si256(), hi = lo;
    const int8_t* ar = a + size_t(r) * Kp;
    for (int kg = 0; kg < Kp / 4; kg++) {
      const int8_t* w = wp + size_t(kg) * 64;
      // w bytes: [j][g]; widen 16 bytes = 4 outputs x 4 k at a time
      __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w + 0)));
      __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w + 16)));
      __m256i w2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w + 32)));
      __m256i w3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w + 48)));
      // activation pattern per output: (a0,a1,a2,a3) repeated -> madd gives (a0w0+a1w1, a2w2+a3w3) per output
      __m256i x = _mm256_set1_epi64x((long long)(uint16_t)ar[kg * 4] | ((long long)(uint16_t)ar[kg * 4 + 1] << 16) |
                                     ((long long)(uint16_t)ar[kg * 4 + 2] << 32) | ((long long)(uint16_t)ar[kg * 4 + 3] << 48));
      __m256i p0 = _mm256_madd_epi16(w0, x), p1 = _mm256_madd_epi16(w1, x);  // outputs 0-3 / 4-7, 2 partials each
      __m256i p2 = _mm256_madd_epi16(w2, x), p3 = _mm256_madd_epi16(w3, x);
      lo = _mm256_add_epi32(lo, _mm256_hadd_epi32(p0, p1));  // lanes: o0 o1 o4 o5 | o2 o3 o6 o7
      hi = _mm256_add_epi32(hi, _mm256_hadd_epi32(p2, p3));
    }
    alignas(32) int32_t t[16];
    _mm256_store_si256((__m256i*)t, lo); _mm256_store_si256((__m256i*)(t + 8), hi);
    static const int perm[8] = {0, 1, 4, 5, 2, 3, 6, 7};  // hadd lane order -> output index
    for (int i = 0; i < 8; i++) { out[r * 16 + perm[i]] = t[i]; out[r * 16 + 8 + perm[i]] = t[8 + i]; }
  }
}
#endif

#ifdef I8_NEON
void BlockNeon(const int8_t* a, const uint8_t*, int R, const int8_t* wp, int Kp, const int32_t*, int32_t* out) {
  int32x4_t acc[4][4];
  for (int r = 0; r < 4; r++) for (int j = 0; j < 4; j++) acc[r][j] = vdupq_n_s32(0);
  for (int k = 0; k < Kp; k += 16) {
    int8x16_t x[4];
    for (int r = 0; r < R; r++) x[r] = vld1q_s8(a + size_t(r) * Kp + k);
    for (int g = 0; g < 4; g++) {
      const int8_t* w = wp + size_t(k / 4 + g) * 64;
      int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16), w2 = vld1q_s8(w + 32), w3 = vld1q_s8(w + 48);
      for (int r = 0; r < R; r++) {
#if defined(__ARM_FEATURE_DOTPROD)
#define I8_DOT(acc_, w_) acc_ = (g == 0 ? vdotq_laneq_s32(acc_, w_, x[r], 0) : g == 1 ? vdotq_laneq_s32(acc_, w_, x[r], 1) \
                                : g == 2 ? vdotq_laneq_s32(acc_, w_, x[r], 2) : vdotq_laneq_s32(acc_, w_, x[r], 3))
#else
#define I8_DOT(acc_, w_) do { int8x16_t xb = vreinterpretq_s8_s32(vdupq_n_s32(vgetq_lane_s32(vreinterpretq_s32_s8(x[r]), 0))); \
        switch (g) { case 1: xb = vreinterpretq_s8_s32(vdupq_laneq_s32(vreinterpretq_s32_s8(x[r]), 1)); break; \
                     case 2: xb = vreinterpretq_s8_s32(vdupq_laneq_s32(vreinterpretq_s32_s8(x[r]), 2)); break; \
                     case 3: xb = vreinterpretq_s8_s32(vdupq_laneq_s32(vreinterpretq_s32_s8(x[r]), 3)); break; default: break; } \
        int16x8_t pl = vmull_s8(vget_low_s8(w_), vget_low_s8(xb)), ph = vmull_s8(vget_high_s8(w_), vget_high_s8(xb)); \
        acc_ = vaddq_s32(acc_, vcombine_s32(vmovn_s64(vpaddlq_s32(vpaddlq_s16(pl))), vmovn_s64(vpaddlq_s32(vpaddlq_s16(ph))))); } while (0)
#endif
        I8_DOT(acc[r][0], w0); I8_DOT(acc[r][1], w1); I8_DOT(acc[r][2], w2); I8_DOT(acc[r][3], w3);
#undef I8_DOT
      }
    }
  }
  for (int r = 0; r < R; r++) for (int j = 0; j < 4; j++) vst1q_s32(out + r * 16 + j * 4, acc[r][j]);
}
#endif

struct Impl { Block16 blk; bool offset; const char* name; };
Impl Select() {
  if (const char* e = std::getenv("PPOCR_INDRA_INT8_ISA"); e && std::strcmp(e, "scalar") == 0) return {BlockScalar, false, "int8-scalar"};
#ifdef I8_X86
  const char* e = std::getenv("PPOCR_INDRA_INT8_ISA");
  bool no512 = e && std::strcmp(e, "avx2") == 0;
#ifdef I8_VNNI
  if (!no512 && cpu::Avx512Vnni()) return {BlockVnni, true, "int8-avx512vnni"};
#else
  (void)no512;
#endif
  if (cpu::Avx2Fma()) return {BlockAvx2, false, "int8-avx2"};
#endif
#ifdef I8_NEON
#if defined(__ARM_FEATURE_DOTPROD)
  return {BlockNeon, false, "int8-neon-sdot"};
#else
  return {BlockNeon, false, "int8-neon"};
#endif
#endif
  return {BlockScalar, false, "int8-scalar"};
}
const Impl& Get() { static const Impl i = Select(); return i; }
}  // namespace

const char* Int8Isa() { return Get().name; }

QMat QuantizeRows(const float* W, int N, int K) {
  QMat m; m.N = N; m.K = K; m.Kp = (K + kAlign - 1) / kAlign * kAlign;
  int Np = (N + 15) / 16 * 16;
  m.q.assign(size_t(Np) * m.Kp, 0); m.scale.assign(Np, 1.f); m.rowsum.assign(Np, 0);
  for (int n = 0; n < N; n++) {
    float mx = 0.f;
    for (int k = 0; k < K; k++) mx = std::max(mx, std::fabs(W[size_t(n) * K + k]));
    float s = mx > 0 ? mx / 127.f : 1.f;
    m.scale[n] = s;
    int32_t rs = 0;
    for (int k = 0; k < K; k++) {
      int v = std::max(-127, std::min(127, (int)std::lrint(W[size_t(n) * K + k] / s)));
      size_t nb = n / 16, j = n % 16, kg = k / 4, g = k % 4;
      m.q[(nb * (m.Kp / 4) + kg) * 64 + j * 4 + g] = (int8_t)v; rs += v;
    }
    m.rowsum[n] = rs;
  }
  return m;
}

void GemmQ(const float* A, int L, int K, int lda, const QMat& W, const float* bias, float* C, int ldc, int act) {
  const Impl& im = Get();
  const int Kp = W.Kp, NB = (W.N + 15) / 16;
  std::vector<int8_t> aq(size_t(L) * Kp, 0); std::vector<uint8_t> au(im.offset ? size_t(L) * Kp : 0);
  std::vector<float> sa(L);
  for (int l = 0; l < L; l++) {
    const float* a = A + size_t(l) * lda;
    float mx = 0.f;
    for (int k = 0; k < K; k++) mx = std::max(mx, std::fabs(a[k]));
    float s = mx > 0 ? mx / 127.f : 1.f, inv = 1.f / s;
    sa[l] = s;
    int8_t* q = &aq[size_t(l) * Kp];
    for (int k = 0; k < K; k++) q[k] = (int8_t)std::max(-127L, std::min(127L, std::lrint(a[k] * inv)));
    if (im.offset) for (int k = 0; k < Kp; k++) au[size_t(l) * Kp + k] = (uint8_t)(q[k] + 128);
  }
  int32_t acc[4 * 16];
  for (int nb = 0; nb < NB; nb++) {
    const int8_t* wp = &W.q[size_t(nb) * (Kp / 4) * 64];
    const int n0 = nb * 16, nn = std::min(16, W.N - n0);
    for (int l = 0; l < L; l += 4) {
      int R = std::min(4, L - l);
      im.blk(&aq[size_t(l) * Kp], im.offset ? &au[size_t(l) * Kp] : nullptr, R, wp, Kp, &W.rowsum[n0], acc);
      for (int r = 0; r < R; r++) {
        float* c = C + size_t(l + r) * ldc + n0;
        const float s = sa[l + r];
        for (int j = 0; j < nn; j++) c[j] = Act(float(acc[r * 16 + j]) * (s * W.scale[n0 + j]) + bias[n0 + j], act);
      }
    }
  }
}

}  // namespace rpocr::indra
