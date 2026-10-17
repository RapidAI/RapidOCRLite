#pragma once
// Portable float/int32 vector types for kern_impl.inc.
//
// Non-x86 GCC/Clang builds (NEON, LoongArch, ...) use the native vector extensions (vector_size);
// x86 builds (and MSVC, which has no vector extensions) get small wrapper structs over SSE2 / AVX2 / AVX-512F intrinsics with the same
// lane-wise IEEE semantics (VW = 1 falls back to a plain scalar lane):
//   +, -, *, / (vector or broadcast scalar), unary -, ordered compares -> mask,
//   select, float<->int32 truncating conversion, int32 shifts / and / or.
// Multiply-adds are written explicitly as KV_FMA(a, b, c): the intrinsic path issues the fused
// instruction when the TU is built for FMA (KV_HAS_FMA) and mul + add otherwise; the native path
// (non-x86) keeps the expression a * b + c for the compiler to contract.
#include <cstdint>
#include <cstring>

// x86 (all compilers) uses the intrinsic wrappers: every multiply-add is then fused or not by
// construction rather than by the compiler's contraction heuristics (GCC's avoid-fma-max-bits
// tuning, for one, leaves some 128/256-bit FMA chains unfused), so GCC, Clang and MSVC builds
// produce the same bits. The TUs are compiled with -ffp-contract=off (MSVC: /fp:precise).
#if !defined(KV_INTRIN) && !defined(KV_NATIVE) && \
    ((defined(_MSC_VER) && !defined(__clang__)) || defined(__x86_64__) || defined(_M_X64) || defined(_M_IX86))
#define KV_INTRIN 1
#endif

#ifndef KV_INTRIN
namespace kv {
template <int N> struct T {
  typedef float vf __attribute__((vector_size(N * 4)));
  typedef int32_t vi __attribute__((vector_size(N * 4)));
};
#define KV_SEL(m, a, b) ((m) ? (a) : (b))
#define KV_CVTI(x, VI) __builtin_convertvector(x, VI)
#define KV_CVTF(x, VF) __builtin_convertvector(x, VF)
#define KV_MASKF(m, VF) __builtin_convertvector(m, VF)
#define KV_FMA(a, b, c) ((a) * (b) + (c))
#define KV_SFMA(a, b, c) ((a) * (b) + (c))
}  // namespace kv
#else
#include <immintrin.h>
#include <cmath>
namespace kv {
template <int N> struct F;
template <int N> struct I;
template <int N> struct M;

// ---------------------------------------------------------------- traits per width
template <int N> struct Tr;
template <> struct Tr<4> {
  typedef __m128 f; typedef __m128i i;
  static f zero() { return _mm_setzero_ps(); }
  static f set1(float s) { return _mm_set1_ps(s); }
  static f add(f a, f b) { return _mm_add_ps(a, b); }
  static f sub(f a, f b) { return _mm_sub_ps(a, b); }
  static f mul(f a, f b) { return _mm_mul_ps(a, b); }
  static f div(f a, f b) { return _mm_div_ps(a, b); }
#ifdef KV_HAS_FMA
  static f fma(f a, f b, f c) { return _mm_fmadd_ps(a, b, c); }
#else
  static f fma(f a, f b, f c) { return _mm_add_ps(_mm_mul_ps(a, b), c); }
#endif
  static f gt(f a, f b) { return _mm_cmpgt_ps(a, b); }
  static f lt(f a, f b) { return _mm_cmplt_ps(a, b); }
  static f sel(f m, f a, f b) { return _mm_or_ps(_mm_and_ps(m, a), _mm_andnot_ps(m, b)); }
  static f band(f a, f b) { return _mm_and_ps(a, b); }
  static f neg(f a) { return _mm_xor_ps(a, _mm_set1_ps(-0.f)); }
  static i cvti(f a) { return _mm_cvttps_epi32(a); }
  static f cvtf(i a) { return _mm_cvtepi32_ps(a); }
  static i iset1(int s) { return _mm_set1_epi32(s); }
  static i iadd(i a, i b) { return _mm_add_epi32(a, b); }
  static i isub(i a, i b) { return _mm_sub_epi32(a, b); }
  static i iand(i a, i b) { return _mm_and_si128(a, b); }
  static i ior(i a, i b) { return _mm_or_si128(a, b); }
  static i ishl(i a, int n) { return _mm_slli_epi32(a, n); }
  static i ishr(i a, int n) { return _mm_srai_epi32(a, n); }
};
#if defined(KV_AVX2) || defined(KV_AVX512)
template <> struct Tr<8> {
  typedef __m256 f; typedef __m256i i;
  static f zero() { return _mm256_setzero_ps(); }
  static f set1(float s) { return _mm256_set1_ps(s); }
  static f add(f a, f b) { return _mm256_add_ps(a, b); }
  static f sub(f a, f b) { return _mm256_sub_ps(a, b); }
  static f mul(f a, f b) { return _mm256_mul_ps(a, b); }
  static f div(f a, f b) { return _mm256_div_ps(a, b); }
#ifdef KV_HAS_FMA
  static f fma(f a, f b, f c) { return _mm256_fmadd_ps(a, b, c); }
#else
  static f fma(f a, f b, f c) { return _mm256_add_ps(_mm256_mul_ps(a, b), c); }
#endif
  static f gt(f a, f b) { return _mm256_cmp_ps(a, b, _CMP_GT_OQ); }
  static f lt(f a, f b) { return _mm256_cmp_ps(a, b, _CMP_LT_OQ); }
  static f sel(f m, f a, f b) { return _mm256_blendv_ps(b, a, m); }
  static f band(f a, f b) { return _mm256_and_ps(a, b); }
  static f neg(f a) { return _mm256_xor_ps(a, _mm256_set1_ps(-0.f)); }
  static i cvti(f a) { return _mm256_cvttps_epi32(a); }
  static f cvtf(i a) { return _mm256_cvtepi32_ps(a); }
  static i iset1(int s) { return _mm256_set1_epi32(s); }
  static i iadd(i a, i b) { return _mm256_add_epi32(a, b); }
  static i isub(i a, i b) { return _mm256_sub_epi32(a, b); }
  static i iand(i a, i b) { return _mm256_and_si256(a, b); }
  static i ior(i a, i b) { return _mm256_or_si256(a, b); }
  static i ishl(i a, int n) { return _mm256_slli_epi32(a, n); }
  static i ishr(i a, int n) { return _mm256_srai_epi32(a, n); }
};
#endif
#if defined(KV_AVX512)
template <> struct Tr<16> {
  typedef __m512 f; typedef __m512i i;
  static f zero() { return _mm512_setzero_ps(); }
  static f set1(float s) { return _mm512_set1_ps(s); }
  static f add(f a, f b) { return _mm512_add_ps(a, b); }
  static f sub(f a, f b) { return _mm512_sub_ps(a, b); }
  static f mul(f a, f b) { return _mm512_mul_ps(a, b); }
  static f div(f a, f b) { return _mm512_div_ps(a, b); }
  static f fma(f a, f b, f c) { return _mm512_fmadd_ps(a, b, c); }
  // masks are kept as all-ones / zero float lanes like the narrower widths
  static f fromk(__mmask16 k) { return _mm512_castsi512_ps(_mm512_maskz_set1_epi32(k, -1)); }
  static __mmask16 tok(f m) { return _mm512_test_epi32_mask(_mm512_castps_si512(m), _mm512_castps_si512(m)); }
  static f gt(f a, f b) { return fromk(_mm512_cmp_ps_mask(a, b, _CMP_GT_OQ)); }
  static f lt(f a, f b) { return fromk(_mm512_cmp_ps_mask(a, b, _CMP_LT_OQ)); }
  static f sel(f m, f a, f b) { return _mm512_mask_blend_ps(tok(m), b, a); }
  static f band(f a, f b) { return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a), _mm512_castps_si512(b))); }
  static f neg(f a) { return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a), _mm512_set1_epi32((int)0x80000000u))); }
  static i cvti(f a) { return _mm512_cvttps_epi32(a); }
  static f cvtf(i a) { return _mm512_cvtepi32_ps(a); }
  static i iset1(int s) { return _mm512_set1_epi32(s); }
  static i iadd(i a, i b) { return _mm512_add_epi32(a, b); }
  static i isub(i a, i b) { return _mm512_sub_epi32(a, b); }
  static i iand(i a, i b) { return _mm512_and_si512(a, b); }
  static i ior(i a, i b) { return _mm512_or_si512(a, b); }
  static i ishl(i a, int n) { return _mm512_slli_epi32(a, n); }
  static i ishr(i a, int n) { return _mm512_srai_epi32(a, n); }
};
#endif
// scalar lane (VW = 1): masks are all-ones / zero bit patterns stored in a float
template <> struct Tr<1> {
  typedef float f; typedef int32_t i;
  static float bits(uint32_t u) { float r; std::memcpy(&r, &u, 4); return r; }
  static uint32_t ubits(float x) { uint32_t r; std::memcpy(&r, &x, 4); return r; }
  static f zero() { return 0.f; }
  static f set1(float s) { return s; }
  static f add(f a, f b) { return a + b; }
  static f sub(f a, f b) { return a - b; }
  static f mul(f a, f b) { return a * b; }
  static f div(f a, f b) { return a / b; }
  static f fma(f a, f b, f c) { return a * b + c; }  // scalar TU: no FMA (matches GCC baseline x86)
  static f gt(f a, f b) { return bits(a > b ? 0xffffffffu : 0u); }
  static f lt(f a, f b) { return bits(a < b ? 0xffffffffu : 0u); }
  static f sel(f m, f a, f b) { return ubits(m) ? a : b; }
  static f band(f a, f b) { return bits(ubits(a) & ubits(b)); }
  static f neg(f a) { return -a; }
  static i cvti(f a) { return (int32_t)a; }
  static f cvtf(i a) { return (float)a; }
  static i iset1(int s) { return s; }
  static i iadd(i a, i b) { return (int32_t)((uint32_t)a + (uint32_t)b); }
  static i isub(i a, i b) { return (int32_t)((uint32_t)a - (uint32_t)b); }
  static i iand(i a, i b) { return a & b; }
  static i ior(i a, i b) { return a | b; }
  static i ishl(i a, int n) { return (int32_t)((uint32_t)a << n); }
  static i ishr(i a, int n) { return a >> n; }
};

template <int N> struct M { typename Tr<N>::f v; };
template <int N> struct F {
  typedef Tr<N> X;
  typename X::f v;
  F() = default;
  F(typename X::f x) : v(x) {}
  float operator[](int j) const { float t[N]; std::memcpy(t, &v, sizeof t); return t[j]; }
  void set(int j, float s) { float t[N]; std::memcpy(t, &v, sizeof t); t[j] = s; std::memcpy(&v, t, sizeof t); }
  F& operator+=(F b) { v = X::add(v, b.v); return *this; }
  F& operator+=(float b) { v = X::add(v, X::set1(b)); return *this; }
};
template <int N> struct I {
  typedef Tr<N> X;
  typename X::i v;
  I() = default;
  I(typename X::i x) : v(x) {}
};
template <int N> inline F<N> zero() { return F<N>(Tr<N>::zero()); }
template <int N> inline F<N> bcast(float s) { return F<N>(Tr<N>::set1(s)); }
#define KV_BIN(op, fn)                                                                                     \
  template <int N> inline F<N> operator op(F<N> a, F<N> b) { return Tr<N>::fn(a.v, b.v); }                \
  template <int N> inline F<N> operator op(F<N> a, float b) { return Tr<N>::fn(a.v, Tr<N>::set1(b)); }     \
  template <int N> inline F<N> operator op(float a, F<N> b) { return Tr<N>::fn(Tr<N>::set1(a), b.v); }
KV_BIN(+, add) KV_BIN(-, sub) KV_BIN(*, mul) KV_BIN(/, div)
#undef KV_BIN
template <int N> inline F<N> operator-(F<N> a) { return Tr<N>::neg(a.v); }
template <int N> inline M<N> operator>(F<N> a, F<N> b) { return {Tr<N>::gt(a.v, b.v)}; }
template <int N> inline M<N> operator<(F<N> a, F<N> b) { return {Tr<N>::lt(a.v, b.v)}; }
template <int N> inline M<N> operator>(F<N> a, float b) { return {Tr<N>::gt(a.v, Tr<N>::set1(b))}; }
template <int N> inline M<N> operator<(F<N> a, float b) { return {Tr<N>::lt(a.v, Tr<N>::set1(b))}; }
template <int N> inline F<N> sel(M<N> m, F<N> a, F<N> b) { return Tr<N>::sel(m.v, a.v, b.v); }
template <int N> inline F<N> maskf(M<N> m) { return Tr<N>::band(m.v, Tr<N>::set1(-1.f)); }  // true -> -1.0f
template <int N> inline I<N> cvti(F<N> a) { return Tr<N>::cvti(a.v); }
template <int N> inline F<N> cvtf(I<N> a) { return Tr<N>::cvtf(a.v); }
template <int N> inline typename Tr<N>::f raw(F<N> a) { return a.v; }
template <int N> inline typename Tr<N>::f raw(float a) { return Tr<N>::set1(a); }
template <typename A, typename B, typename C> struct W3 {
  static constexpr int n = 0;
};
template <int N, typename B, typename C> struct W3<F<N>, B, C> { static constexpr int n = N; };
template <typename A, int N, typename C> struct W3<A, F<N>, C> { static constexpr int n = N; };
template <int N, typename C> struct W3<F<N>, F<N>, C> { static constexpr int n = N; };
template <typename A, typename B, typename C, int N = W3<A, B, C>::n>
inline F<N> fma(A a, B b, C c) { return Tr<N>::fma(raw<N>(a), raw<N>(b), raw<N>(c)); }
template <int N> inline I<N> operator+(I<N> a, int b) { return Tr<N>::iadd(a.v, Tr<N>::iset1(b)); }
template <int N> inline I<N> operator-(I<N> a, int b) { return Tr<N>::isub(a.v, Tr<N>::iset1(b)); }
template <int N> inline I<N> operator&(I<N> a, int b) { return Tr<N>::iand(a.v, Tr<N>::iset1(b)); }
template <int N> inline I<N> operator|(I<N> a, int b) { return Tr<N>::ior(a.v, Tr<N>::iset1(b)); }
template <int N> inline I<N> operator<<(I<N> a, int n) { return Tr<N>::ishl(a.v, n); }
template <int N> inline I<N> operator>>(I<N> a, int n) { return Tr<N>::ishr(a.v, n); }
template <int N> struct T { typedef F<N> vf; typedef I<N> vi; };
#define KV_SEL(m, a, b) ::kv::sel((m), (a), (b))
#define KV_CVTI(x, VI) ::kv::cvti(x)
#define KV_CVTF(x, VF) ::kv::cvtf(x)
#define KV_MASKF(m, VF) ::kv::maskf(m)
#define KV_FMA(a, b, c) ::kv::fma((a), (b), (c))
#ifdef KV_HAS_FMA
#define KV_SFMA(a, b, c) std::fma((float)(a), (float)(b), (float)(c))  // the FMA TUs fuse scalar a * b + c too
#else
#define KV_SFMA(a, b, c) ((a) * (b) + (c))
#endif
}  // namespace kv
#endif
