#pragma once
// LoongArch LSX implementation of the small set of NEON intrinsics used by
// kernels.cpp, so the NEON code paths compile unchanged for loongarch64
// (128-bit LSX, 4 float lanes, same operation order). Fused multiply-add maps
// to vfmadd/vfnmsub (single rounding, as vfmaq/vfmsq). Only the intrinsics
// the kernels use are provided; adding a NEON intrinsic to kernels.cpp needs
// a matching entry here (the loongarch64 CI build catches a missing one).
#if !defined(__loongarch_sx)
#error "neon_lsx_shim.hpp needs LSX (-mlsx)"
#endif
#include <lsxintrin.h>

#include <cstdint>
#include <cstring>

typedef float float32x4_t __attribute__((vector_size(16)));
typedef std::int32_t int32x4_t __attribute__((vector_size(16)));
typedef std::uint32_t uint32x4_t __attribute__((vector_size(16)));
typedef std::uint16_t uint16x8_t __attribute__((vector_size(16)));
typedef std::uint16_t uint16x4_t __attribute__((vector_size(8)));
typedef std::uint8_t uint8x16_t __attribute__((vector_size(16)));
typedef std::uint8_t uint8x8_t __attribute__((vector_size(8)));
struct float32x4x2_t { float32x4_t val[2]; };

#define PPOCR_LSX_INLINE static inline __attribute__((always_inline))

PPOCR_LSX_INLINE float32x4_t vdupq_n_f32(float x) { return float32x4_t{x, x, x, x}; }
PPOCR_LSX_INLINE int32x4_t vdupq_n_s32(std::int32_t x) { return int32x4_t{x, x, x, x}; }
PPOCR_LSX_INLINE uint32x4_t vdupq_n_u32(std::uint32_t x) { return uint32x4_t{x, x, x, x}; }
PPOCR_LSX_INLINE float32x4_t vld1q_f32(const float* p) {
  float32x4_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
PPOCR_LSX_INLINE void vst1q_f32(float* p, float32x4_t v) { std::memcpy(p, &v, sizeof(v)); }
PPOCR_LSX_INLINE uint8x16_t vld1q_u8(const std::uint8_t* p) {
  uint8x16_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}
PPOCR_LSX_INLINE float32x4x2_t vld2q_f32(const float* p) {
  const float32x4_t a = vld1q_f32(p), b = vld1q_f32(p + 4);
  return {{__builtin_shufflevector(a, b, 0, 2, 4, 6), __builtin_shufflevector(a, b, 1, 3, 5, 7)}};
}
PPOCR_LSX_INLINE float32x4x2_t vzipq_f32(float32x4_t a, float32x4_t b) {
  return {{__builtin_shufflevector(a, b, 0, 4, 1, 5), __builtin_shufflevector(a, b, 2, 6, 3, 7)}};
}
PPOCR_LSX_INLINE float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) { return a + b; }
PPOCR_LSX_INLINE float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) { return a - b; }
PPOCR_LSX_INLINE float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) { return a * b; }
PPOCR_LSX_INLINE float32x4_t vdivq_f32(float32x4_t a, float32x4_t b) { return a / b; }
// a + b * c, one rounding.
PPOCR_LSX_INLINE float32x4_t vfmaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
  return (float32x4_t)__lsx_vfmadd_s((__m128)b, (__m128)c, (__m128)a);
}
// a + b * c with two roundings (vmlaq is not fused on AArch64 either).
PPOCR_LSX_INLINE float32x4_t vmlaq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
  const __m128 product = __lsx_vfmul_s((__m128)b, (__m128)c);
  return (float32x4_t)__lsx_vfadd_s((__m128)a, product);
}
// a - b * c, one rounding.
PPOCR_LSX_INLINE float32x4_t vfmsq_f32(float32x4_t a, float32x4_t b, float32x4_t c) {
  return (float32x4_t)__lsx_vfnmsub_s((__m128)b, (__m128)c, (__m128)a);
}
PPOCR_LSX_INLINE float32x4_t vmaxq_f32(float32x4_t a, float32x4_t b) {
  return (float32x4_t)__lsx_vfmax_s((__m128)a, (__m128)b);
}
PPOCR_LSX_INLINE float32x4_t vminq_f32(float32x4_t a, float32x4_t b) {
  return (float32x4_t)__lsx_vfmin_s((__m128)a, (__m128)b);
}
PPOCR_LSX_INLINE float32x4_t vabsq_f32(float32x4_t a) {
  return (float32x4_t)((uint32x4_t)a & uint32x4_t{0x7fffffffu, 0x7fffffffu, 0x7fffffffu, 0x7fffffffu});
}
PPOCR_LSX_INLINE float32x4_t vrndnq_f32(float32x4_t a) {
  return (float32x4_t)__lsx_vfrintrne_s((__m128)a);
}
PPOCR_LSX_INLINE int32x4_t vcvtq_s32_f32(float32x4_t a) {
  return (int32x4_t)__lsx_vftintrz_w_s((__m128)a);
}
PPOCR_LSX_INLINE float32x4_t vcvtq_f32_u32(uint32x4_t a) {
  return (float32x4_t)__lsx_vffint_s_wu((__m128i)a);
}
PPOCR_LSX_INLINE int32x4_t vaddq_s32(int32x4_t a, int32x4_t b) { return a + b; }
#define vshlq_n_s32(a, n) ((int32x4_t)((a) << (n)))
PPOCR_LSX_INLINE uint32x4_t vorrq_u32(uint32x4_t a, uint32x4_t b) { return a | b; }
PPOCR_LSX_INLINE uint32x4_t vandq_u32(uint32x4_t a, uint32x4_t b) { return a & b; }
PPOCR_LSX_INLINE uint32x4_t vreinterpretq_u32_f32(float32x4_t a) { return (uint32x4_t)a; }
PPOCR_LSX_INLINE float32x4_t vreinterpretq_f32_u32(uint32x4_t a) { return (float32x4_t)a; }
PPOCR_LSX_INLINE float32x4_t vreinterpretq_f32_s32(int32x4_t a) { return (float32x4_t)a; }
PPOCR_LSX_INLINE uint32x4_t vcgtq_f32(float32x4_t a, float32x4_t b) { return (uint32x4_t)(a > b); }
PPOCR_LSX_INLINE uint32x4_t vceqq_f32(float32x4_t a, float32x4_t b) { return (uint32x4_t)(a == b); }
PPOCR_LSX_INLINE std::uint32_t vmaxvq_u32(uint32x4_t a) {
  std::uint32_t m = a[0] > a[1] ? a[0] : a[1];
  m = m > a[2] ? m : a[2];
  return m > a[3] ? m : a[3];
}
#define vgetq_lane_u32(v, lane) ((std::uint32_t)(v)[lane])
PPOCR_LSX_INLINE uint8x8_t vget_low_u8(uint8x16_t a) {
  return __builtin_shufflevector(a, a, 0, 1, 2, 3, 4, 5, 6, 7);
}
PPOCR_LSX_INLINE uint8x8_t vget_high_u8(uint8x16_t a) {
  return __builtin_shufflevector(a, a, 8, 9, 10, 11, 12, 13, 14, 15);
}
PPOCR_LSX_INLINE uint16x4_t vget_low_u16(uint16x8_t a) { return __builtin_shufflevector(a, a, 0, 1, 2, 3); }
PPOCR_LSX_INLINE uint16x4_t vget_high_u16(uint16x8_t a) { return __builtin_shufflevector(a, a, 4, 5, 6, 7); }
PPOCR_LSX_INLINE uint16x8_t vmovl_u8(uint8x8_t a) { return __builtin_convertvector(a, uint16x8_t); }
PPOCR_LSX_INLINE uint32x4_t vmovl_u16(uint16x4_t a) { return __builtin_convertvector(a, uint32x4_t); }
