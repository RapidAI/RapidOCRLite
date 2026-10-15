// LoongArch LASX (256-bit) kernels. Built only for loongarch64 with -mlasx and
// entered only after the HWCAP_LOONGARCH_LASX check (kernels.cpp HasLasx), so
// the rest of the library stays runnable on LSX-only cores.
//
// The GEMM mirrors NeonGemmRows (2 rows x 32 columns instead of 2 x 16):
// every output element is bias + sum_k left[k] * right[k] accumulated in k
// order with a separate multiply and add, exactly like the 4-lane LSX/NEON
// path, so LASX and LSX give bit-identical results. Columns past the last
// 16-column block use the same scalar tail expression as kernels.cpp.
#include <lasxintrin.h>
#include <lsxintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ppocr::detail::kernels {
namespace {

inline __m256 Load(const float* p) noexcept { return (__m256)__lasx_xvld(p, 0); }
inline void Store(float* p, __m256 v) noexcept { __lasx_xvst((__m256i)v, p, 0); }
inline __m256 Splat(float x) noexcept { return (__m256){x, x, x, x, x, x, x, x}; }
// a + b * c, two roundings (matches vmlaq_f32 in the LSX shim).
inline __m256 Mla(__m256 a, __m256 b, __m256 c) noexcept {
  return __lasx_xvfadd_s(a, __lasx_xvfmul_s(b, c));
}

template <bool Accumulate>
void GemmRowsImpl(float* dst, const float* a, const float* b, const float* bias,
                  int first_row, int last_row, int cols, int depth) noexcept {
  // Columns the 4-lane path vectorizes: whole 16-column blocks.
  const int vector_cols = cols / 16 * 16;
  int row = first_row;
  for (; row + 1 < last_row; row += 2) {
    float* out0 = dst + std::size_t(row) * cols;
    float* out1 = out0 + cols;
    const float* left0 = a + std::size_t(row) * depth;
    const float* left1 = left0 + depth;
    int col = 0;
    for (; col + 32 <= vector_cols; col += 32) {
      __m256 a0, a1, a2, a3, b0, b1, b2, b3;
      if (Accumulate) {
        a0 = Load(out0 + col); a1 = Load(out0 + col + 8);
        a2 = Load(out0 + col + 16); a3 = Load(out0 + col + 24);
        b0 = Load(out1 + col); b1 = Load(out1 + col + 8);
        b2 = Load(out1 + col + 16); b3 = Load(out1 + col + 24);
      } else {
        a0 = bias ? Load(bias + col) : Splat(0.F);
        a1 = bias ? Load(bias + col + 8) : Splat(0.F);
        a2 = bias ? Load(bias + col + 16) : Splat(0.F);
        a3 = bias ? Load(bias + col + 24) : Splat(0.F);
        b0 = a0; b1 = a1; b2 = a2; b3 = a3;
      }
      for (int k = 0; k < depth; ++k) {
        const __m256 l0 = Splat(left0[k]);
        const __m256 l1 = Splat(left1[k]);
        const float* right = b + std::size_t(k) * cols + col;
        const __m256 r0 = Load(right), r1 = Load(right + 8);
        const __m256 r2 = Load(right + 16), r3 = Load(right + 24);
        a0 = Mla(a0, l0, r0); b0 = Mla(b0, l1, r0);
        a1 = Mla(a1, l0, r1); b1 = Mla(b1, l1, r1);
        a2 = Mla(a2, l0, r2); b2 = Mla(b2, l1, r2);
        a3 = Mla(a3, l0, r3); b3 = Mla(b3, l1, r3);
      }
      Store(out0 + col, a0); Store(out0 + col + 8, a1);
      Store(out0 + col + 16, a2); Store(out0 + col + 24, a3);
      Store(out1 + col, b0); Store(out1 + col + 8, b1);
      Store(out1 + col + 16, b2); Store(out1 + col + 24, b3);
    }
    for (; col + 8 <= vector_cols; col += 8) {
      __m256 c0, c1;
      if (Accumulate) {
        c0 = Load(out0 + col); c1 = Load(out1 + col);
      } else {
        c0 = bias ? Load(bias + col) : Splat(0.F);
        c1 = c0;
      }
      for (int k = 0; k < depth; ++k) {
        const __m256 r = Load(b + std::size_t(k) * cols + col);
        c0 = Mla(c0, Splat(left0[k]), r);
        c1 = Mla(c1, Splat(left1[k]), r);
      }
      Store(out0 + col, c0); Store(out1 + col, c1);
    }
    for (; col < cols; ++col) {
      float v0 = Accumulate ? out0[col] : (bias ? bias[col] : 0.F);
      float v1 = Accumulate ? out1[col] : v0;
      for (int k = 0; k < depth; ++k) {
        const float weight = b[std::size_t(k) * cols + col];
        v0 += left0[k] * weight;
        v1 += left1[k] * weight;
      }
      out0[col] = v0;
      out1[col] = v1;
    }
  }
  for (; row < last_row; ++row) {
    float* out = dst + std::size_t(row) * cols;
    const float* left = a + std::size_t(row) * depth;
    int col = 0;
    for (; col + 8 <= vector_cols; col += 8) {
      __m256 c = Accumulate ? Load(out + col) : (bias ? Load(bias + col) : Splat(0.F));
      for (int k = 0; k < depth; ++k)
        c = Mla(c, Splat(left[k]), Load(b + std::size_t(k) * cols + col));
      Store(out + col, c);
    }
    for (; col < cols; ++col) {
      float value = Accumulate ? out[col] : (bias ? bias[col] : 0.F);
      for (int k = 0; k < depth; ++k) value += left[k] * b[std::size_t(k) * cols + col];
      out[col] = value;
    }
  }
}

}  // namespace

void LasxGemmRows(float* dst, const float* a, const float* b, const float* bias,
                  int first_row, int last_row, int cols, int depth) noexcept {
  GemmRowsImpl<false>(dst, a, b, bias, first_row, last_row, cols, depth);
}

void LasxGemmAccumulateRows(float* dst, const float* a, const float* b,
                            int first_row, int last_row, int cols, int depth) noexcept {
  GemmRowsImpl<true>(dst, a, b, nullptr, first_row, last_row, cols, depth);
}

namespace {

// Lane-wise twins of NeonExpPs / NeonErfPs / NeonGeluPs (same operations in
// the same order, fused where NEON fuses), so every lane is bit-identical.
inline __m256 Fma(__m256 a, __m256 b, __m256 c) noexcept {  // a + b * c
  return __lasx_xvfmadd_s(b, c, a);
}
inline __m256 Fms(__m256 a, __m256 b, __m256 c) noexcept {  // a - b * c
  return __lasx_xvfnmsub_s(b, c, a);
}
inline __m256 ExpPs(__m256 x) noexcept {
  x = __lasx_xvfmin_s(__lasx_xvfmax_s(x, Splat(-87.F)), Splat(87.F));
  const __m256 n = __lasx_xvfrintrne_s(__lasx_xvfmul_s(x, Splat(1.4426950408889634F)));
  const __m256 f = Fms(x, n, Splat(0.6931471805599453F));
  __m256 p = Fma(Splat(1.F / 6.F), f, Splat(1.F / 24.F));
  p = Fma(Splat(.5F), p, f);
  p = Fma(Splat(1.F), p, f);
  p = Fma(Splat(1.F), p, f);
  const __m256i biased = __lasx_xvadd_w(__lasx_xvftintrz_w_s(n), __lasx_xvreplgr2vr_w(127));
  return __lasx_xvfmul_s(p, (__m256)__lasx_xvslli_w(biased, 23));
}
inline __m256 ErfPs(__m256 x) noexcept {
  const __m256 one = Splat(1.F);
  const __m256i bits = (__m256i)x;
  const __m256 sign = (__m256)__lasx_xvor_v(
      __lasx_xvand_v(bits, __lasx_xvreplgr2vr_w(int(0x80000000u))), (__m256i)one);
  const __m256 ax = (__m256)__lasx_xvbitclri_w(bits, 31);
  const __m256 t = __lasx_xvfdiv_s(one, Fma(one, Splat(.5F), ax));
  __m256 poly = Fma(Splat(-.82215223F), t, Splat(.17087277F));
  poly = Fma(Splat(1.48851587F), poly, t);
  poly = Fma(Splat(-1.13520398F), poly, t);
  poly = Fma(Splat(.27886807F), poly, t);
  poly = Fma(Splat(-.18628806F), poly, t);
  poly = Fma(Splat(.09678418F), poly, t);
  poly = Fma(Splat(.37409196F), poly, t);
  poly = Fma(Splat(1.00002368F), poly, t);
  poly = Fma(Splat(-1.26551223F), poly, t);
  const __m256 e = ExpPs(Fma(poly, __lasx_xvfmul_s(ax, ax), Splat(-1.F)));
  return __lasx_xvfmul_s(sign, Fms(one, t, e));
}
inline __m256 GeluPs(__m256 x) noexcept {
  const __m256 erf = ErfPs(__lasx_xvfmul_s(x, Splat(0.7071067811865475244F)));
  return __lasx_xvfmul_s(Splat(.5F), __lasx_xvfmul_s(x, __lasx_xvfadd_s(Splat(1.F), erf)));
}

}  // namespace

// Same element split as the NEON loop: vector lanes for the leading
// multiple of 4, scalar erf for the last count % 4 elements.
void LasxExactGeluRange(float* dst, const float* src, std::size_t count) noexcept {
  std::size_t i = 0;
  for (; i + 8 <= count; i += 8) Store(dst + i, GeluPs(Load(src + i)));
  const std::size_t vec_end = count / 4 * 4;
  if (i < vec_end) {
    alignas(32) float tmp[8] = {};
    for (std::size_t j = i; j < vec_end; ++j) tmp[j - i] = src[j];
    Store(tmp, GeluPs(Load(tmp)));
    for (std::size_t j = i; j < vec_end; ++j) dst[j] = tmp[j - i];
    i = vec_end;
  }
  for (; i < count; ++i) {
    const float x = src[i];
    dst[i] = x * .5F * (1.F + std::erf(x * 0.7071067811865475244F));
  }
}

// First index of the maximum, same result as the scalar strict-> scan.
int LasxArgMax(const float* values, int count) noexcept {
  int i = 0;
  float best = values[0];
  if (count >= 8) {
    __m256 m = Load(values);
    for (i = 8; i + 8 <= count; i += 8) m = __lasx_xvfmax_s(m, Load(values + i));
    alignas(32) float lanes[8];
    Store(lanes, m);
    for (float v : lanes) best = std::max(best, v);
  }
  for (; i < count; ++i) best = std::max(best, values[i]);
  for (int j = 0; j < count; ++j) {
    if (values[j] == best) return j;
  }
  return 0;
}

// Mirrors NeonDepthwiseConv: each output is base + sum over (ky, kx) in
// ascending order with fused multiply-add, so lanes match the 4-wide path.
void LasxDepthwiseConv(float* dst, const float* src, const float* weights, const float* bias,
                       int first_channel, int last_channel, int input_h, int input_w,
                       int output_h, int output_w, int kernel_h, int kernel_w, int pad_top,
                       int pad_left) noexcept {
  const std::size_t input_plane = std::size_t(input_h) * input_w;
  const std::size_t output_plane = std::size_t(output_h) * output_w;
  const std::size_t kernel_plane = std::size_t(kernel_h) * kernel_w;
  const int first_y = std::max(0, pad_top);
  const int first_x = std::max(0, pad_left);
  const int last_y = std::min(output_h, input_h + pad_top - kernel_h + 1);
  const int last_x = std::min(output_w, input_w + pad_left - kernel_w + 1);
  const auto edge = [&](const float* in, const float* filter, float base, int y, int x) {
    float sum = base;
    const int iy0 = y - pad_top, ix0 = x - pad_left;
    for (int ky = 0; ky < kernel_h; ++ky) {
      const int iy = iy0 + ky;
      if (iy < 0 || iy >= input_h) continue;
      for (int kx = 0; kx < kernel_w; ++kx) {
        const int ix = ix0 + kx;
        if (ix >= 0 && ix < input_w) sum += in[std::size_t(iy) * input_w + ix] * filter[ky * kernel_w + kx];
      }
    }
    return sum;
  };
  for (int channel = first_channel; channel < last_channel; ++channel) {
    const float* in = src + std::size_t(channel) * input_plane;
    const float* filter = weights + std::size_t(channel) * kernel_plane;
    float* out = dst + std::size_t(channel) * output_plane;
    const float base = bias ? bias[channel] : 0.F;
    for (int y = 0; y < output_h; ++y) {
      const int iy0 = y - pad_top;
      const bool inside = y >= first_y && y < last_y;
      const int interior_begin = inside ? first_x : 0;
      const int interior_end = inside ? last_x : 0;
      float* o = out + std::size_t(y) * output_w;
      int x = 0;
      for (; x < interior_begin; ++x) o[x] = edge(in, filter, base, y, x);
      for (; x + 32 <= interior_end; x += 32) {
        __m256 s0 = Splat(base), s1 = s0, s2 = s0, s3 = s0;
        for (int ky = 0; ky < kernel_h; ++ky) {
          const float* row = in + std::size_t(iy0 + ky) * input_w + x - pad_left;
          const float* kernel = filter + ky * kernel_w;
          for (int kx = 0; kx < kernel_w; ++kx) {
            const __m256 w = Splat(kernel[kx]);
            s0 = Fma(s0, Load(row + kx), w);
            s1 = Fma(s1, Load(row + kx + 8), w);
            s2 = Fma(s2, Load(row + kx + 16), w);
            s3 = Fma(s3, Load(row + kx + 24), w);
          }
        }
        Store(o + x, s0); Store(o + x + 8, s1); Store(o + x + 16, s2); Store(o + x + 24, s3);
      }
      for (; x + 8 <= interior_end; x += 8) {
        __m256 sum = Splat(base);
        for (int ky = 0; ky < kernel_h; ++ky) {
          const float* row = in + std::size_t(iy0 + ky) * input_w + x - pad_left;
          const float* kernel = filter + ky * kernel_w;
          for (int kx = 0; kx < kernel_w; ++kx) sum = Fma(sum, Load(row + kx), Splat(kernel[kx]));
        }
        Store(o + x, sum);
      }
      for (; x + 4 <= interior_end; x += 4) {
        __m128 sum = (__m128){base, base, base, base};
        for (int ky = 0; ky < kernel_h; ++ky) {
          const float* row = in + std::size_t(iy0 + ky) * input_w + x - pad_left;
          const float* kernel = filter + ky * kernel_w;
          for (int kx = 0; kx < kernel_w; ++kx) {
            const float k = kernel[kx];
            sum = __lsx_vfmadd_s((__m128)__lsx_vld(row + kx, 0), (__m128){k, k, k, k}, sum);
          }
        }
        __lsx_vst((__m128i)sum, o + x, 0);
      }
      for (; x < output_w; ++x) o[x] = edge(in, filter, base, y, x);
    }
  }
}

}  // namespace ppocr::detail::kernels
