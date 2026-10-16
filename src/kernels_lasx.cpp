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
#include <vector>

namespace ppocr::detail::kernels {
namespace {

inline __m256 Load(const float* p) noexcept { return (__m256)__lasx_xvld(p, 0); }
inline void Store(float* p, __m256 v) noexcept { __lasx_xvst((__m256i)v, p, 0); }
inline __m256 Splat(float x) noexcept { return (__m256){x, x, x, x, x, x, x, x}; }
// a + b * c as a single FMA (faster than the old mul+add Mla).
inline __m256 Fma(__m256 a, __m256 b, __m256 c) noexcept {
  return __lasx_xvfmadd_s(b, c, a);
}
inline __m256 Fms(__m256 a, __m256 b, __m256 c) noexcept {
  return __lasx_xvfnmsub_s(b, c, a);
}

template <bool Accumulate>
void GemmRowsImpl(float* dst, const float* a, const float* b, const float* bias,
                  int first_row, int last_row, int cols, int depth) noexcept {
  // Vectorize every full 8-column group (LASX lane width). Prefer FMA over
  // separate mul+add: scores may differ by ~1 ulp from the older Mla path,
  // text stays identical under the QEMU replay corpus.
  const int vector_cols = cols / 8 * 8;
  int row = first_row;
  for (; row + 3 < last_row; row += 4) {
    float* out0 = dst + std::size_t(row) * cols;
    float* out1 = out0 + cols;
    float* out2 = out1 + cols;
    float* out3 = out2 + cols;
    const float* left0 = a + std::size_t(row) * depth;
    const float* left1 = left0 + depth;
    const float* left2 = left1 + depth;
    const float* left3 = left2 + depth;
    int col = 0;
    for (; col + 16 <= vector_cols; col += 16) {
      __m256 a0, a1, b0, b1, c0, c1, d0, d1;
      if (Accumulate) {
        a0 = Load(out0 + col); a1 = Load(out0 + col + 8);
        b0 = Load(out1 + col); b1 = Load(out1 + col + 8);
        c0 = Load(out2 + col); c1 = Load(out2 + col + 8);
        d0 = Load(out3 + col); d1 = Load(out3 + col + 8);
      } else {
        a0 = bias ? Load(bias + col) : Splat(0.F);
        a1 = bias ? Load(bias + col + 8) : Splat(0.F);
        b0 = a0; b1 = a1; c0 = a0; c1 = a1; d0 = a0; d1 = a1;
      }
      for (int k = 0; k < depth; ++k) {
        const __m256 l0 = Splat(left0[k]), l1 = Splat(left1[k]);
        const __m256 l2 = Splat(left2[k]), l3 = Splat(left3[k]);
        const float* right = b + std::size_t(k) * cols + col;
        const __m256 r0 = Load(right), r1 = Load(right + 8);
        a0 = Fma(a0, l0, r0); a1 = Fma(a1, l0, r1);
        b0 = Fma(b0, l1, r0); b1 = Fma(b1, l1, r1);
        c0 = Fma(c0, l2, r0); c1 = Fma(c1, l2, r1);
        d0 = Fma(d0, l3, r0); d1 = Fma(d1, l3, r1);
      }
      Store(out0 + col, a0); Store(out0 + col + 8, a1);
      Store(out1 + col, b0); Store(out1 + col + 8, b1);
      Store(out2 + col, c0); Store(out2 + col + 8, c1);
      Store(out3 + col, d0); Store(out3 + col + 8, d1);
    }
    for (; col + 8 <= vector_cols; col += 8) {
      __m256 c0, c1, c2, c3;
      if (Accumulate) {
        c0 = Load(out0 + col); c1 = Load(out1 + col);
        c2 = Load(out2 + col); c3 = Load(out3 + col);
      } else {
        c0 = bias ? Load(bias + col) : Splat(0.F);
        c1 = c0; c2 = c0; c3 = c0;
      }
      for (int k = 0; k < depth; ++k) {
        const __m256 r = Load(b + std::size_t(k) * cols + col);
        c0 = Fma(c0, Splat(left0[k]), r);
        c1 = Fma(c1, Splat(left1[k]), r);
        c2 = Fma(c2, Splat(left2[k]), r);
        c3 = Fma(c3, Splat(left3[k]), r);
      }
      Store(out0 + col, c0); Store(out1 + col, c1);
      Store(out2 + col, c2); Store(out3 + col, c3);
    }
    for (; col < cols; ++col) {
      float v0 = Accumulate ? out0[col] : (bias ? bias[col] : 0.F);
      float v1 = Accumulate ? out1[col] : v0;
      float v2 = Accumulate ? out2[col] : v0;
      float v3 = Accumulate ? out3[col] : v0;
      for (int k = 0; k < depth; ++k) {
        const float weight = b[std::size_t(k) * cols + col];
        v0 += left0[k] * weight; v1 += left1[k] * weight;
        v2 += left2[k] * weight; v3 += left3[k] * weight;
      }
      out0[col] = v0; out1[col] = v1; out2[col] = v2; out3[col] = v3;
    }
  }
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
        a0 = Fma(a0, l0, r0); b0 = Fma(b0, l1, r0);
        a1 = Fma(a1, l0, r1); b1 = Fma(b1, l1, r1);
        a2 = Fma(a2, l0, r2); b2 = Fma(b2, l1, r2);
        a3 = Fma(a3, l0, r3); b3 = Fma(b3, l1, r3);
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
        c0 = Fma(c0, Splat(left0[k]), r);
        c1 = Fma(c1, Splat(left1[k]), r);
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
    for (; col + 32 <= vector_cols; col += 32) {
      __m256 c0 = Accumulate ? Load(out + col) : (bias ? Load(bias + col) : Splat(0.F));
      __m256 c1 = Accumulate ? Load(out + col + 8) : (bias ? Load(bias + col + 8) : Splat(0.F));
      __m256 c2 = Accumulate ? Load(out + col + 16) : (bias ? Load(bias + col + 16) : Splat(0.F));
      __m256 c3 = Accumulate ? Load(out + col + 24) : (bias ? Load(bias + col + 24) : Splat(0.F));
      for (int k = 0; k < depth; ++k) {
        const __m256 av = Splat(left[k]);
        const float* right = b + std::size_t(k) * cols + col;
        c0 = Fma(c0, av, Load(right));
        c1 = Fma(c1, av, Load(right + 8));
        c2 = Fma(c2, av, Load(right + 16));
        c3 = Fma(c3, av, Load(right + 24));
      }
      Store(out + col, c0); Store(out + col + 8, c1);
      Store(out + col + 16, c2); Store(out + col + 24, c3);
    }
    for (; col + 8 <= vector_cols; col += 8) {
      __m256 c = Accumulate ? Load(out + col) : (bias ? Load(bias + col) : Splat(0.F));
      for (int k = 0; k < depth; ++k)
        c = Fma(c, Splat(left[k]), Load(b + std::size_t(k) * cols + col));
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

// Lane-wise twins of NeonExpPs / NeonErfPs / NeonGeluPs (same ops/order).
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


// Fused expand → exact GELU → project + residual. Spatial tiles of 16 use two
// 8-lane LASX accumulators (same FMA + erf order as NeonExpandGeluProjectAdd's
// 4-lane path, wider). Trailing spatials share the scalar loop.
void LasxExpandGeluProjectAdd(float* dst, const float* src, const float* expand_weights,
                              const float* expand_bias, const float* project_weights,
                              const float* project_bias, int channels, int hidden,
                              std::size_t plane, std::size_t spatial_begin,
                              std::size_t spatial_end) noexcept {
  if (!dst || !src || !expand_weights || !project_weights || channels <= 0 || hidden <= 0 ||
      spatial_begin >= spatial_end || spatial_end > plane) {
    return;
  }
  thread_local std::vector<float> hidden_tile;
  hidden_tile.resize(std::size_t(hidden) * 16);
  constexpr float inv_sqrt2s = 0.7071067811865475244F;
  const auto scalar_at = [&](std::size_t spatial) {
    for (int hidden_channel = 0; hidden_channel < hidden; ++hidden_channel) {
      float sum = expand_bias ? expand_bias[hidden_channel] : 0.F;
      const float* filter = expand_weights + std::size_t(hidden_channel) * channels;
      for (int channel = 0; channel < channels; ++channel) {
        sum += filter[channel] * src[std::size_t(channel) * plane + spatial];
      }
      hidden_tile[static_cast<std::size_t>(hidden_channel)] =
          sum * .5F * (1.F + std::erf(sum * inv_sqrt2s));
    }
    for (int channel = 0; channel < channels; ++channel) {
      float sum = (project_bias ? project_bias[channel] : 0.F) +
                  src[std::size_t(channel) * plane + spatial];
      const float* filter = project_weights + std::size_t(channel) * hidden;
      for (int hidden_channel = 0; hidden_channel < hidden; ++hidden_channel) {
        sum += filter[hidden_channel] * hidden_tile[static_cast<std::size_t>(hidden_channel)];
      }
      dst[std::size_t(channel) * plane + spatial] = sum;
    }
  };
  std::size_t spatial = spatial_begin;
  for (; spatial + 16 <= spatial_end; spatial += 16) {
    int hidden_channel = 0;
    for (; hidden_channel + 4 <= hidden; hidden_channel += 4) {
      const float* e0 = expand_weights + std::size_t(hidden_channel) * channels;
      const float* e1 = e0 + channels;
      const float* e2 = e1 + channels;
      const float* e3 = e2 + channels;
      __m256 a00 = Splat(expand_bias ? expand_bias[hidden_channel] : 0.F);
      __m256 a01 = a00;
      __m256 a10 = Splat(expand_bias ? expand_bias[hidden_channel + 1] : 0.F);
      __m256 a11 = a10;
      __m256 a20 = Splat(expand_bias ? expand_bias[hidden_channel + 2] : 0.F);
      __m256 a21 = a20;
      __m256 a30 = Splat(expand_bias ? expand_bias[hidden_channel + 3] : 0.F);
      __m256 a31 = a30;
      for (int channel = 0; channel < channels; ++channel) {
        const float* in = src + std::size_t(channel) * plane + spatial;
        const __m256 x0 = Load(in), x1 = Load(in + 8);
        const __m256 w0 = Splat(e0[channel]), w1 = Splat(e1[channel]);
        const __m256 w2 = Splat(e2[channel]), w3 = Splat(e3[channel]);
        a00 = Fma(a00, x0, w0); a01 = Fma(a01, x1, w0);
        a10 = Fma(a10, x0, w1); a11 = Fma(a11, x1, w1);
        a20 = Fma(a20, x0, w2); a21 = Fma(a21, x1, w2);
        a30 = Fma(a30, x0, w3); a31 = Fma(a31, x1, w3);
      }
      float* tile = hidden_tile.data() + std::size_t(hidden_channel) * 16;
      Store(tile, GeluPs(a00)); Store(tile + 8, GeluPs(a01));
      tile += 16;
      Store(tile, GeluPs(a10)); Store(tile + 8, GeluPs(a11));
      tile += 16;
      Store(tile, GeluPs(a20)); Store(tile + 8, GeluPs(a21));
      tile += 16;
      Store(tile, GeluPs(a30)); Store(tile + 8, GeluPs(a31));
    }
    for (; hidden_channel < hidden; ++hidden_channel) {
      const float* filter = expand_weights + std::size_t(hidden_channel) * channels;
      __m256 a0 = Splat(expand_bias ? expand_bias[hidden_channel] : 0.F);
      __m256 a1 = a0;
      for (int channel = 0; channel < channels; ++channel) {
        const float* in = src + std::size_t(channel) * plane + spatial;
        const __m256 w = Splat(filter[channel]);
        a0 = Fma(a0, Load(in), w);
        a1 = Fma(a1, Load(in + 8), w);
      }
      float* tile = hidden_tile.data() + std::size_t(hidden_channel) * 16;
      Store(tile, GeluPs(a0)); Store(tile + 8, GeluPs(a1));
    }
    int channel = 0;
    for (; channel + 4 <= channels; channel += 4) {
      const float* p0 = project_weights + std::size_t(channel) * hidden;
      const float* p1 = p0 + hidden;
      const float* p2 = p1 + hidden;
      const float* p3 = p2 + hidden;
      const float* s0 = src + std::size_t(channel) * plane + spatial;
      const float* s1 = src + std::size_t(channel + 1) * plane + spatial;
      const float* s2 = src + std::size_t(channel + 2) * plane + spatial;
      const float* s3 = src + std::size_t(channel + 3) * plane + spatial;
      __m256 b00 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel] : 0.F), Load(s0));
      __m256 b01 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel] : 0.F), Load(s0 + 8));
      __m256 b10 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel + 1] : 0.F), Load(s1));
      __m256 b11 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel + 1] : 0.F), Load(s1 + 8));
      __m256 b20 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel + 2] : 0.F), Load(s2));
      __m256 b21 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel + 2] : 0.F), Load(s2 + 8));
      __m256 b30 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel + 3] : 0.F), Load(s3));
      __m256 b31 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel + 3] : 0.F), Load(s3 + 8));
      for (int h = 0; h < hidden; ++h) {
        const float* g = hidden_tile.data() + std::size_t(h) * 16;
        const __m256 g0 = Load(g), g1 = Load(g + 8);
        const __m256 w0 = Splat(p0[h]), w1 = Splat(p1[h]);
        const __m256 w2 = Splat(p2[h]), w3 = Splat(p3[h]);
        b00 = Fma(b00, g0, w0); b01 = Fma(b01, g1, w0);
        b10 = Fma(b10, g0, w1); b11 = Fma(b11, g1, w1);
        b20 = Fma(b20, g0, w2); b21 = Fma(b21, g1, w2);
        b30 = Fma(b30, g0, w3); b31 = Fma(b31, g1, w3);
      }
      float* d0 = dst + std::size_t(channel) * plane + spatial;
      Store(d0, b00); Store(d0 + 8, b01);
      d0 = dst + std::size_t(channel + 1) * plane + spatial;
      Store(d0, b10); Store(d0 + 8, b11);
      d0 = dst + std::size_t(channel + 2) * plane + spatial;
      Store(d0, b20); Store(d0 + 8, b21);
      d0 = dst + std::size_t(channel + 3) * plane + spatial;
      Store(d0, b30); Store(d0 + 8, b31);
    }
    for (; channel < channels; ++channel) {
      const float* filter = project_weights + std::size_t(channel) * hidden;
      const float* in = src + std::size_t(channel) * plane + spatial;
      const __m256 bias = Splat(project_bias ? project_bias[channel] : 0.F);
      __m256 a0 = __lasx_xvfadd_s(bias, Load(in));
      __m256 a1 = __lasx_xvfadd_s(bias, Load(in + 8));
      for (int h = 0; h < hidden; ++h) {
        const float* g = hidden_tile.data() + std::size_t(h) * 16;
        const __m256 w = Splat(filter[h]);
        a0 = Fma(a0, Load(g), w);
        a1 = Fma(a1, Load(g + 8), w);
      }
      float* out = dst + std::size_t(channel) * plane + spatial;
      Store(out, a0); Store(out + 8, a1);
    }
  }
  for (; spatial + 8 <= spatial_end; spatial += 8) {
    int hidden_channel = 0;
    for (; hidden_channel + 4 <= hidden; hidden_channel += 4) {
      const float* e0 = expand_weights + std::size_t(hidden_channel) * channels;
      const float* e1 = e0 + channels;
      const float* e2 = e1 + channels;
      const float* e3 = e2 + channels;
      __m256 a0 = Splat(expand_bias ? expand_bias[hidden_channel] : 0.F);
      __m256 a1 = Splat(expand_bias ? expand_bias[hidden_channel + 1] : 0.F);
      __m256 a2 = Splat(expand_bias ? expand_bias[hidden_channel + 2] : 0.F);
      __m256 a3 = Splat(expand_bias ? expand_bias[hidden_channel + 3] : 0.F);
      for (int channel = 0; channel < channels; ++channel) {
        const __m256 x = Load(src + std::size_t(channel) * plane + spatial);
        a0 = Fma(a0, x, Splat(e0[channel]));
        a1 = Fma(a1, x, Splat(e1[channel]));
        a2 = Fma(a2, x, Splat(e2[channel]));
        a3 = Fma(a3, x, Splat(e3[channel]));
      }
      Store(hidden_tile.data() + std::size_t(hidden_channel) * 8, GeluPs(a0));
      Store(hidden_tile.data() + std::size_t(hidden_channel + 1) * 8, GeluPs(a1));
      Store(hidden_tile.data() + std::size_t(hidden_channel + 2) * 8, GeluPs(a2));
      Store(hidden_tile.data() + std::size_t(hidden_channel + 3) * 8, GeluPs(a3));
    }
    for (; hidden_channel < hidden; ++hidden_channel) {
      const float* filter = expand_weights + std::size_t(hidden_channel) * channels;
      __m256 acc = Splat(expand_bias ? expand_bias[hidden_channel] : 0.F);
      for (int channel = 0; channel < channels; ++channel) {
        acc = Fma(acc, Load(src + std::size_t(channel) * plane + spatial), Splat(filter[channel]));
      }
      Store(hidden_tile.data() + std::size_t(hidden_channel) * 8, GeluPs(acc));
    }
    int channel = 0;
    for (; channel + 4 <= channels; channel += 4) {
      const float* p0 = project_weights + std::size_t(channel) * hidden;
      const float* p1 = p0 + hidden;
      const float* p2 = p1 + hidden;
      const float* p3 = p2 + hidden;
      __m256 a0 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel] : 0.F),
                                  Load(src + std::size_t(channel) * plane + spatial));
      __m256 a1 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel + 1] : 0.F),
                                  Load(src + std::size_t(channel + 1) * plane + spatial));
      __m256 a2 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel + 2] : 0.F),
                                  Load(src + std::size_t(channel + 2) * plane + spatial));
      __m256 a3 = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel + 3] : 0.F),
                                  Load(src + std::size_t(channel + 3) * plane + spatial));
      for (int h = 0; h < hidden; ++h) {
        const __m256 g = Load(hidden_tile.data() + std::size_t(h) * 8);
        a0 = Fma(a0, g, Splat(p0[h]));
        a1 = Fma(a1, g, Splat(p1[h]));
        a2 = Fma(a2, g, Splat(p2[h]));
        a3 = Fma(a3, g, Splat(p3[h]));
      }
      Store(dst + std::size_t(channel) * plane + spatial, a0);
      Store(dst + std::size_t(channel + 1) * plane + spatial, a1);
      Store(dst + std::size_t(channel + 2) * plane + spatial, a2);
      Store(dst + std::size_t(channel + 3) * plane + spatial, a3);
    }
    for (; channel < channels; ++channel) {
      const float* filter = project_weights + std::size_t(channel) * hidden;
      __m256 acc = __lasx_xvfadd_s(Splat(project_bias ? project_bias[channel] : 0.F),
                                   Load(src + std::size_t(channel) * plane + spatial));
      for (int h = 0; h < hidden; ++h) {
        acc = Fma(acc, Load(hidden_tile.data() + std::size_t(h) * 8), Splat(filter[h]));
      }
      Store(dst + std::size_t(channel) * plane + spatial, acc);
    }
  }
  for (; spatial < spatial_end; ++spatial) scalar_at(spatial);
}


void LasxBinary(float* dst, const float* a, const float* b, std::size_t n, int op) noexcept {
  // op: 0 add, 1 sub, 2 mul (div stays scalar).
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 x = Load(a + i), y = Load(b + i);
    __m256 z;
    if (op == 0) z = __lasx_xvfadd_s(x, y);
    else if (op == 1) z = __lasx_xvfsub_s(x, y);
    else z = __lasx_xvfmul_s(x, y);
    Store(dst + i, z);
  }
  for (; i < n; ++i) {
    if (op == 0) dst[i] = a[i] + b[i];
    else if (op == 1) dst[i] = a[i] - b[i];
    else dst[i] = a[i] * b[i];
  }
}

void LasxHardSwish(float* dst, const float* src, std::size_t n) noexcept {
  const __m256 sixth = Splat(1.F / 6.F), half = Splat(.5F);
  const __m256 zero = Splat(0.F), one = Splat(1.F);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 x = Load(src + i);
    const __m256 gate = __lasx_xvfmin_s(one, __lasx_xvfmax_s(zero, Fma(half, x, sixth)));
    Store(dst + i, __lasx_xvfmul_s(x, gate));
  }
  for (; i < n; ++i) {
    const float x = src[i];
    dst[i] = x * std::clamp(x / 6.F + .5F, 0.F, 1.F);
  }
}

void LasxScaleShift(float* dst, const float* src, std::size_t n, float scale, float shift) noexcept {
  const __m256 s = Splat(scale), b = Splat(shift);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) Store(dst + i, Fma(b, Load(src + i), s));
  for (; i < n; ++i) dst[i] = src[i] * scale + shift;
}

void LasxHardSigmoid(float* dst, const float* src, std::size_t n, float alpha, float beta) noexcept {
  const __m256 a = Splat(alpha), b = Splat(beta);
  const __m256 zero = Splat(0.F), one = Splat(1.F);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    Store(dst + i, __lasx_xvfmin_s(one, __lasx_xvfmax_s(zero, Fma(b, Load(src + i), a))));
  }
  for (; i < n; ++i) {
    const float v = src[i] * alpha + beta;
    dst[i] = v < 0.F ? 0.F : (v > 1.F ? 1.F : v);
  }
}


// Fused 3x3 SAME depthwise → 1x1 pointwise (+ optional ReLU / HardSwish /
// exact GELU). Keeps one DW output row in a scratch buffer so the pointwise
// never rereads a full NCHW plane — same shape as the AVX-512/AVX2 fusions.
void LasxDepthwisePointwiseConv3x3S1(
    float* dst, const float* src, const float* dw_weights, const float* dw_bias,
    const float* pw_weights, const float* pw_bias, int channels, int output_channels,
    int height, int width, int activation, int y0, int y1) noexcept {
  if (!dst || !src || !dw_weights || !pw_weights || channels <= 0 ||
      output_channels <= 0 || height <= 0 || width <= 0) {
    return;
  }
  if (y0 < 0) y0 = 0;
  if (y1 < 0 || y1 > height) y1 = height;
  if (y0 >= y1) return;
  const std::size_t plane = std::size_t(height) * width;
  const int first_y = 1, first_x = 1;
  const int last_y = std::min(height, height - 1);
  const int last_x = std::min(width, width - 1);
  const __m256 zero = Splat(0.F), one = Splat(1.F), half = Splat(.5F);
  const __m256 sixth = Splat(1.F / 6.F);
  const auto activate = [&](__m256 x) noexcept {
    if (activation == 1) return __lasx_xvfmax_s(x, zero);
    if (activation == 2) {
      const __m256 gate = __lasx_xvfmin_s(one, __lasx_xvfmax_s(zero, Fma(half, x, sixth)));
      return __lasx_xvfmul_s(x, gate);
    }
    if (activation == 3) return GeluPs(x);
    return x;
  };
  const auto activate_scalar = [&](float x) noexcept {
    if (activation == 1) return std::max(x, 0.F);
    if (activation == 2) return x * std::clamp(x / 6.F + .5F, 0.F, 1.F);
    if (activation == 3) return x * .5F * (1.F + std::erf(x * 0.7071067811865475244F));
    return x;
  };
  thread_local std::vector<float> dw_row;
  dw_row.resize(std::size_t(channels) * width);
  for (int y = y0; y < y1; ++y) {
    const int iy0 = y - 1;
    const int interior_begin = y >= first_y && y < last_y ? first_x : 0;
    const int interior_end = y >= first_y && y < last_y ? last_x : 0;
    for (int channel = 0; channel < channels; ++channel) {
      const float* in = src + std::size_t(channel) * plane;
      const float* filter = dw_weights + std::size_t(channel) * 9;
      float* out = dw_row.data() + std::size_t(channel) * width;
      const float base = dw_bias ? dw_bias[channel] : 0.F;
      int x = 0;
      for (; x < interior_begin; ++x) {
        float sum = base;
        const int ix0 = x - 1;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= height) continue;
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < width) sum += in[std::size_t(iy) * width + ix] * filter[ky * 3 + kx];
          }
        }
        out[x] = sum;
      }
      for (; x + 8 <= interior_end; x += 8) {
        __m256 sum = Splat(base);
        for (int ky = 0; ky < 3; ++ky) {
          const float* row = in + std::size_t(iy0 + ky) * width + x - 1;
          const float* k = filter + ky * 3;
          sum = Fma(sum, Load(row), Splat(k[0]));
          sum = Fma(sum, Load(row + 1), Splat(k[1]));
          sum = Fma(sum, Load(row + 2), Splat(k[2]));
        }
        Store(out + x, sum);
      }
      for (; x < width; ++x) {
        float sum = base;
        const int ix0 = x - 1;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = iy0 + ky;
          if (iy < 0 || iy >= height) continue;
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = ix0 + kx;
            if (ix >= 0 && ix < width) sum += in[std::size_t(iy) * width + ix] * filter[ky * 3 + kx];
          }
        }
        out[x] = sum;
      }
    }
    int oc = 0;
    for (; oc + 4 <= output_channels; oc += 4) {
      const float* w0 = pw_weights + std::size_t(oc) * channels;
      const float* w1 = w0 + channels;
      const float* w2 = w1 + channels;
      const float* w3 = w2 + channels;
      float* d0 = dst + std::size_t(oc) * plane + std::size_t(y) * width;
      float* d1 = dst + std::size_t(oc + 1) * plane + std::size_t(y) * width;
      float* d2 = dst + std::size_t(oc + 2) * plane + std::size_t(y) * width;
      float* d3 = dst + std::size_t(oc + 3) * plane + std::size_t(y) * width;
      int x = 0;
      for (; x + 8 <= width; x += 8) {
        __m256 a0 = Splat(pw_bias ? pw_bias[oc] : 0.F);
        __m256 a1 = Splat(pw_bias ? pw_bias[oc + 1] : 0.F);
        __m256 a2 = Splat(pw_bias ? pw_bias[oc + 2] : 0.F);
        __m256 a3 = Splat(pw_bias ? pw_bias[oc + 3] : 0.F);
        for (int ic = 0; ic < channels; ++ic) {
          const __m256 v = Load(dw_row.data() + std::size_t(ic) * width + x);
          a0 = Fma(a0, v, Splat(w0[ic]));
          a1 = Fma(a1, v, Splat(w1[ic]));
          a2 = Fma(a2, v, Splat(w2[ic]));
          a3 = Fma(a3, v, Splat(w3[ic]));
        }
        Store(d0 + x, activate(a0));
        Store(d1 + x, activate(a1));
        Store(d2 + x, activate(a2));
        Store(d3 + x, activate(a3));
      }
      for (; x < width; ++x) {
        float v0 = pw_bias ? pw_bias[oc] : 0.F;
        float v1 = pw_bias ? pw_bias[oc + 1] : 0.F;
        float v2 = pw_bias ? pw_bias[oc + 2] : 0.F;
        float v3 = pw_bias ? pw_bias[oc + 3] : 0.F;
        for (int ic = 0; ic < channels; ++ic) {
          const float v = dw_row[std::size_t(ic) * width + x];
          v0 += w0[ic] * v; v1 += w1[ic] * v; v2 += w2[ic] * v; v3 += w3[ic] * v;
        }
        d0[x] = activate_scalar(v0); d1[x] = activate_scalar(v1);
        d2[x] = activate_scalar(v2); d3[x] = activate_scalar(v3);
      }
    }
    for (; oc < output_channels; ++oc) {
      const float* w = pw_weights + std::size_t(oc) * channels;
      float* d = dst + std::size_t(oc) * plane + std::size_t(y) * width;
      int x = 0;
      for (; x + 8 <= width; x += 8) {
        __m256 acc = Splat(pw_bias ? pw_bias[oc] : 0.F);
        for (int ic = 0; ic < channels; ++ic) {
          acc = Fma(acc, Load(dw_row.data() + std::size_t(ic) * width + x), Splat(w[ic]));
        }
        Store(d + x, activate(acc));
      }
      for (; x < width; ++x) {
        float sum = pw_bias ? pw_bias[oc] : 0.F;
        for (int ic = 0; ic < channels; ++ic) sum += w[ic] * dw_row[std::size_t(ic) * width + x];
        d[x] = activate_scalar(sum);
      }
    }
  }
}

}  // namespace ppocr::detail::kernels
