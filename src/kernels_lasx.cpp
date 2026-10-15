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

#include <cstddef>

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

}  // namespace ppocr::detail::kernels
