#pragma once
#include <cstdint>
namespace ppocr::indra {
// act: 0 none, 1 hardswish, 2 relu
struct Kern {
  const char* name;
  int nb;  // lanes per packed N block (2 vectors)
  // W row-major [N][K] -> packed [ceil(N/nb)][K][nb] (zero padded)
  void (*pack)(const float* W, int N, int K, float* out);
  // C[l][n] = act(sum_k A[l*lda+k] * W[n][k] + bias[n]), n < Npad (multiple of nb), ldc >= Npad
  void (*gemm)(const float* A, int L, int K, int lda, const float* Wp, const float* bias, int Npad,
               float* C, int ldc, int act);
  // depthwise 1D, kernel 9, pad 4, zero outside [0,L): out[l][c]; wT is [9][C]
  void (*dw9)(const float* in, int L, int C, const float* wT, const float* b, float* out);
  // per row of length Npad: argmax (first), max, conf = 1/sum(exp(x-max))
  void (*rowmax)(const float* x, int n, int* arg, float* conf);
  // depthwise 2D on NHWC (channel stride Cs, multiple of nb), kernel k, stride s, pad p.
  // wT is [k*k][Cs]; computes output rows [oy0, oy1) of y (Wo x Cs) with activation.
  void (*dw2d)(const float* x, int H, int W, int Cs, const float* wT, const float* b, int k, int s, int p,
               float* y, int Wo, int oy0, int oy1, int act);
  // dense conv over a 1-channel image (row stride W), weights [k*k][Cs] (k <= 8)
  void (*c1conv)(const float* x, int H, int W, int Cs, const float* wT, const float* b, int k, int s, int p,
                 float* y, int Wo, int oy0, int oy1, int act);
  // (Npad may also be a multiple of nb/2: the trailing half block computes one vector)
  // sum of three GEMMs sharing output: C = act(bias + sum_t A[t] (L x K, lda) * W[t]^T); Wp[t] packed like pack()
  void (*gemm3)(const float* const* A, int L, int K, int lda, const float* const* Wp, const float* bias, int Npad,
                float* C, int ldc, int act);
  void (*softplus_mul)(float* x, size_t n, float mul);  // x = softplus(x) * mul
  // narrow GEMM, N <= 4: W4 is [K][4] (zero padded), bias4[4]; writes C[l*ldc + n] for n < N
  void (*gemm_n4)(const float* A, int L, int K, int lda, const float* W4, const float* bias4, int N,
                  float* C, int ldc, int act);
  // like gemm but accumulators start from C (no bias): C = act(C + A * W^T)
  void (*gemm_acc)(const float* A, int L, int K, int lda, const float* Wp, int Npad, float* C, int ldc, int act);
  void (*softmax_row)(float* x, int n);
  void (*gelu_row)(float* x, int n);
};
const Kern* KernScalar();
const Kern* KernAvx2();
const Kern* KernAvx512();
const Kern* KernNeon();
const Kern* SelectKern();  // runtime dispatch, honours PPOCR_FORCE_ISA
}  // namespace ppocr::indra
