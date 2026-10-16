// Optional INT8 GEMM for the lite recognizer (PPOCR_INDRA_INT8): weights quantized per output row
// (symmetric, scale = max|w| / 127), activations per input row at run time. Not bit-exact with the
// float path; accuracy is measured as CER vs ground truth (native_bench/cer_lite.py).
#pragma once
#include <cstdint>
#include <vector>

namespace ppocr::indra {

struct QMat {
  int N = 0, K = 0, Kp = 0;            // Kp: K rounded up to 64
  std::vector<int8_t> q;               // packed [ceil(N/16)][Kp/4][16][4]: 16 outputs x 4 consecutive k
  std::vector<float> scale;            // [N]
  std::vector<int32_t> rowsum;         // [N] sum_k q (for the u8 x s8 offset trick)
  bool empty() const { return N == 0; }
};

QMat QuantizeRows(const float* W, int N, int K);
// C[l*ldc + n] = act(sum_k A[l*lda + k] * W[n][k] + bias[n]) for n < W.N; act: 0 none, 1 hardswish, 2 ReLU
void GemmQ(const float* A, int L, int K, int lda, const QMat& W, const float* bias, float* C, int ldc, int act);
const char* Int8Isa();

}  // namespace ppocr::indra
