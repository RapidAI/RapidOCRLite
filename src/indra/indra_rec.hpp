#pragma once
#include <memory>
#include <string>
#include <vector>
#include "int8_gemm.hpp"
#include "kern.hpp"
namespace rpocr::indra {
// Native CTC recognizer for Indra rec.onnx / rec_cjk.onnx (patch-embed stem + 1D
// depthwise/SE blocks + 1x1 head with in-graph argmax/softmax-max).
class RecNet {
 public:
  RecNet(const std::string& onnx_path, int threads);
  // x: [B,1,H,W] float32 ink map, widths[B] -> ids/conf [B,T], lengths[B]; T = W / stride
  void Run(const float* x, int B, int H, int W, const int64_t* widths, int64_t* ids, float* conf,
           int64_t* lengths) const;
  int steps(int W) const { return W / sw_; }
  // Whole forward on the GPU (Metal, PPOCR_INDRA_METAL=2): lines are independent there, so all lines of a page can go
  // in one submission. Line b: x[b] is [H, w[b]] row-major, L[b] time steps (zero past w[b]). ids/conf: sum(L) entries.
  int stride() const { return sw_; }
  bool ragged_gpu() const { return mrec_ != nullptr; }
  void RunRagged(const std::vector<const float*>& x, const std::vector<int>& w, const std::vector<int>& L, int H,
                 int64_t* ids, float* conf) const;
  const char* isa() const { return (metal_ || mrec_) ? metal_isa_.c_str() : isa_.c_str(); }
 private:
  struct Block { std::vector<float> dwT, dwb, ex, exb, f1, f1b, f2, f2b, pr, prb; QMat qex, qpr; };
  void Line(const float* x, int H, int W, int L, int64_t* ids, float* conf, float* hout = nullptr) const;
  std::shared_ptr<class MetalRec> mrec_;  // Apple: whole forward on the GPU (PPOCR_INDRA_METAL=2)
  std::shared_ptr<class MetalHead> metal_;  // Apple: CTC head on the GPU (PPOCR_INDRA_METAL=1)
  const Kern* k_;
  int threads_;
  int C_ = 0, E_ = 0, S_ = 0, V_ = 0, kh_ = 0, sw_ = 0;
  int Cp_ = 0, Ep_ = 0, Sp_ = 0, Vp_ = 0;
  std::vector<float> stem_, stemb_, head_, headb_;
  std::vector<Block> blocks_;
  std::string metal_isa_;
  std::vector<float> headRaw_;  // Apple Accelerate head path only
  QMat qhead_;                  // PPOCR_INDRA_INT8=head|all (CPU only; not bit-exact)
  std::string isa_;
};
}  // namespace rpocr::indra
