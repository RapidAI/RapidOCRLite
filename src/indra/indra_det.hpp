// Native interpreter for Indra's det.onnx (line detector): NHWC, channel-padded, SIMD kernels.
#pragma once
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "kern.hpp"
#include "onnx_min.hpp"
#include "pool.hpp"
namespace rpocr::indra {
class DetNet {
 public:
  DetNet(const std::string& path, int threads);
  // x: [H][W] float (already normalised as the ORT model expects). out: [C][oh][ow]; returns dims.
  void Run(const float* x, int H, int W, std::vector<float>& out, int& C, int& oh, int& ow) const;
  const char* isa() const { return k_->name; }
 private:
  struct ConvW {
    int cout, cin, k, s, p, group;
    int act;  // fused activation (1 hardswish)
    std::vector<float> w;  // packed (gemm) or [k*k][Cs] (depthwise) or [cout][k*k] (dense stem)
    std::vector<float> wn;  // narrow 1x1 weights packed for kn_
    std::vector<float> w4, b4;  // cout <= 4: [K][4] + bias[4]
    std::vector<float> w2;  // depthwise / 1-channel layouts with channel stride rounded to one vector
    std::vector<float> b;  // padded bias
  };
  const Kern* k_;
  const Kern* kn_;  // kernel for narrow (cout <= nb/2) 1x1 convs: AVX2 when k_ is AVX-512 (half the padded lanes)
  std::unique_ptr<Pool> pool_;
  OModel m_;
  std::unordered_map<int, ConvW> conv_;  // node index -> prepared conv
  std::unordered_map<std::string, int> fusedOut_;  // HardSwish output name -> conv node index
  std::vector<char> skip_;
  std::unordered_map<std::string, int> uses_;
  mutable std::unordered_map<int, std::vector<std::vector<float>>> splitw_;  // Concat->1x1 split weights
  mutable std::mutex mu_;
  mutable std::vector<std::vector<float>> free_;  // recycled activation buffers
};
}  // namespace rpocr::indra
