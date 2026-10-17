// Native Indra script.onnx (Chinese-line router): AvgPool2 -> 4x [Conv3x3+ReLU, MaxPool2] -> masked mean -> Gemm -> Sigmoid.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "kern.hpp"
#include "pool.hpp"
namespace rpocr::indra {
class ScriptNet {
 public:
  ScriptNet(const std::string& path, int threads);
  void Run(const float* x, int B, int H, int W, const int64_t* widths, float* p) const;
  const char* isa() const { return k_->name; }
 private:
  struct L { int cin, cout, cs; std::vector<float> w[3], b, w1; std::vector<float> u[16]; };  // u: Winograd F(2x2,3x3) packed  // w1: cin==1 taps [9][cs]  // per kernel row ky: packed [3*cin] -> cout
  float Item(const float* x, int H, int W, int64_t width) const;
  const Kern* k_;
  std::unique_ptr<Pool> pool_;
  std::vector<L> layers_;
  std::vector<float> fc_; float fcb_ = 0.f;
};
}  // namespace rpocr::indra
