// Native Indra order.onnx: pre-LN transformer over line features + pairwise heads (a, same, tab) and cls.
#pragma once
#include <memory>
#include <string>
#include <vector>
#include "kern.hpp"
#include "pool.hpp"
namespace rpocr::indra {
class OrderNet {
 public:
  OrderNet(const std::string& path, int threads);
  // feats [N][F], mask [N] (bool bytes). Outputs: a [N][N], same [N][N], cls [N][ncls], tab [3][N][N]
  void Run(const float* feats, const uint8_t* mask, int N, float* a, float* same, float* cls, float* tab) const;
  int ncls() const { return ncls_; }
  int nfeat() const { return F_; }
 private:
  struct Lin { int K = 0, N = 0, Np = 0; std::vector<float> w, b, raw; };
  struct Layer { std::vector<float> g1, b1, g2, b2; Lin qkv, proj, ff1, ff2; };
  void Linear(const Lin& l, const float* x, int rows, int ldx, float* y, int ldy, int act) const;
  void Pair(const float* u, int ldu, const float* w, int ldw, int N, int D, float* out, int sym) const;
  const Kern* k_;
  std::unique_ptr<Pool> pool_;
  int F_ = 16, D_ = 128, H_ = 4, ncls_ = 0, nfreq_ = 0, npos_ = 4;
  std::vector<float> freq_;
  Lin e1_, e2_, q_, k_w_, bu_, bw_, tu_, tw_, cls_;
  std::vector<Layer> layers_;
  std::vector<float> gN_, bN_;
  float lnEps_ = 1e-5f, scale_ = 11.3137083f;
};
}  // namespace rpocr::indra
