// Metal GPU path for the Indra recognizer's CTC head (1x1 conv C->V + argmax + softmax-max).
// Apple only; compiled from metal_head.mm. MSL is compiled at runtime from an embedded string.
#pragma once
#include <cstdint>
#include <memory>
namespace rpocr::indra {
class MetalHead {
 public:
  // W: [V][C] row-major, b: [V]. Throws if no Metal device or C unsupported.
  MetalHead(const float* W, const float* b, int V, int C);
  ~MetalHead();
  // h: rows x C (row stride ldh); writes ids/conf per row
  void Run(const float* h, int rows, int ldh, int64_t* ids, float* conf);
  const char* device() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};
}  // namespace rpocr::indra
