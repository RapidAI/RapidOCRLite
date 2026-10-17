// Metal GPU path for the whole Indra recognizer forward (stem, 10 SE-MBConv1D blocks, CTC head) over a
// batch of lines packed as ragged rows. Activations stay in GPU (unified) memory across all layers; one
// command buffer per batch. Apple only; compiled from metal_rec.mm.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>
namespace rpocr::indra {
struct MetalRecBlock {
  std::vector<float> dwT, dwb;      // [9][C], [C]
  std::vector<float> ex, exb;       // [E][C], [E]
  std::vector<float> f1, f1b;       // [S][E], [S]
  std::vector<float> f2, f2b;       // [E][S], [E]
  std::vector<float> pr, prb;       // [C][E], [C]
};
class MetalRec {
 public:
  MetalRec(const std::vector<float>& stem, const std::vector<float>& stemb, int K0, int C, int E, int S,
           const std::vector<MetalRecBlock>& blocks, const float* headW, const float* headb, int V);
  ~MetalRec();
  // patches: rows x K0; starts: B+1 row offsets per line. Writes ids/conf per row.
  void Run(const float* patches, int rows, const std::vector<int>& starts, int64_t* ids, float* conf);
 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};
}  // namespace rpocr::indra
