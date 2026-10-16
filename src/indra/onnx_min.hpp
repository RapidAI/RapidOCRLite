// Minimal ONNX protobuf reader for the Indra models: initializers (float/int64)
// and node list (op_type, inputs, outputs, int/ints/float attributes, Constant tensors).
#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ppocr::indra {

struct OTensor {
  std::vector<int64_t> dims;
  int dtype = 0;  // 1 float, 7 int64
  std::vector<float> f;
  std::vector<int64_t> i;
};

struct OAttr {
  std::vector<int64_t> ints;
  float f = 0.f;
  int64_t i = 0;
  bool has_t = false;
  OTensor t;
};

struct ONode {
  std::string op;
  std::vector<std::string> in, out;
  std::unordered_map<std::string, OAttr> attr;
};

struct OModel {
  std::vector<ONode> nodes;
  std::unordered_map<std::string, OTensor> init;
};

OModel LoadOnnx(const std::string& path);  // throws std::runtime_error

}  // namespace ppocr::indra
