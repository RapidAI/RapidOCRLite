// Indra "lite" OCR pipeline in C++ (ported from indra-ocr 0.3.2, Apache-2.0; license text in
// src/indra/LICENSE-indra-ocr): page deskew, line detection, script routing,
// CTC recognition with both vocabularies, line splitting, reading order. Selected by
// rpocr_create / rpocr::OCR when det_model is a folder containing meta.json.
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rpocr::indra {

struct LiteLine {
  float x0{}, y0{}, x1{}, y1{};  // axis-aligned box in input pixels (0.1 px resolution like indra-ocr)
  std::string text;              // UTF-8
  float conf{};
};

class LiteEngine {
 public:
  // dir: folder with det.onnx, rec.onnx, meta.json (+ script.onnx, rec_cjk.onnx, order.onnx).
  LiteEngine(const std::string& dir, int threads);
  ~LiteEngine();
  // tasks: 1 = det, 2 = rec, 3 = both (RPOCR_TASK_*). rgb: packed RGB, stride = 3 * w.
  std::vector<LiteLine> Run(const std::uint8_t* rgb, int w, int h, int tasks) const;
  // Several pages: deskew/detection of page i+1 overlaps recognition/order of page i (PPOCR_LITE_PIPELINE=0 disables).
  // Output per page is identical to Run(..., 3).
  struct BatchImage { const std::uint8_t* rgb; int w, h; };
  std::vector<std::vector<LiteLine>> RunBatch(const std::vector<BatchImage>& images) const;
  std::string Note() const;
  static bool IsLiteDir(const std::string& path);

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

}  // namespace rpocr::indra
