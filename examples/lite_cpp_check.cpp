// C++ API check for the lite (Indra) model: ppocr::OCR(<lite folder>, "", "") must give the same lines as
// the C API (texts and integer-rounded boxes). usage: ppocr_lite_cpp_check <lite_dir> <page.ppm> [threads]
#include <exception>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "ppocr/ppocr.h"
#include "ppocr/ppocr.hpp"

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s lite_dir page.ppm [threads]\n", argv[0]); return 2; }
  ppocr::Options o; o.backend = ppocr::Backend::cpu_only; o.rec_parallelism = argc > 3 ? std::atoi(argv[3]) : 4;
  {  // Options::model_type: explicit lite works, ppocr on a lite folder and lite on a file throw
    ppocr::Options e = o; e.model_type = ppocr::ModelType::lite;
    ppocr::OCR ok(argv[1], "", "", e);
    bool t1 = false, t2 = false;
    e.model_type = ppocr::ModelType::ppocr;
    try { ppocr::OCR bad(argv[1], "", "", e); } catch (const std::exception&) { t1 = true; }
    e.model_type = ppocr::ModelType::lite;
    try { ppocr::OCR bad(std::string(argv[1]) + "/det.onnx", "", "", e); } catch (const std::exception&) { t2 = true; }
    std::printf("model_type checks: ppocr-on-lite throws=%s lite-on-file throws=%s\n", t1 ? "yes" : "NO", t2 ? "yes" : "NO");
    if (!t1 || !t2) return 1;
  }
  ppocr::OCR ocr(argv[1], "", "", o);
  ppocr::Image img = ppocr::LoadPPM(argv[2]);
  auto lines = ocr.Recognize(img);
  ppocr_options co; ppocr_options_init(&co); co.backend = PPOCR_BACKEND_CPU; co.rec_parallelism = o.rec_parallelism;
  ppocr_ocr* h = ppocr_create(argv[1], nullptr, nullptr, &co);
  if (!h) { std::fprintf(stderr, "create: %s\n", ppocr_last_error()); return 1; }
  ppocr_image ci{img.width, img.height, img.rgb.data()}; ppocr_result r{};
  if (ppocr_recognize(h, &ci, &r) != PPOCR_OK) { std::fprintf(stderr, "recognize: %s\n", ppocr_last_error()); return 1; }
  int bad = r.count != lines.size();
  for (size_t i = 0; !bad && i < r.count; i++) {
    const auto& a = lines[i]; const auto& b = r.items[i];
    if (a.text != b.text || a.box[0][0] != (int)std::lround(b.box[0].x) || a.box[2][1] != (int)std::lround(b.box[2].y)) bad = 1;
  }
  std::printf("cpp_lines=%zu c_lines=%zu match=%s note=%s\n", lines.size(), r.count, bad ? "NO" : "yes", ocr.BackendNote().c_str());
  ppocr_result_free(&r); ppocr_destroy(h);
  return bad;
}
