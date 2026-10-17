#include "rapidocrlite/rapidocrlite.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
  const auto cpu = rpocr::QueryCpuInfo();
  if (argc == 2 && std::string(argv[1]) == "--cpu-info") {
    std::cout << "active_isa=" << cpu.active_isa
              << " avx2=" << cpu.avx2
              << " avx512=" << cpu.avx512
              << " neon=" << cpu.neon
              << " threads=" << cpu.threads << '\n';
    return 0;
  }
  if (argc != 5) {
    std::cerr << "usage: ppocr_cpp_api_demo DET.onnx REC.onnx dict.txt image.ppm\n";
    return 2;
  }
  try {
    rpocr::Options options;
    options.backend = rpocr::Backend::cpu_only;
    rpocr::OCR ocr(argv[1], argv[2], argv[3], options);
    const rpocr::Image image = rpocr::LoadPPM(argv[4]);
    const auto results = ocr.Recognize(image);
    std::cout << "isa=" << cpu.active_isa << " boxes=" << results.size() << '\n';
    for (const auto& result : results) {
      std::cout << result.text << '\t' << result.confidence << '\t'
                << result.bbox[0] << ',' << result.bbox[1] << ','
                << result.bbox[2] << ',' << result.bbox[3] << '\n';
    }
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << '\n';
    return 1;
  }
  return 0;
}
