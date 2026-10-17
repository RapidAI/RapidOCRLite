// Stub LiteEngine for compilers that cannot build the native lite pipeline (MSVC cl.exe: the
// lite sources use GCC/Clang vector extensions and builtins). Lite model folders are recognised
// and rejected with a clear error; PP-OCR models work as usual. Build with clang-cl for lite.
#include <filesystem>
#include <stdexcept>
#include "lite_pipeline.hpp"

namespace rpocr::indra {

struct LiteEngine::Impl {};

LiteEngine::LiteEngine(const std::string&, int) {
  throw std::runtime_error(
      "the lite model is not available in this build (MSVC); build RapidOCRLite with clang-cl, GCC or Clang");
}
LiteEngine::~LiteEngine() = default;
std::vector<LiteLine> LiteEngine::Run(const std::uint8_t*, int, int, int) const { return {}; }
std::vector<std::vector<LiteLine>> LiteEngine::RunBatch(const std::vector<BatchImage>&) const { return {}; }
std::string LiteEngine::Note() const { return {}; }
bool LiteEngine::IsLiteDir(const std::string& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(std::filesystem::path(path) / "meta.json", ec);
}

}  // namespace rpocr::indra
