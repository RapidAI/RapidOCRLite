#include <cstdlib>
#include <cstring>
#include <string_view>
#include "cpu_x86.hpp"
#include "kern.hpp"
namespace rpocr::indra {
// The build defines PPOCR_INDRA_AVX2 / PPOCR_INDRA_AVX512 when it compiles those kernel TUs.
#ifndef PPOCR_INDRA_AVX2
const Kern* KernAvx2() { return nullptr; }
#endif
#ifndef PPOCR_INDRA_AVX512
const Kern* KernAvx512() { return nullptr; }
#endif
#if !(defined(__aarch64__) || defined(_M_ARM64))
const Kern* KernNeon() { return nullptr; }
#endif
const Kern* SelectKern() {
  const char* f = std::getenv("PPOCR_FORCE_ISA");
  std::string_view force = f ? f : "";
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
  bool a512 = KernAvx512() != nullptr && cpu::Avx512Kern();
  bool a2 = KernAvx2() != nullptr && cpu::Avx2Fma();
  if (force == "scalar") return KernScalar();
  if (force == "avx2") return a2 ? KernAvx2() : KernScalar();
  if (a512 && std::getenv("PPOCR_DISABLE_AVX512") == nullptr && (force.empty() || force == "avx512")) return KernAvx512();
  if (a2) return KernAvx2();
  return KernScalar();
#elif defined(__aarch64__) || defined(_M_ARM64)
  if (force == "scalar") return KernScalar();
  return KernNeon();
#else
  return KernScalar();
#endif
}
}  // namespace rpocr::indra
