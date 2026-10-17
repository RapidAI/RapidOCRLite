#include <cstdlib>
#include <cstring>
#include <string_view>
#include "kern.hpp"
namespace rpocr::indra {
#if !(defined(__x86_64__) || defined(_M_X64))
const Kern* KernAvx2() { return nullptr; }
const Kern* KernAvx512() { return nullptr; }
#endif
#if !(defined(__aarch64__) || defined(_M_ARM64))
const Kern* KernNeon() { return nullptr; }
#endif
const Kern* SelectKern() {
  const char* f = std::getenv("PPOCR_FORCE_ISA");
  std::string_view force = f ? f : "";
#if defined(__x86_64__) || defined(_M_X64)
  __builtin_cpu_init();
  bool a512 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
              __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl");
  bool a2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
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
