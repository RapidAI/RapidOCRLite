#include "cpu_x86.hpp"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#include <cstdint>

namespace rpocr::indra::cpu {
namespace {
void Cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
#if defined(_MSC_VER) && !defined(__clang__)
  int t[4];
  __cpuidex(t, (int)leaf, (int)sub);
  for (int i = 0; i < 4; ++i) r[i] = (unsigned)t[i];
#else
  r[0] = r[1] = r[2] = r[3] = 0;
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}
uint64_t Xcr0() {
#if defined(_MSC_VER) && !defined(__clang__)
  return _xgetbv(0);
#else
  uint32_t lo, hi;
  __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(lo), "=d"(hi) : "c"(0));  // xgetbv
  return ((uint64_t)hi << 32) | lo;
#endif
}
struct Features {
  bool avx2fma = false, a512f = false, a512kern = false, a512vnni = false;
  Features() {
    unsigned r[4];
    Cpuid(0, 0, r);
    const unsigned max_leaf = r[0];
    if (max_leaf < 7) return;
    Cpuid(1, 0, r);
    const bool fma = (r[2] >> 12) & 1, osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1;
    if (!osxsave || !avx) return;
    const uint64_t xcr0 = Xcr0();
    const bool ymm = (xcr0 & 0x6) == 0x6;     // SSE + AVX state
    const bool zmm = (xcr0 & 0xe6) == 0xe6;   // + opmask, ZMM_Hi256, Hi16_ZMM
    Cpuid(7, 0, r);
    const unsigned b = r[1], c = r[2];
    const bool avx2 = (b >> 5) & 1, f = (b >> 16) & 1, dq = (b >> 17) & 1, bw = (b >> 30) & 1, vl = (b >> 31) & 1;
    const bool vnni = (c >> 11) & 1;
    avx2fma = ymm && avx2 && fma;
    a512f = avx2fma && zmm && f;
    a512kern = a512f && dq && bw && vl;
    a512vnni = a512f && bw && vnni;
  }
};
const Features& Get() { static const Features f; return f; }
}  // namespace
bool Avx2Fma() { return Get().avx2fma; }
bool Avx512F() { return Get().a512f; }
bool Avx512Kern() { return Get().a512kern; }
bool Avx512Vnni() { return Get().a512vnni; }
}  // namespace rpocr::indra::cpu
#else
namespace rpocr::indra::cpu {
bool Avx2Fma() { return false; }
bool Avx512F() { return false; }
bool Avx512Kern() { return false; }
bool Avx512Vnni() { return false; }
}  // namespace rpocr::indra::cpu
#endif
