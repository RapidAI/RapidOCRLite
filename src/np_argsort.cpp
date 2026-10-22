#include "np_argsort.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#include <immintrin.h>
#define PPOCR_ARGSORT_X86 1
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#define PPOCR_ARGSORT_X86 1
#endif

namespace rpocr::detail {
namespace {

// npy::cmp<double_tag>: a < b, NaNs sort last.
inline bool Less(double a, double b) { return a < b || (b != b && a == a); }

int GetMsb(std::size_t n) {  // npy_get_msb
  int depth = 0;
  while (n >>= 1) ++depth;
  return depth;
}

using intp = std::ptrdiff_t;

void AHeapsort(const double* v, intp* tosort, intp n) {  // aheapsort_
  intp* a = tosort - 1;
  intp i, j, l, tmp;
  for (l = n >> 1; l > 0; --l) {
    tmp = a[l];
    for (i = l, j = l << 1; j <= n;) {
      if (j < n && Less(v[a[j]], v[a[j + 1]])) j += 1;
      if (Less(v[tmp], v[a[j]])) {
        a[i] = a[j];
        i = j;
        j += j;
      } else {
        break;
      }
    }
    a[i] = tmp;
  }
  for (; n > 1;) {
    tmp = a[n];
    a[n] = a[1];
    n -= 1;
    for (i = 1, j = 2; j <= n;) {
      if (j < n && Less(v[a[j]], v[a[j + 1]])) j++;
      if (Less(v[tmp], v[a[j]])) {
        a[i] = a[j];
        i = j;
        j += j;
      } else {
        break;
      }
    }
    a[i] = tmp;
  }
}

constexpr int kSmallQuicksort = 15;
constexpr int kStack = static_cast<int>(sizeof(intp) * 8 * 2);

void AQuicksort(const double* v, intp* tosort, intp num) {  // aquicksort_
  double vp;
  intp* pl = tosort;
  intp* pr = tosort + num - 1;
  intp* stack[kStack];
  intp** sptr = stack;
  intp *pm, *pi, *pj, *pk, vi;
  int depth[kStack];
  int* psdepth = depth;
  int cdepth = GetMsb(static_cast<std::size_t>(num)) * 2;
  for (;;) {
    if (cdepth < 0) {
      AHeapsort(v, pl, pr - pl + 1);
      goto stack_pop;
    }
    while ((pr - pl) > kSmallQuicksort) {
      pm = pl + ((pr - pl) >> 1);
      if (Less(v[*pm], v[*pl])) std::swap(*pm, *pl);
      if (Less(v[*pr], v[*pm])) std::swap(*pr, *pm);
      if (Less(v[*pm], v[*pl])) std::swap(*pm, *pl);
      vp = v[*pm];
      pi = pl;
      pj = pr - 1;
      std::swap(*pm, *pj);
      for (;;) {
        do { ++pi; } while (Less(v[*pi], vp));
        do { --pj; } while (Less(vp, v[*pj]));
        if (pi >= pj) break;
        std::swap(*pi, *pj);
      }
      pk = pr - 1;
      std::swap(*pi, *pk);
      if (pi - pl < pr - pi) {
        *sptr++ = pi + 1;
        *sptr++ = pr;
        pr = pi - 1;
      } else {
        *sptr++ = pl;
        *sptr++ = pi - 1;
        pl = pi + 1;
      }
      *psdepth++ = --cdepth;
    }
    for (pi = pl + 1; pi <= pr; ++pi) {
      vi = *pi;
      vp = v[vi];
      pj = pi;
      pk = pi - 1;
      while (pj > pl && Less(vp, v[*pk])) *pj-- = *pk--;
      *pj = vi;
    }
  stack_pop:
    if (sptr == stack) break;
    pr = *(--sptr);
    pl = *(--sptr);
    cdepth = *(--psdepth);
  }
}

#if defined(PPOCR_ARGSORT_X86)
void Cpuid(int leaf, int sub, unsigned r[4]) {
#if defined(_MSC_VER)
  int out[4];
  __cpuidex(out, leaf, sub);
  for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(out[i]);
#else
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}
std::uint64_t Xgetbv() {
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  unsigned lo, hi;
  __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
  return (static_cast<std::uint64_t>(hi) << 32) | lo;
#endif
}

ArgsortIsa DetectIsa() {
  unsigned r[4];
  Cpuid(0, 0, r);
  const unsigned max_leaf = r[0];
  if (max_leaf < 7) return ArgsortIsa::scalar;
  Cpuid(1, 0, r);
  const unsigned ecx1 = r[2];
  const bool osxsave = (ecx1 >> 27) & 1;
  if (!osxsave) return ArgsortIsa::scalar;
  const std::uint64_t xcr0 = Xgetbv();
  const bool ymm_os = (xcr0 & 0x6) == 0x6;
  const bool zmm_os = (xcr0 & 0xE6) == 0xE6;
  const bool fma = (ecx1 >> 12) & 1, movbe = (ecx1 >> 22) & 1, avx = (ecx1 >> 28) & 1,
             f16c = (ecx1 >> 29) & 1, popcnt = (ecx1 >> 23) & 1, sse41 = (ecx1 >> 19) & 1,
             sse42 = (ecx1 >> 20) & 1;
  Cpuid(7, 0, r);
  const unsigned ebx7 = r[1];
  const bool bmi1 = (ebx7 >> 3) & 1, avx2 = (ebx7 >> 5) & 1, bmi2 = (ebx7 >> 8) & 1,
             avx512f = (ebx7 >> 16) & 1, avx512dq = (ebx7 >> 17) & 1,
             avx512cd = (ebx7 >> 28) & 1, avx512bw = (ebx7 >> 30) & 1,
             avx512vl = (ebx7 >> 31) & 1;
  Cpuid(0x80000000u, 0, r);
  bool lzcnt = false;
  if (r[0] >= 0x80000001u) {
    Cpuid(0x80000001u, 0, r);
    lzcnt = (r[2] >> 5) & 1;
  }
  // NumPy's X86_V3 / X86_V4 groups (meson_cpu/x86).
  const bool v3 = ymm_os && avx && avx2 && fma && f16c && bmi1 && bmi2 && lzcnt && movbe &&
                  popcnt && sse41 && sse42;
  const bool v4 = v3 && zmm_os && avx512f && avx512cd && avx512vl && avx512bw && avx512dq;
  if (v4) return ArgsortIsa::avx512;
  if (v3) return ArgsortIsa::avx2;
  return ArgsortIsa::scalar;
}
#else
ArgsortIsa DetectIsa() { return ArgsortIsa::scalar; }
#endif

ArgsortIsa Compiled(ArgsortIsa isa) {
#if !defined(PPOCR_HAS_NP_ARGSORT_AVX512)
  if (isa == ArgsortIsa::avx512) isa = ArgsortIsa::avx2;
#endif
#if !defined(PPOCR_HAS_NP_ARGSORT_AVX2)
  if (isa == ArgsortIsa::avx2) isa = ArgsortIsa::scalar;
#endif
  return isa;
}

}  // namespace

ArgsortIsa NumpyArgsortIsa() {
  static const ArgsortIsa hw = DetectIsa();
  ArgsortIsa isa = hw;
  if (const char* forced = std::getenv("PPOCR_ARGSORT_ISA")) {
    ArgsortIsa want = isa;
    if (std::strcmp(forced, "scalar") == 0) want = ArgsortIsa::scalar;
    else if (std::strcmp(forced, "avx2") == 0) want = ArgsortIsa::avx2;
    else if (std::strcmp(forced, "avx512") == 0) want = ArgsortIsa::avx512;
    if (static_cast<int>(want) < static_cast<int>(isa)) isa = want;
  }
  return Compiled(isa);
}

void NumpyArgsortScalar(const double* values, std::size_t n, std::size_t* order) {
  std::vector<intp> idx(n);
  for (std::size_t i = 0; i < n; ++i) idx[i] = static_cast<intp>(i);
  if (n > 0) AQuicksort(values, idx.data(), static_cast<intp>(n));
  for (std::size_t i = 0; i < n; ++i) order[i] = static_cast<std::size_t>(idx[i]);
}

void NumpyArgsort(const double* values, std::size_t n, std::size_t* order, ArgsortIsa isa) {
  for (std::size_t i = 0; i < n; ++i) order[i] = i;  // np.argsort starts from arange(n)
  if (n == 0) return;
  switch (Compiled(isa)) {
#if defined(PPOCR_HAS_NP_ARGSORT_AVX512)
    case ArgsortIsa::avx512: NumpyArgsortAvx512(values, n, order); return;
#endif
#if defined(PPOCR_HAS_NP_ARGSORT_AVX2)
    case ArgsortIsa::avx2: NumpyArgsortAvx2(values, n, order); return;
#endif
    default: NumpyArgsortScalar(values, n, order); return;
  }
}

void NumpyArgsort(const double* values, std::size_t n, std::size_t* order) {
  NumpyArgsort(values, n, order, NumpyArgsortIsa());
}

}  // namespace rpocr::detail
