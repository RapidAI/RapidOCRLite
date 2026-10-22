#pragma once
// np.argsort(values) (kind="quicksort", the default) for float64, index for
// index as NumPy 2.x computes it. rapidocr main batches recognition lines in
// this order, and ties decide which lines share a padded batch.
//
// NumPy's aquicksort_double first tries its x86-simd-sort dispatch targets
// (X86_V4 = AVX-512 F/CD/VL/BW/DQ, then X86_V3 = AVX2/FMA/BMI/F16C/LZCNT/MOVBE)
// and only then runs its scalar introsort. The two SIMD argsorts place ties
// differently from the scalar one and from each other, so this picks the same
// variant NumPy would on this CPU: the vendored x86-simd-sort (the commit
// NumPy v2.5.4 ships) compiled for that ISA, or a port of NumPy's scalar
// aquicksort_ (median of 3, insertion sort below 16 elements, heapsort after
// 2*msb(n) levels, NaN last) on other CPUs and architectures (NumPy has no
// SIMD argsort for ARM/NEON/SVE, LoongArch, ...).
#include <cstddef>
#include <vector>

namespace rpocr::detail {

enum class ArgsortIsa { scalar = 0, avx2 = 1, avx512 = 2 };

// The variant NumPy selects on this CPU. PPOCR_ARGSORT_ISA=scalar|avx2|avx512
// forces one (clamped to what the CPU supports), for tests.
ArgsortIsa NumpyArgsortIsa();
// order[i] = index of the i-th smallest value, exactly np.argsort(values).
void NumpyArgsort(const double* values, std::size_t n, std::size_t* order);
void NumpyArgsort(const double* values, std::size_t n, std::size_t* order, ArgsortIsa isa);
// NumPy's scalar aquicksort_<double_tag> (also the result on ARM).
void NumpyArgsortScalar(const double* values, std::size_t n, std::size_t* order);
#if defined(PPOCR_HAS_NP_ARGSORT_AVX2)
void NumpyArgsortAvx2(const double* values, std::size_t n, std::size_t* order);
#endif
#if defined(PPOCR_HAS_NP_ARGSORT_AVX512)
void NumpyArgsortAvx512(const double* values, std::size_t n, std::size_t* order);
#endif

}  // namespace rpocr::detail
