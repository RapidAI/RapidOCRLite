// x86-simd-sort argsort for double, compiled for NumPy's avx512 dispatch
// target exactly as numpy/_core/src/npysort/x86_simd_argsort.dispatch.cpp
// does (x86simdsortStatic::argsort(arr, arg, size, hasnan=true)).
#include "np_argsort.hpp"
#include "x86simdsort-static-incl.h"

namespace rpocr::detail {
void NumpyArgsortAvx512(const double* values, std::size_t n, std::size_t* order) {
  x86simdsortStatic::argsort(values, order, n, true);
}
}  // namespace rpocr::detail
