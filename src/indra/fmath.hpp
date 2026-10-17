#pragma once
// Float transcendental functions that give the same bits with every C runtime.
// glibc, MSVC's CRT and Apple's libm differ in the last ulp of expf / log1pf / erff / sinf / cosf
// for some inputs. Evaluating in double and rounding once to float is correctly rounded except
// when the double result lies within ~1e-9 relative of a float rounding boundary, so results agree
// across platforms in practice.
#include <cmath>
namespace rpocr::indra::fm {
inline float Exp(float x) { return (float)std::exp((double)x); }
inline float Log1p(float x) { return (float)std::log1p((double)x); }
inline float Erf(float x) { return (float)std::erf((double)x); }
inline float Sin(float x) { return (float)std::sin((double)x); }
inline float Cos(float x) { return (float)std::cos((double)x); }
inline float Log(float x) { return (float)std::log((double)x); }
}  // namespace rpocr::indra::fm
