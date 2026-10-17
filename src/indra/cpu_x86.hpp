#pragma once
// Runtime x86 feature detection shared by the lite engine (GCC, Clang and MSVC).
// cpuid feature bits are combined with the OS-enabled register state from XCR0
// (YMM for AVX/AVX2, ZMM + opmask for AVX-512), so a CPU feature the OS does not
// save on context switch is reported as absent.
namespace rpocr::indra::cpu {
bool Avx2Fma();     // AVX2 + FMA + OS YMM state
bool Avx512F();     // AVX-512F + OS ZMM state
bool Avx512Kern();  // AVX-512 F/DQ/BW/VL (the kern_avx512 TU)
bool Avx512Vnni();  // AVX-512 F/BW/VNNI (int8 GEMM)
}  // namespace rpocr::indra::cpu
