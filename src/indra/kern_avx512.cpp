#define KV_AVX512 1
#define KV_HAS_FMA 1
#define VW 16
#define KNAME "avx512"
#include "kern_impl.inc"
namespace rpocr::indra { const Kern* KernAvx512() { return &kKern; } }
