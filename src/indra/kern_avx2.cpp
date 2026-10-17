#define KV_AVX2 1
#define KV_HAS_FMA 1
#define VW 8
#define KNAME "avx2"
#include "kern_impl.inc"
namespace rpocr::indra { const Kern* KernAvx2() { return &kKern; } }
