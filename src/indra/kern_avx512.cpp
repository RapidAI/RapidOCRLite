#define VW 16
#define KNAME "avx512"
#include "kern_impl.inc"
namespace ppocr::indra { const Kern* KernAvx512() { return &kKern; } }
