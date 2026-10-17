#define VW 8
#define KNAME "avx2"
#include "kern_impl.inc"
namespace rpocr::indra { const Kern* KernAvx2() { return &kKern; } }
