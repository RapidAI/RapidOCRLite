#ifndef NEON_VW
#define NEON_VW 8
#endif
#ifndef KRT
#define KRT 4
#endif
#define VW NEON_VW
#define KNAME "neon"
#include "kern_impl.inc"
namespace ppocr::indra { const Kern* KernNeon() { return &kKern; } }
