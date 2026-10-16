// AVX2+FMA build of the warpAffine kernel (same bits as the baseline build: explicit FMAs, no contraction).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#define IMG_NS img_avx2
#include "imgproc_warp.inc"
