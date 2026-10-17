// Bit-exact re-implementations of the OpenCV 5.0 routines rapidocr 3.10 uses
// for its image front end: cv2.resize (INTER_LINEAR, 8-bit), and
// get_rotate_crop_image (cv2.getPerspectiveTransform + cv2.warpPerspective
// with INTER_CUBIC / BORDER_REPLICATE, then np.rot90 for tall crops).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rpocr::detail::cvc {

// cv::resize(src, dst, Size(dw, dh), 0, 0, INTER_LINEAR) for interleaved
// 8-bit images with `cn` channels. Includes OpenCV's identity copy and its
// exact-2x INTER_AREA shortcut.
void ResizeLinear8u(const std::uint8_t* src, int sw, int sh, std::size_t sstep, int cn,
                    std::uint8_t* dst, int dw, int dh, std::size_t dstep);

// rapidocr.utils.process_img.get_rotate_crop_image on a packed 3-channel
// image. `pts` holds the four box corners (x0,y0,...,x3,y3) as float32.
// The (possibly rotated) packed crop is written to `out`.
void RotateCropImage(const std::uint8_t* src, int sw, int sh, const float pts[8],
                     std::vector<std::uint8_t>& out, int& out_w, int& out_h);

// TextRecognizer.resize_norm_img width: ceil(rec_h * w / h) in double,
// capped by the batch width.
int RecResizedWidth(int w, int h, int rec_h, int batch_width);
// int(rec_h * max(img_w/rec_h, max w/h)) for one 3.10 rec batch.
int RecBatchWidth(const int* widths, const int* heights, std::size_t count, int rec_h,
                  int rec_w);

// float32((v / 255 - mean) / std) as numpy evaluates TextRecognizer and
// DetPreProcess normalization for every 8-bit sample value.
std::array<float, 256> RecNormalizeTable(float mean, float std);
std::array<float, 256> DetNormalizeTable(float mean, float std);

// Kernel entry points (internal).
struct CubicRow {
  const std::uint8_t* src;
  int sw, sh;
  std::size_t sstep;
};
void BicubicRowScalar(const CubicRow& s, const float* xs, const float* ys, int len,
                      std::uint8_t* dst);
#if defined(PPOCR_HAS_AVX2_KERNELS)
void BicubicRowAvx2(const CubicRow& s, const float* xs, const float* ys, int len,
                    std::uint8_t* dst);
#endif

}  // namespace rpocr::detail::cvc
