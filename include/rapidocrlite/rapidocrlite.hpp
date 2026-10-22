#pragma once

#include "rapidocrlite/export.h"

#include <array>
#include <cstddef>
#include <functional>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rpocr {

// Dependency-free 8-bit RGB image.  Pixels are packed RGB, row-major.
struct Image {
  int width{};
  int height{};
  std::vector<std::uint8_t> rgb;

  [[nodiscard]] bool empty() const noexcept {
    return width <= 0 || height <= 0 ||
           rgb.size() != static_cast<std::size_t>(width) * height * 3;
  }
};

struct Point { float x{}, y{}; };

// Thrown when a requested mode is not available on the selected backend.
struct UnsupportedError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Result {
  std::string text;
  float confidence{};
  std::array<int, 4> bbox{};                 // x, y, width, height
  std::array<std::array<int, 2>, 4> box{};   // TL, TR, BR, BL
  // Text-line orientation (rapidocr main TextClassifier). -1: the classifier
  // did not run for this line. Otherwise the arg-max class index (0 = "0",
  // 1 = "180" with the default label list) and its softmax probability. The
  // line was rotated 180 degrees before recognition when cls_label == 1 and
  // cls_score > Options::cls_thresh.
  int cls_label = -1;
  float cls_score = 0.F;
  // Per-character CTC data for word / character boxes (rapidocr main
  // return_word_box). Filled by the CPU / hybrid recognition path (and the
  // Metal path that feeds the CPU front end's tensors); empty elsewhere
  // (rec_steps == 0). char_cols[k] is the CTC time step that emitted the
  // k-th decoded character and char_probs[k] its max softmax probability.
  // rec_steps is the recognizer output length of the line's batch,
  // crop_width / crop_height the rotated crop size (get_rotate_crop_image)
  // and rec_max_wh_ratio the batch's max_wh_ratio, so rapidocr main's
  // line_txt_len = rec_steps * (crop_width / crop_height) / rec_max_wh_ratio.
  std::vector<int> char_cols;
  std::vector<float> char_probs;
  int rec_steps = 0;
  int crop_width = 0;
  int crop_height = 0;
  double rec_max_wh_ratio = 0.0;
};

// Execution policy for PP-OCRv6. CPU is always available. GPU-only never
// silently falls back: preprocessing, neural graphs, DB detection postprocess
// and CTC reduction execute on Vulkan; the host receives only final OCR box
// records/UTF-8 assembly metadata. Hybrid may use only Vulkan segments whose
// end-to-end (including transfer) benchmark is no slower than CPU, otherwise
// it retains the CPU path.
enum class Backend {
  cpu_only,
  gpu_only,
  hybrid,
  // Native Metal GPU graph on Apple Silicon: the same GPU-only graph,
  // executed by Metal. Without Metal (not built, or no device) the engine
  // falls back to CPU instead of failing. Metal is chosen for the whole
  // process on the first GPU initialization.
  metal,
};

struct BackendInfo {
  bool vulkan_loader_available{};
  bool vulkan_compute_available{};
  bool full_graph_gpu_available{};
  std::string device_name;
  // Changes whenever Vulkan rebuilds its logical device/arena after a failed
  // submission. Hybrid admission decisions are qualified for this exact
  // runtime generation and are invalidated when it changes.
  std::uint64_t vulkan_runtime_generation{};
};

[[nodiscard]] RPOCR_API BackendInfo QueryBackendInfo();
// True when the library was built with Metal and a Metal device exists.
// Does not initialize the GPU runtime.
[[nodiscard]] RPOCR_API bool MetalAvailable();

// Which SIMD path the process will actually enter. active_isa is one of
// "avx512", "avx2", "neon", or "scalar". AVX-512 is reported only when the
// CPU and OS have enabled it; PPOCR_FORCE_ISA=avx2 (or scalar) keeps an
// AVX-512 capable host on the narrower path. An AVX2-only CPU never reports
// avx512, and the AVX-512 object file is not entered.
struct CpuInfo {
  bool avx2_compiled{};
  bool avx512_compiled{};
  bool neon_compiled{};
  bool avx2{};
  bool avx512{};
  bool neon{};
  std::string active_isa;
  int threads{};
};

[[nodiscard]] RPOCR_API CpuInfo QueryCpuInfo();

// Which model family OCR() loads. auto_detect: det_model naming a folder with
// meta.json selects the lite (Indra) model, anything else PP-OCR.
enum class ModelType { auto_detect = 0, ppocr = 1, lite = 2 };

struct Options {
  Backend backend = Backend::hybrid;
  float det_threshold = 0.20F;
  float det_box_threshold = 0.45F;
  float det_unclip_ratio = 1.40F;
  int det_limit_side_len = 960;
  int rec_height = 48;
  int rec_max_width = 3200;
  // Recognition crops with exactly the same resized width are evaluated in
  // bounded dynamic batches. Keeping this finite controls activation peak
  // memory on dense pages while removing per-crop executor overhead.
  // Tiny/small CNN+CTC recognizers are width-padding-safe: right-zero padding
  // equals convolution's implicit boundary, so N=4 dense-page batching is the
  // measured default. Transformer exports may still change CTC results for
  // N>1; deployments can set 1 to restore strict per-crop attention.
  int rec_batch_size = 4;
  // false: recognition input reproduces rapidocr 3.10 exactly (width-sorted
  // batches padded to the widest line, minimum width rec_height*320/48).
  // true: fast mode; each line runs at about its own width (similar widths
  // share a batch, no minimum width). Env PPOCR_REC_FAST=0/1 overrides.
  bool rec_fast_mode = false;
  // Crops whose natural widths land in the same small bucket are packed into
  // one NCHW batch. Each crop keeps its own aspect-ratio resize and is padded
  // on the right, so bucketing avoids resampling text while increasing useful
  // batch density on heterogeneous pages. Set to 1 to retain exact-width
  // grouping (the lower-memory A/B policy) for measurements. The bounded
  // default of 256 packs nearby crop widths without multiplying activation
  // capacity by a large crop count.
  // Wider buckets are available for deployment-specific GPU/batch tuning.
  int rec_batch_width_bucket = 256;
  // Independent recognition batches are safe to run concurrently. This is
  // deliberately capped by default so dense pages do not multiply activation
  // peak memory by all logical CPU cores.
  // Dense-page sweeps on the target 16-logical-core host selected 12 outer
  // batches: it improves tiny/small throughput over eight while avoiding the
  // medium model's measured 16-way bandwidth/cache regression. The recognizer
  // itself still uses channel/row-parallel SIMD kernels.
  int rec_parallelism = 12;
  // A hybrid OCR call uses one persistent Vulkan queue/context. Concurrent
  // graph batches would serialize at that queue while retaining multiple CPU
  // activation workspaces, so hybrid defaults to one graph submitter. CPU
  // execution keeps `rec_parallelism` independent batches in flight. Set a
  // positive value here to override the hybrid cap after measuring a driver.
  int hybrid_graph_parallelism = 1;
  // Same-shape detector pages can share a real NCHW batch as well. Pages are
  // grouped only after their normal PP-OCR resize dimensions have been
  // calculated, so this never pads unlike-sized documents or changes the
  // detector's resize semantics. Keep the batch bounded because detector
  // feature maps are substantially larger than recognition crops.
  int det_batch_size = 4;
  // Resize/normalization is independent for pages that already share a
  // detector batch shape.  Build those NCHW planes concurrently before the
  // single batched graph submission, so a GPU batch does not leave CPU cores
  // idle preparing its input.  This is a bound rather than a per-page thread
  // count; use 1 for strict serial preprocessing or tune it to the host.
  int batch_preprocess_parallelism = 4;
  // Independent page images may also be submitted as one bounded batch. A
  // value of one keeps detector work serial; higher values bound detector
  // concurrency.  Regardless of this value, RecognizeBatch coalesces
  // same-width text crops across pages into real recognizer NCHW batches, so
  // CPU SIMD and Vulkan's batch dimension both see useful work without
  // padding detector pages to a common size.
  int image_batch_parallelism = 1;
  // 0 shrinks on the long side (historical C++ default). 1 grows on the
  // short side, which is rapidocr 3.x limit_type "min".
  int det_limit_type = 0;
  // B, G, R plane order. Defaults reproduce the previous ImageNet detector
  // coefficients. rapidocr 3.x passes 0.5/0.5 through the C ABI.
  std::array<float, 3> det_mean{0.485F, 0.456F, 0.406F};
  std::array<float, 3> det_std{0.229F, 0.224F, 0.225F};
  std::array<float, 3> rec_mean{0.5F, 0.5F, 0.5F};
  std::array<float, 3> rec_std{0.5F, 0.5F, 0.5F};
  // rapidocr 3.x config enables a 2x2 dilation. The C++ default stays off so
  // existing callers keep the previous component mask.
  int det_use_dilation = 0;
  int det_max_candidates = 1000;
  // -1 keeps the automatic adapter choice (CPU/lavapipe devices are skipped).
  // A non-negative index selects that physical device, including a software
  // Vulkan device. The choice is process-wide and is read on the first
  // Vulkan initialization.
  int vulkan_device_index = -1;
  // Appended: model family (see ModelType). lite: det_model is the lite folder,
  // rec_model / dictionary are ignored; rec_parallelism sets its thread count.
  ModelType model_type = ModelType::auto_detect;
  // Text-line orientation classification (0/180), rapidocr main Global.use_cls.
  // On by default; it runs only when cls_model names a classifier ONNX file
  // (ch_ppocr_mobile_v2.0_cls_mobile or a PP-LCNet textline_ori model).
  // The PPOCR_USE_CLS environment variable (0/1) overrides use_cls and
  // PPOCR_CLS_MODEL supplies cls_model when it is empty. The classifier runs
  // on the CPU crops of the cpu_only and hybrid paths; the gpu_only / Metal
  // graphs and the lite (Indra) model do not classify (cls_label stays -1).
  bool use_cls = true;
  std::string cls_model;
  // rapidocr main Cls defaults (PP-OCRv4 = ch_ppocr_mobile_v2.0_cls_mobile):
  // 48x192 input, batches of 6, rotate when "180" scores above 0.9, and
  // classify the centre of long lines at the model aspect ratio (#748).
  // PP-OCRv5 PP-LCNet textline_ori: 80x160 and cls_crop_long_lines = false.
  float cls_thresh = 0.9F;
  int cls_batch_size = 6;
  int cls_height = 48;
  int cls_width = 192;
  bool cls_crop_long_lines = true;
};

// PP-OCRv6 detector + recognizer with a deliberately narrow built-in ONNX
// interpreter.  It does not link against ONNX Runtime, Paddle Inference, or
// any other inference engine.
class RPOCR_API OCR {
 public:
  OCR(const std::string& det_model, const std::string& rec_model,
      std::string dictionary_path, Options options = {});
  ~OCR();
  OCR(OCR&&) noexcept;
  OCR& operator=(OCR&&) noexcept;
  OCR(const OCR&) = delete;
  OCR& operator=(const OCR&) = delete;

  // Replaces detector post-process thresholds without reloading models.
  // The handle is not safe to mutate concurrently with Recognize.
  void SetDetectionThresholds(float det_threshold, float det_box_threshold,
                              float det_unclip_ratio);

  // The backend that runs the full det+rec pipeline: Backend::metal only when
  // the Metal GPU graph is really in use, Backend::cpu_only after a Metal
  // request fell back. BackendNote() says why a Metal request fell back
  // (empty otherwise).
  [[nodiscard]] Backend ActiveBackend() const noexcept;
  [[nodiscard]] const std::string& BackendNote() const noexcept;

  [[nodiscard]] std::vector<Result> Recognize(const Image& image) const;
  // rapidocr 3.x use_det / use_rec. use_det=false recognizes the whole image
  // as one text line (one result whose box is the image). use_rec=false
  // returns detector boxes with empty text and the DB box score as
  // confidence. Both true is Recognize(). Both false is invalid. The partial
  // modes run on the CPU graph (also with Backend::metal); Backend::gpu_only
  // throws for them.
  [[nodiscard]] std::vector<Result> Run(const Image& image, bool use_det, bool use_rec) const;
  // Same with an explicit per-call use_cls (the 3-argument form uses
  // Options::use_cls). use_det=false, use_rec=false, use_cls=true classifies
  // the whole image: one result with the image box, empty text and the
  // cls_label / cls_score fields set. use_rec=false with use_det=true returns
  // detector boxes and, with use_cls, each box's orientation.
  [[nodiscard]] std::vector<Result> Run(const Image& image, bool use_det, bool use_rec,
                                        bool use_cls) const;
  // True when a classifier model is loaded and use_cls is in effect by
  // default for this handle (after PPOCR_USE_CLS).
  [[nodiscard]] bool ClsEnabled() const noexcept;
  [[nodiscard]] bool ClsLoaded() const noexcept;
  // Order in which the exact (non-fast) recognition path batches crops.
  // rapidocr main sorts them with np.argsort(width / height), numpy's
  // default unstable sort, so equal ratios can land in different batches
  // than the default std::stable_sort; batch padding then changes
  // max_wh_ratio, the word boxes and (slightly) the scores. fn receives each
  // crop's width / height and must fill `order` (pre-sized) with a
  // permutation; anything else keeps the stable order. Empty fn restores it.
  // Set it before recognizing; the call is not synchronized with Run.
  using RecOrderFn =
      std::function<void(const std::vector<double>& ratios, std::vector<std::size_t>& order)>;
  void SetRecOrder(RecOrderFn fn);
  // Runs a batch of independent page images. Results preserve input order;
  // each nested vector is identical to calling Recognize() for that image.
  // This is intentionally a page-level scheduler rather than padding unlike
  // sized detector images into one tensor, which would waste compute and can
  // change PP-OCR's dynamic detector resize semantics.
  [[nodiscard]] std::vector<std::vector<Result>> RecognizeBatch(
      const std::vector<Image>& images) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Small no-dependency input helper for the portable P6 PPM format.  The OCR
// core accepts Image directly, so applications may use their own PNG/JPEG
// decoder without coupling it to this library.
[[nodiscard]] RPOCR_API Image LoadPPM(const std::string& path);

}  // namespace rpocr
