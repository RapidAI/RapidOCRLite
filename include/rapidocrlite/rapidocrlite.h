#ifndef PPOCR_PPOCR_H_
#define PPOCR_PPOCR_H_

#include "rapidocrlite/export.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Stable C ABI for the PP-OCRv6 engine. Handles are opaque. Strings and
   result arrays returned to the caller are owned by the result object and
   released with rpocr_result_free / rpocr_image_free. The library does not
   throw across this boundary; failures return a status and a thread-local
   message from rpocr_last_error(). */

#define RPOCR_VERSION_MAJOR 5
#define RPOCR_VERSION_MINOR 0
#define RPOCR_VERSION_PATCH 0

#define RPOCR_OK 0
#define RPOCR_ERR_INVALID_ARGUMENT 1
#define RPOCR_ERR_MODEL 2
#define RPOCR_ERR_IO 3
#define RPOCR_ERR_RUNTIME 4
#define RPOCR_ERR_NOMEM 5
#define RPOCR_ERR_UNSUPPORTED 6

#define RPOCR_BACKEND_CPU 0
#define RPOCR_BACKEND_GPU 1
#define RPOCR_BACKEND_HYBRID 2
/* Native Metal GPU graph (Apple). Falls back to CPU when Metal is not built
   in or no Metal device exists; it never fails rpocr_create for that reason. */
#define RPOCR_BACKEND_METAL 3

typedef struct rpocr_ocr rpocr_ocr;

typedef struct rpocr_options {
  int struct_size;
  int backend; /* RPOCR_BACKEND_* */
  float det_threshold;
  float det_box_threshold;
  float det_unclip_ratio;
  int det_limit_side_len;
  int rec_height;
  int rec_max_width;
  int rec_batch_size;
  int rec_batch_width_bucket;
  int rec_parallelism;
  int hybrid_graph_parallelism;
  int det_batch_size;
  int batch_preprocess_parallelism;
  int image_batch_parallelism;
  /* 0: shrink when the long side exceeds det_limit_side_len (C++ default).
     1: grow when the short side is below det_limit_side_len (rapidocr 3.x "min"). */
  int det_limit_type;
  /* Plane order is B, G, R, matching an OpenCV image and rapidocr 3.x. */
  float det_mean[3];
  float det_std[3];
  float rec_mean[3];
  float rec_std[3];
  int det_use_dilation; /* 1: 2x2 dilation before DB components */
  int det_max_candidates;
  /* -1: automatic adapter. >=0: that physical device, including lavapipe.
     Read on the first Vulkan initialization in the process. */
  int vulkan_device_index;
  /* 0 (default): recognition reproduces rapidocr 3.10 exactly (width-sorted
     batches of rec_batch_size padded to the widest line, minimum width
     rec_height*320/48). 1: fast mode, each line runs at about its own width
     (lines of similar width share a batch; no minimum width). Text can differ
     slightly from 3.10. The PPOCR_REC_FAST environment variable (0/1)
     overrides this field. Appended field: callers built against an older
     header pass a smaller struct_size and get 0. */
  int rec_fast_mode;
  /* RPOCR_MODEL_*. AUTO (0, default): det_model naming a folder that holds
     meta.json selects the lite (Indra) model, anything else the PP-OCR models
     (previous behavior). RPOCR_MODEL_LITE: det_model must be a lite folder;
     rec_model and dictionary may be NULL. RPOCR_MODEL_PPOCR: det_model and
     rec_model must be ONNX files. Appended field: callers built against an
     older header pass a smaller struct_size and get AUTO. */
  int model_type;
  /* Appended (5.0.0a14): text-line orientation classification (0/180), the
     rapidocr main Global.use_cls step between detection and recognition.
     rpocr_options_init sets use_cls = 1 and the rapidocr main PP-OCRv4 Cls
     defaults below; the classifier runs only when cls_model names an ONNX
     file (ch_ppocr_mobile_v2.0_cls_mobile.onnx, or a PP-OCRv5 PP-LCNet
     textline_ori model with cls_height 80, cls_width 160,
     cls_crop_long_lines 0). Callers built against an older header pass a
     smaller struct_size and get use_cls = 0 (previous behavior). The
     PPOCR_USE_CLS environment variable (0/1) overrides use_cls and
     PPOCR_CLS_MODEL supplies cls_model when it is NULL. Only the CPU and
     hybrid backends classify; the GPU-only / Metal graphs and the lite model
     leave lines unclassified. */
  int use_cls;
  float cls_thresh;        /* rotate when label "180" scores above it (0.9) */
  int cls_batch_size;      /* 6 */
  int cls_height;          /* 48 */
  int cls_width;           /* 192 */
  int cls_crop_long_lines; /* 1: classify the centre of long lines (v2.0 model) */
  const char* cls_model;   /* copied by rpocr_create; may be NULL */
} rpocr_options;

#define RPOCR_MODEL_AUTO 0
#define RPOCR_MODEL_PPOCR 1
#define RPOCR_MODEL_LITE 2

typedef struct rpocr_image {
  int width;
  int height;
  const uint8_t* rgb; /* packed RGB, row-major, width * height * 3 bytes */
} rpocr_image;

typedef struct rpocr_image_buffer {
  int width;
  int height;
  uint8_t* rgb;
} rpocr_image_buffer;

typedef struct rpocr_point {
  float x;
  float y;
} rpocr_point;

typedef struct rpocr_text_box {
  char* text;
  float confidence;
  int bbox_x;
  int bbox_y;
  int bbox_w;
  int bbox_h;
  rpocr_point box[4]; /* TL, TR, BR, BL */
} rpocr_text_box;

typedef struct rpocr_result {
  rpocr_text_box* items;
  size_t count;
} rpocr_result;

typedef struct rpocr_backend_info {
  int vulkan_loader_available;
  int vulkan_compute_available;
  int full_graph_gpu_available;
  char device_name[256];
  uint64_t vulkan_runtime_generation;
} rpocr_backend_info;

typedef struct rpocr_cpu_info {
  int avx2_compiled;
  int avx512_compiled;
  int neon_compiled;
  int avx2;    /* runtime path enabled (hardware AND not forced off) */
  int avx512;
  int neon;
  int threads;
  char active_isa[16]; /* "avx512", "avx2", "neon", "lasx", "lsx", or "scalar";
                          LoongArch LSX/LASX report through the neon_* fields */
} rpocr_cpu_info;

/* Fills options with the same defaults as rpocr::Options and sets struct_size.
   The exported rpocr_options_init symbol writes only the fields up to
   rec_fast_mode (the layout binaries built against older headers have);
   rpocr_options_init_ex writes min(size, sizeof(rpocr_options)) bytes. Code
   built against this header calls _ex with its own struct size through the
   macro below, so it gets every field, including model_type. */
RPOCR_API void rpocr_options_init(rpocr_options* options);
RPOCR_API void rpocr_options_init_ex(rpocr_options* options, size_t size);
#ifndef RPOCR_NO_OPTIONS_INIT_MACRO
#define rpocr_options_init(options) rpocr_options_init_ex((options), sizeof(*(options)))
#endif

/* Returns NULL on failure. rpocr_last_error() is set. options may be NULL. */
RPOCR_API rpocr_ocr* rpocr_create(const char* det_model, const char* rec_model,
                                  const char* dictionary,
                                  const rpocr_options* options);
RPOCR_API void rpocr_destroy(rpocr_ocr* ocr);

/* Updates detector post-process thresholds on an existing handle. */
RPOCR_API int rpocr_set_det_thresholds(rpocr_ocr* ocr, float det_threshold,
                                       float det_box_threshold,
                                       float det_unclip_ratio);

RPOCR_API int rpocr_recognize(rpocr_ocr* ocr, const rpocr_image* image,
                              rpocr_result* result);
/* Task flags for rpocr_run, matching rapidocr 3.x use_det / use_rec. */
#define RPOCR_TASK_DET 1
#define RPOCR_TASK_REC 2
/* rapidocr main use_cls. rpocr_run / rpocr_recognize follow the handle's
   use_cls (rpocr_options) and treat this bit as "also classify";
   rpocr_run_ex takes the bit literally (absent = no classification). */
#define RPOCR_TASK_CLS 4
/* tasks = DET|REC is rpocr_recognize. DET only: boxes with empty text and the
   DB box score as confidence. REC only: the whole image is one text line and
   the single result's box is the image rectangle. The partial modes run on
   the CPU graph; with RPOCR_BACKEND_GPU they return RPOCR_ERR_UNSUPPORTED. */
RPOCR_API int rpocr_run(rpocr_ocr* ocr, const rpocr_image* image, int tasks,
                        rpocr_result* result);
/* tasks: any non-empty combination of RPOCR_TASK_DET / _REC / _CLS, where
   RPOCR_TASK_CLS alone classifies the whole image (one result, image box,
   empty text). Unlike rpocr_run, use_cls is exactly (tasks & RPOCR_TASK_CLS).
   Read each line's orientation with rpocr_result_cls. */
RPOCR_API int rpocr_run_ex(rpocr_ocr* ocr, const rpocr_image* image, int tasks,
                           rpocr_result* result);
/* Orientation of result item `index`: label -1 when the classifier did not
   run for it, else the class index (0 = "0", 1 = "180") and its probability.
   The line was rotated before recognition when label == 1 and
   score > cls_thresh. Works on every result this library returns. */
RPOCR_API int rpocr_result_cls(const rpocr_result* result, size_t index, int* label,
                               float* score);
/* Per-character CTC data of result item `index` (5.0.0a15), the input of
   rapidocr main's return_word_box / return_single_char_box word boxes.
   cols[k] is the recognizer time step that emitted the k-th decoded
   character (CTC blank / repeat collapsed) and probs[k] its max softmax
   probability. rec_steps is the recognizer output length of the line's
   batch, crop_width x crop_height the rotated line crop
   (get_rotate_crop_image; the whole image when use_det is off) and
   rec_max_wh_ratio the batch's max_wh_ratio, so main's
   line_txt_len = rec_steps * crop_width / crop_height / rec_max_wh_ratio.
   rec_steps == 0 when the path that produced the result does not record it
   (strict GPU-only Vulkan / Metal device graphs, the lite model, detection
   only). The arrays are owned by the result (rpocr_result_free). */
typedef struct rpocr_line_chars {
  int rec_steps;
  int crop_width;
  int crop_height;
  double rec_max_wh_ratio;
  size_t count;
  const int* cols;
  const float* probs;
} rpocr_line_chars;
RPOCR_API int rpocr_result_chars(const rpocr_result* result, size_t index,
                                 rpocr_line_chars* chars);
/* Recognition batch order (5.0.0a15). rapidocr main batches line crops in
   np.argsort(width / height) order, NumPy's default unstable sort; ties
   decide which lines share a padded batch, which changes word boxes and,
   slightly, scores. The engine reproduces that order natively by default
   (rpocr_argsort). A callback set here overrides it: it receives the
   `count` ratios (double width / height of each crop, detection order) and
   fills `order` with a permutation of 0..count-1; an invalid permutation
   keeps the native order. NULL removes the override. Only the exact
   (non-fast) CPU / hybrid / Metal recognition path batches this way. */
typedef void (*rpocr_rec_order_fn)(const double* ratios, size_t count, size_t* order,
                                   void* user);
RPOCR_API int rpocr_set_rec_order(rpocr_ocr* ocr, rpocr_rec_order_fn fn, void* user);
/* np.argsort(values) for a float64 array (kind="quicksort"), index for index
   as NumPy 2.x computes it on this CPU: NumPy dispatches to x86-simd-sort on
   x86 CPUs with its X86_V4 (AVX-512 F/CD/VL/BW/DQ) or X86_V3 (AVX2, FMA,
   BMI1/2, F16C, LZCNT, MOVBE) feature groups and otherwise (older x86, ARM,
   LoongArch, ...) runs its scalar introsort; ties come out differently in
   the three. The vendored x86-simd-sort is the commit NumPy v2.5.4 ships.
   The PPOCR_ARGSORT_ISA environment variable (scalar, avx2, avx512) forces a
   lower variant. rpocr_argsort_isa returns the variant in use: 0 scalar,
   1 avx2, 2 avx512. */
/* Word and character boxes, rapidocr main's word_results for line `index`
   (CalRecBoxes on the line box and its rpocr_result_chars data):
   single_char = 0 is return_word_box (English / number runs as words, CJK
   as characters), 1 adds return_single_char_box (one box per character).
   `words` receives an array owned by the caller, released with
   rpocr_words_free; count 0 for blank lines and lines without character
   data. Boxes are integer pixel corners in main's order_points order;
   confidence is main's per-character round(prob, 5). Identical to the
   Python binding's word_results for the same result (the Python binding
   also maps its resized input back to the original image first). */
typedef struct rpocr_word {
  char* text;
  double confidence;
  rpocr_point box[4];
} rpocr_word;

typedef struct rpocr_words {
  rpocr_word* items;
  size_t count;
} rpocr_words;

RPOCR_API int rpocr_result_words(const rpocr_result* result, size_t index, int single_char,
                                 rpocr_words* words);
RPOCR_API void rpocr_words_free(rpocr_words* words);
RPOCR_API void rpocr_argsort(const double* values, size_t count, size_t* order);
RPOCR_API int rpocr_argsort_isa(void);
/* 1 when this handle has a classifier model loaded, 0 otherwise. */
RPOCR_API int rpocr_cls_loaded(const rpocr_ocr* ocr);
/* outs must point at count result objects. Input order is preserved. */
RPOCR_API int rpocr_recognize_batch(rpocr_ocr* ocr, const rpocr_image* images,
                                    size_t count, rpocr_result* outs);
RPOCR_API void rpocr_result_free(rpocr_result* result);

RPOCR_API int rpocr_load_ppm(const char* path, rpocr_image_buffer* image);
RPOCR_API void rpocr_image_free(rpocr_image_buffer* image);

RPOCR_API void rpocr_query_backend_info(rpocr_backend_info* info);
/* cv2.resize(src, (dst_width, dst_height)) with INTER_LINEAR on a packed
   3-channel 8-bit image, bit-exact with the OpenCV 5.0 wheels rapidocr 3.10
   uses. Returns 0 on success, nonzero for invalid arguments. */
RPOCR_API int rpocr_resize_linear_rgb(const unsigned char* src, int src_width, int src_height,
                                      unsigned char* dst, int dst_width, int dst_height);
/* Pins the Vulkan physical device before the first initialization.
   index < 0 leaves the current choice unchanged. */
RPOCR_API void rpocr_request_vulkan_device(int device_index);
/* ERROR-severity VK_LAYER_KHRONOS_validation messages since process start.
   The layer is enabled only when PPOCR_VULKAN_VALIDATION is set and not "0". */
RPOCR_API int rpocr_vulkan_validation_error_count(void);
/* 1 when this library has the Metal backend and the system has a Metal
   device. Does not initialize the GPU runtime. */
RPOCR_API int rpocr_metal_available(void);
/* RPOCR_BACKEND_* that actually runs this handle's full pipeline:
   RPOCR_BACKEND_METAL only when the Metal GPU graph is in use, and
   RPOCR_BACKEND_CPU when a Metal request fell back. -1 for NULL. */
RPOCR_API int rpocr_active_backend(const rpocr_ocr* ocr);
/* Why a Metal request fell back to CPU; "" when it did not. Owned by the
   handle. */
RPOCR_API const char* rpocr_backend_note(const rpocr_ocr* ocr);
RPOCR_API void rpocr_query_cpu_info(rpocr_cpu_info* info);

/* Thread-local. Pointer is valid until the next API call on this thread. */
RPOCR_API const char* rpocr_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
