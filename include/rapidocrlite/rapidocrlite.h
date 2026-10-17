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
/* tasks = DET|REC is rpocr_recognize. DET only: boxes with empty text and the
   DB box score as confidence. REC only: the whole image is one text line and
   the single result's box is the image rectangle. The partial modes run on
   the CPU graph; with RPOCR_BACKEND_GPU they return RPOCR_ERR_UNSUPPORTED. */
RPOCR_API int rpocr_run(rpocr_ocr* ocr, const rpocr_image* image, int tasks,
                        rpocr_result* result);
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
