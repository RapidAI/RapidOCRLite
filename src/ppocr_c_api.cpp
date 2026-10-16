#define PPOCR_NO_OPTIONS_INIT_MACRO
#include "ppocr/ppocr.h"

#include "ppocr/ppocr.hpp"
#include "cv_compat.hpp"
#include "vulkan_backend.hpp"
#include "indra/lite_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

thread_local std::string g_error;

void SetError(const std::string& message) { g_error = message; }

void ClearError() { g_error.clear(); }

int Fail(int status, const std::string& message) {
  SetError(message);
  return status;
}

ppocr::Options ToOptions(const ppocr_options* options) {
  ppocr::Options out;
  if (!options) return out;
  ppocr_options local{};
  ppocr_options_init_ex(&local, sizeof(local));
  const int bytes = options->struct_size > 0 ? options->struct_size : static_cast<int>(sizeof(ppocr_options));
  const int copy = bytes < static_cast<int>(sizeof(local)) ? bytes : static_cast<int>(sizeof(local));
  if (copy > 0) std::memcpy(&local, options, static_cast<std::size_t>(copy));
  switch (local.backend) {
    case PPOCR_BACKEND_CPU: out.backend = ppocr::Backend::cpu_only; break;
    case PPOCR_BACKEND_GPU: out.backend = ppocr::Backend::gpu_only; break;
    case PPOCR_BACKEND_METAL: out.backend = ppocr::Backend::metal; break;
    default: out.backend = ppocr::Backend::hybrid; break;
  }
  out.det_threshold = local.det_threshold;
  out.det_box_threshold = local.det_box_threshold;
  out.det_unclip_ratio = local.det_unclip_ratio;
  out.det_limit_side_len = local.det_limit_side_len;
  out.rec_height = local.rec_height;
  out.rec_max_width = local.rec_max_width;
  out.rec_batch_size = local.rec_batch_size;
  out.rec_batch_width_bucket = local.rec_batch_width_bucket;
  out.rec_parallelism = local.rec_parallelism;
  out.hybrid_graph_parallelism = local.hybrid_graph_parallelism;
  out.det_batch_size = local.det_batch_size;
  out.batch_preprocess_parallelism = local.batch_preprocess_parallelism;
  out.image_batch_parallelism = local.image_batch_parallelism;
  out.det_limit_type = local.det_limit_type;
  for (int channel = 0; channel < 3; ++channel) {
    out.det_mean[static_cast<std::size_t>(channel)] = local.det_mean[channel];
    out.det_std[static_cast<std::size_t>(channel)] = local.det_std[channel];
    out.rec_mean[static_cast<std::size_t>(channel)] = local.rec_mean[channel];
    out.rec_std[static_cast<std::size_t>(channel)] = local.rec_std[channel];
  }
  out.det_use_dilation = local.det_use_dilation;
  out.det_max_candidates = local.det_max_candidates;
  out.vulkan_device_index = local.vulkan_device_index;
  out.rec_fast_mode = local.rec_fast_mode != 0;
  // model_type only counts when the caller's struct_size covers it (else AUTO).
  const bool has_model_type = options->struct_size >= static_cast<int>(offsetof(ppocr_options, model_type) + sizeof(int));
  switch (has_model_type ? local.model_type : PPOCR_MODEL_AUTO) {
    case PPOCR_MODEL_PPOCR: out.model_type = ppocr::ModelType::ppocr; break;
    case PPOCR_MODEL_LITE: out.model_type = ppocr::ModelType::lite; break;
    default: out.model_type = ppocr::ModelType::auto_detect; break;
  }
  return out;
}

char* Dup(const std::string& text) {
  char* owned = static_cast<char*>(std::malloc(text.size() + 1));
  if (!owned) return nullptr;
  std::memcpy(owned, text.data(), text.size() + 1);
  return owned;
}

int FillResult(const std::vector<ppocr::Result>& source, ppocr_result* result) {
  result->items = nullptr;
  result->count = 0;
  if (source.empty()) return PPOCR_OK;
  auto* items = static_cast<ppocr_text_box*>(std::calloc(source.size(), sizeof(ppocr_text_box)));
  if (!items) return Fail(PPOCR_ERR_NOMEM, "out of memory");
  for (std::size_t i = 0; i < source.size(); ++i) {
    items[i].text = Dup(source[i].text);
    if (!items[i].text) {
      for (std::size_t j = 0; j < i; ++j) std::free(items[j].text);
      std::free(items);
      return Fail(PPOCR_ERR_NOMEM, "out of memory");
    }
    items[i].confidence = source[i].confidence;
    items[i].bbox_x = source[i].bbox[0];
    items[i].bbox_y = source[i].bbox[1];
    items[i].bbox_w = source[i].bbox[2];
    items[i].bbox_h = source[i].bbox[3];
    for (int k = 0; k < 4; ++k) {
      items[i].box[k].x = static_cast<float>(source[i].box[k][0]);
      items[i].box[k].y = static_cast<float>(source[i].box[k][1]);
    }
  }
  result->items = items;
  result->count = source.size();
  return PPOCR_OK;
}

ppocr::Image Borrow(const ppocr_image& image) {
  ppocr::Image out;
  out.width = image.width;
  out.height = image.height;
  if (image.rgb && image.width > 0 && image.height > 0) {
    const std::size_t bytes = static_cast<std::size_t>(image.width) * image.height * 3;
    out.rgb.assign(image.rgb, image.rgb + bytes);
  }
  return out;
}

}  // namespace

namespace {
// Lite (Indra) results keep their fractional box coordinates in the C ABI.
int FillLite(const std::vector<ppocr::indra::LiteLine>& src, ppocr_result* result) {
  result->items = nullptr;
  result->count = 0;
  if (src.empty()) return PPOCR_OK;
  auto* items = static_cast<ppocr_text_box*>(std::calloc(src.size(), sizeof(ppocr_text_box)));
  if (!items) return Fail(PPOCR_ERR_NOMEM, "out of memory");
  for (std::size_t i = 0; i < src.size(); ++i) {
    const auto& l = src[i];
    items[i].text = Dup(l.text);
    if (!items[i].text) {
      for (std::size_t j = 0; j < i; ++j) std::free(items[j].text);
      std::free(items);
      return Fail(PPOCR_ERR_NOMEM, "out of memory");
    }
    items[i].confidence = l.conf;
    const int x0 = static_cast<int>(std::floor(l.x0)), y0 = static_cast<int>(std::floor(l.y0));
    items[i].bbox_x = x0;
    items[i].bbox_y = y0;
    items[i].bbox_w = static_cast<int>(std::ceil(l.x1)) - x0;
    items[i].bbox_h = static_cast<int>(std::ceil(l.y1)) - y0;
    items[i].box[0] = {l.x0, l.y0};
    items[i].box[1] = {l.x1, l.y0};
    items[i].box[2] = {l.x1, l.y1};
    items[i].box[3] = {l.x0, l.y1};
  }
  result->items = items;
  result->count = src.size();
  return PPOCR_OK;
}
}  // namespace

// Either the PP-OCRv6 engine or, when det_model is a lite folder (meta.json), the Indra lite engine.
struct ppocr_ocr {
  std::unique_ptr<ppocr::OCR> ocr;
  std::unique_ptr<ppocr::indra::LiteEngine> lite;
  std::string lite_note;
  explicit ppocr_ocr(ppocr::OCR engine_in) : ocr(std::make_unique<ppocr::OCR>(std::move(engine_in))) {}
  explicit ppocr_ocr(std::unique_ptr<ppocr::indra::LiteEngine> l) : lite(std::move(l)) { lite_note = lite->Note(); }
  ppocr::OCR& engine() { return *ocr; }
  const ppocr::OCR& engine() const { return *ocr; }
};

extern "C" {

void ppocr_options_init_ex(ppocr_options* options, size_t size) {
  if (!options || size < sizeof(int)) return;
  ppocr_options full;
  ppocr_options* out = options;
  options = &full;
  std::memset(options, 0, sizeof(*options));
  options->struct_size = static_cast<int>(sizeof(ppocr_options));
  options->backend = PPOCR_BACKEND_HYBRID;
  const ppocr::Options defaults;
  options->det_threshold = defaults.det_threshold;
  options->det_box_threshold = defaults.det_box_threshold;
  options->det_unclip_ratio = defaults.det_unclip_ratio;
  options->det_limit_side_len = defaults.det_limit_side_len;
  options->rec_height = defaults.rec_height;
  options->rec_max_width = defaults.rec_max_width;
  options->rec_batch_size = defaults.rec_batch_size;
  options->rec_batch_width_bucket = defaults.rec_batch_width_bucket;
  options->rec_parallelism = defaults.rec_parallelism;
  options->hybrid_graph_parallelism = defaults.hybrid_graph_parallelism;
  options->det_batch_size = defaults.det_batch_size;
  options->batch_preprocess_parallelism = defaults.batch_preprocess_parallelism;
  options->image_batch_parallelism = defaults.image_batch_parallelism;
  options->det_limit_type = defaults.det_limit_type;
  for (int channel = 0; channel < 3; ++channel) {
    options->det_mean[channel] = defaults.det_mean[static_cast<std::size_t>(channel)];
    options->det_std[channel] = defaults.det_std[static_cast<std::size_t>(channel)];
    options->rec_mean[channel] = defaults.rec_mean[static_cast<std::size_t>(channel)];
    options->rec_std[channel] = defaults.rec_std[static_cast<std::size_t>(channel)];
  }
  options->det_use_dilation = defaults.det_use_dilation;
  options->det_max_candidates = defaults.det_max_candidates;
  options->vulkan_device_index = defaults.vulkan_device_index;
  options->rec_fast_mode = defaults.rec_fast_mode ? 1 : 0;
  options->model_type = PPOCR_MODEL_AUTO;
  const size_t n = size < sizeof(full) ? size : sizeof(full);
  std::memcpy(out, &full, n);
  out->struct_size = static_cast<int>(n);
}

// Pre-model_type layout (binaries built against older headers).
void ppocr_options_init(ppocr_options* options) {
  ppocr_options_init_ex(options, offsetof(ppocr_options, model_type));
}

const char* ppocr_last_error(void) { return g_error.c_str(); }

ppocr_ocr* ppocr_create(const char* det_model, const char* rec_model,
                        const char* dictionary, const ppocr_options* options) {
  ClearError();
  int model_type = PPOCR_MODEL_AUTO;
  if (options && options->struct_size >= static_cast<int>(offsetof(ppocr_options, model_type) + sizeof(int)))
    model_type = options->model_type;
  if (model_type != PPOCR_MODEL_AUTO && model_type != PPOCR_MODEL_PPOCR && model_type != PPOCR_MODEL_LITE) {
    Fail(PPOCR_ERR_INVALID_ARGUMENT, "model_type must be PPOCR_MODEL_AUTO, PPOCR_MODEL_PPOCR or PPOCR_MODEL_LITE");
    return nullptr;
  }
  const bool lite_dir = det_model && ppocr::indra::LiteEngine::IsLiteDir(det_model);
  if (model_type == PPOCR_MODEL_LITE && !lite_dir) {
    Fail(det_model ? PPOCR_ERR_MODEL : PPOCR_ERR_INVALID_ARGUMENT,
         "model_type=lite needs det_model to be a lite model folder containing meta.json");
    return nullptr;
  }
  if (model_type == PPOCR_MODEL_PPOCR && lite_dir) {
    Fail(PPOCR_ERR_MODEL, "model_type=ppocr, but det_model is a lite model folder (meta.json); use PPOCR_MODEL_LITE or AUTO");
    return nullptr;
  }
  const bool lite = lite_dir;
  if (!det_model || (!lite && (!rec_model || !dictionary))) {
    Fail(PPOCR_ERR_INVALID_ARGUMENT, "model and dictionary paths are required");
    return nullptr;
  }
  try {
    if (lite) {  // model type "lite": det_model is the Indra folder; rec_model / dictionary are ignored
      const ppocr::Options o = ToOptions(options);
      const int hw = static_cast<int>(std::thread::hardware_concurrency());
      const int threads = std::max(1, hw > 0 ? std::min(o.rec_parallelism, hw) : o.rec_parallelism);
      return new ppocr_ocr(std::make_unique<ppocr::indra::LiteEngine>(det_model, threads));
    }
    auto* handle = new ppocr_ocr(ppocr::OCR(det_model, rec_model, dictionary, ToOptions(options)));
    return handle;
  } catch (const std::bad_alloc&) {
    Fail(PPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    const std::string message = ex.what();
    const int status = message.find("cannot open") != std::string::npos ? PPOCR_ERR_IO
        : message.find("GPU-only") != std::string::npos || message.find("Vulkan") != std::string::npos
            ? PPOCR_ERR_UNSUPPORTED
            : PPOCR_ERR_MODEL;
    Fail(status, message);
  }
  return nullptr;
}

void ppocr_destroy(ppocr_ocr* ocr) { delete ocr; }

int ppocr_set_det_thresholds(ppocr_ocr* ocr, float det_threshold, float det_box_threshold,
                             float det_unclip_ratio) {
  ClearError();
  if (!ocr) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null OCR handle");
  try {
    if (ocr->lite) return PPOCR_OK;  // Indra's line decoder has fixed thresholds
    ocr->engine().SetDetectionThresholds(det_threshold, det_box_threshold, det_unclip_ratio);
    return PPOCR_OK;
  } catch (const std::exception& ex) {
    return Fail(PPOCR_ERR_INVALID_ARGUMENT, ex.what());
  }
}

int ppocr_recognize(ppocr_ocr* ocr, const ppocr_image* image, ppocr_result* result) {
  ClearError();
  if (!result) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null result");
  result->items = nullptr;
  result->count = 0;
  if (!ocr || !image) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null OCR handle or image");
  try {
    if (ocr->lite) {
      if (!image->rgb || image->width <= 0 || image->height <= 0) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "empty image");
      return FillLite(ocr->lite->Run(image->rgb, image->width, image->height, 3), result);
    }
    return FillResult(ocr->engine().Recognize(Borrow(*image)), result);
  } catch (const std::bad_alloc&) {
    return Fail(PPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    return Fail(PPOCR_ERR_RUNTIME, ex.what());
  }
}

int ppocr_run(ppocr_ocr* ocr, const ppocr_image* image, int tasks, ppocr_result* result) {
  ClearError();
  if (!result) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null result");
  result->items = nullptr;
  result->count = 0;
  if (!ocr || !image) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null OCR handle or image");
  const bool det = (tasks & PPOCR_TASK_DET) != 0, rec = (tasks & PPOCR_TASK_REC) != 0;
  if ((!det && !rec) || (tasks & ~(PPOCR_TASK_DET | PPOCR_TASK_REC)) != 0) {
    return Fail(PPOCR_ERR_INVALID_ARGUMENT, "tasks must be PPOCR_TASK_DET, PPOCR_TASK_REC or both");
  }
  try {
    if (ocr->lite) {
      if (!image->rgb || image->width <= 0 || image->height <= 0) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "empty image");
      return FillLite(ocr->lite->Run(image->rgb, image->width, image->height, (det ? 1 : 0) | (rec ? 2 : 0)), result);
    }
    return FillResult(ocr->engine().Run(Borrow(*image), det, rec), result);
  } catch (const ppocr::UnsupportedError& ex) {
    return Fail(PPOCR_ERR_UNSUPPORTED, ex.what());
  } catch (const std::bad_alloc&) {
    return Fail(PPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    return Fail(PPOCR_ERR_RUNTIME, ex.what());
  }
}

int ppocr_recognize_batch(ppocr_ocr* ocr, const ppocr_image* images, size_t count,
                          ppocr_result* outs) {
  ClearError();
  if (!ocr || (!images && count > 0) || !outs) {
    return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null batch argument");
  }
  for (size_t i = 0; i < count; ++i) {
    outs[i].items = nullptr;
    outs[i].count = 0;
  }
  try {
    if (ocr->lite) {
      std::vector<ppocr::indra::LiteEngine::BatchImage> ims;
      for (size_t i = 0; i < count; ++i) {
        if (!images[i].rgb || images[i].width <= 0 || images[i].height <= 0) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "empty image");
        ims.push_back({images[i].rgb, images[i].width, images[i].height});
      }
      auto pages = ocr->lite->RunBatch(ims);
      for (size_t i = 0; i < count; ++i) {
        const int status = FillLite(pages[i], &outs[i]);
        if (status != PPOCR_OK) {
          for (size_t j = 0; j < i; ++j) ppocr_result_free(&outs[j]);
          return status;
        }
      }
      return PPOCR_OK;
    }
    std::vector<ppocr::Image> owned;
    owned.reserve(count);
    for (size_t i = 0; i < count; ++i) owned.push_back(Borrow(images[i]));
    const auto batch = ocr->engine().RecognizeBatch(owned);
    if (batch.size() != count) return Fail(PPOCR_ERR_RUNTIME, "batch size mismatch");
    for (size_t i = 0; i < count; ++i) {
      const int status = FillResult(batch[i], &outs[i]);
      if (status != PPOCR_OK) {
        for (size_t j = 0; j < i; ++j) ppocr_result_free(&outs[j]);
        return status;
      }
    }
    return PPOCR_OK;
  } catch (const std::bad_alloc&) {
    return Fail(PPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    return Fail(PPOCR_ERR_RUNTIME, ex.what());
  }
}

void ppocr_result_free(ppocr_result* result) {
  if (!result) return;
  if (result->items) {
    for (size_t i = 0; i < result->count; ++i) std::free(result->items[i].text);
    std::free(result->items);
  }
  result->items = nullptr;
  result->count = 0;
}

int ppocr_load_ppm(const char* path, ppocr_image_buffer* image) {
  ClearError();
  if (!image) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null image");
  image->width = 0;
  image->height = 0;
  image->rgb = nullptr;
  if (!path) return Fail(PPOCR_ERR_INVALID_ARGUMENT, "null path");
  try {
    ppocr::Image loaded = ppocr::LoadPPM(path);
    const std::size_t bytes = loaded.rgb.size();
    image->rgb = static_cast<uint8_t*>(std::malloc(bytes));
    if (!image->rgb) return Fail(PPOCR_ERR_NOMEM, "out of memory");
    std::memcpy(image->rgb, loaded.rgb.data(), bytes);
    image->width = loaded.width;
    image->height = loaded.height;
    return PPOCR_OK;
  } catch (const std::exception& ex) {
    return Fail(PPOCR_ERR_IO, ex.what());
  }
}

void ppocr_image_free(ppocr_image_buffer* image) {
  if (!image) return;
  std::free(image->rgb);
  image->rgb = nullptr;
  image->width = 0;
  image->height = 0;
}

void ppocr_query_backend_info(ppocr_backend_info* info) {
  if (!info) return;
  std::memset(info, 0, sizeof(*info));
  try {
    const ppocr::BackendInfo source = ppocr::QueryBackendInfo();
    info->vulkan_loader_available = source.vulkan_loader_available ? 1 : 0;
    info->vulkan_compute_available = source.vulkan_compute_available ? 1 : 0;
    info->full_graph_gpu_available = source.full_graph_gpu_available ? 1 : 0;
    info->vulkan_runtime_generation = source.vulkan_runtime_generation;
    std::snprintf(info->device_name, sizeof(info->device_name), "%s", source.device_name.c_str());
  } catch (const std::exception& ex) {
    SetError(ex.what());
  }
}

void ppocr_request_vulkan_device(int device_index) {
  ppocr::detail::RequestVulkanDeviceIndex(device_index);
}

int ppocr_vulkan_validation_error_count(void) {
  return ppocr::detail::VulkanValidationErrorCount();
}

int ppocr_active_backend(const ppocr_ocr* ocr) {
  if (!ocr) return -1;
  if (ocr->lite) return PPOCR_BACKEND_CPU;
  switch (ocr->engine().ActiveBackend()) {
    case ppocr::Backend::cpu_only: return PPOCR_BACKEND_CPU;
    case ppocr::Backend::gpu_only: return PPOCR_BACKEND_GPU;
    case ppocr::Backend::hybrid: return PPOCR_BACKEND_HYBRID;
    case ppocr::Backend::metal: return PPOCR_BACKEND_METAL;
  }
  return -1;
}

const char* ppocr_backend_note(const ppocr_ocr* ocr) {
  if (ocr && ocr->lite) return ocr->lite_note.c_str();
  return ocr ? ocr->engine().BackendNote().c_str() : "";
}

int ppocr_metal_available(void) {
  try {
    return ppocr::MetalAvailable() ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

int ppocr_resize_linear_rgb(const unsigned char* src, int src_width, int src_height,
                            unsigned char* dst, int dst_width, int dst_height) {
  if (!src || !dst || src_width <= 0 || src_height <= 0 || dst_width <= 0 || dst_height <= 0)
    return 1;
  ppocr::detail::cvc::ResizeLinear8u(src, src_width, src_height, std::size_t(src_width) * 3, 3,
                                     dst, dst_width, dst_height, std::size_t(dst_width) * 3);
  return 0;
}

void ppocr_query_cpu_info(ppocr_cpu_info* info) {
  if (!info) return;
  std::memset(info, 0, sizeof(*info));
  const ppocr::CpuInfo source = ppocr::QueryCpuInfo();
  info->avx2_compiled = source.avx2_compiled ? 1 : 0;
  info->avx512_compiled = source.avx512_compiled ? 1 : 0;
  info->neon_compiled = source.neon_compiled ? 1 : 0;
  info->avx2 = source.avx2 ? 1 : 0;
  info->avx512 = source.avx512 ? 1 : 0;
  info->neon = source.neon ? 1 : 0;
  info->threads = source.threads;
  std::snprintf(info->active_isa, sizeof(info->active_isa), "%s", source.active_isa.c_str());
}

}  // extern "C"
