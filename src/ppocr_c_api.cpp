#define RPOCR_NO_OPTIONS_INIT_MACRO
#include "rapidocrlite/rapidocrlite.h"

#include "rapidocrlite/rapidocrlite.hpp"
#include "np_argsort.hpp"
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

rpocr::Options ToOptions(const rpocr_options* options) {
  rpocr::Options out;
  if (!options) return out;
  rpocr_options local{};
  rpocr_options_init_ex(&local, sizeof(local));
  const int bytes = options->struct_size > 0 ? options->struct_size : static_cast<int>(sizeof(rpocr_options));
  const int copy = bytes < static_cast<int>(sizeof(local)) ? bytes : static_cast<int>(sizeof(local));
  if (copy > 0) std::memcpy(&local, options, static_cast<std::size_t>(copy));
  switch (local.backend) {
    case RPOCR_BACKEND_CPU: out.backend = rpocr::Backend::cpu_only; break;
    case RPOCR_BACKEND_GPU: out.backend = rpocr::Backend::gpu_only; break;
    case RPOCR_BACKEND_METAL: out.backend = rpocr::Backend::metal; break;
    default: out.backend = rpocr::Backend::hybrid; break;
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
  const bool has_model_type = options->struct_size >= static_cast<int>(offsetof(rpocr_options, model_type) + sizeof(int));
  switch (has_model_type ? local.model_type : RPOCR_MODEL_AUTO) {
    case RPOCR_MODEL_PPOCR: out.model_type = rpocr::ModelType::ppocr; break;
    case RPOCR_MODEL_LITE: out.model_type = rpocr::ModelType::lite; break;
    default: out.model_type = rpocr::ModelType::auto_detect; break;
  }
  // Orientation classifier fields: each counts only when struct_size covers
  // it. An older caller does not know use_cls and keeps it off.
  const int given = options->struct_size > 0 ? options->struct_size : static_cast<int>(sizeof(rpocr_options));
  const auto covers = [&](std::size_t offset, std::size_t size) {
    return given >= static_cast<int>(offset + size);
  };
  out.use_cls = covers(offsetof(rpocr_options, use_cls), sizeof(int)) && local.use_cls != 0;
  if (covers(offsetof(rpocr_options, cls_thresh), sizeof(float))) out.cls_thresh = local.cls_thresh;
  if (covers(offsetof(rpocr_options, cls_batch_size), sizeof(int)) && local.cls_batch_size > 0)
    out.cls_batch_size = local.cls_batch_size;
  if (covers(offsetof(rpocr_options, cls_height), sizeof(int)) && local.cls_height > 0)
    out.cls_height = local.cls_height;
  if (covers(offsetof(rpocr_options, cls_width), sizeof(int)) && local.cls_width > 0)
    out.cls_width = local.cls_width;
  if (covers(offsetof(rpocr_options, cls_crop_long_lines), sizeof(int)))
    out.cls_crop_long_lines = local.cls_crop_long_lines != 0;
  if (covers(offsetof(rpocr_options, cls_model), sizeof(const char*)) && local.cls_model)
    out.cls_model = local.cls_model;
  return out;
}

// Per-line orientation stored right after the rpocr_text_box array in the
// same allocation, so rpocr_text_box keeps its layout and rpocr_result_free
// stays a single free().
// Also holds the per-character CTC data of rpocr_result_chars; its arrays
// are separate allocations released by rpocr_result_free.
struct ClsTrailer {
  int label;
  float score;
  int rec_steps;
  int crop_width;
  int crop_height;
  double rec_max_wh_ratio;
  size_t char_count;
  int* char_cols;
  float* char_probs;
};

rpocr_text_box* AllocItems(std::size_t count) {
  const std::size_t bytes = count * sizeof(rpocr_text_box) + count * sizeof(ClsTrailer);
  auto* items = static_cast<rpocr_text_box*>(std::calloc(1, bytes));
  if (!items) return nullptr;
  auto* trailer = reinterpret_cast<ClsTrailer*>(items + count);
  for (std::size_t i = 0; i < count; ++i) trailer[i] = {-1, 0.F, 0, 0, 0, 0.0, 0, nullptr, nullptr};
  return items;
}

ClsTrailer* Trailer(const rpocr_result* result) {
  return reinterpret_cast<ClsTrailer*>(result->items + result->count);
}

char* Dup(const std::string& text) {
  char* owned = static_cast<char*>(std::malloc(text.size() + 1));
  if (!owned) return nullptr;
  std::memcpy(owned, text.data(), text.size() + 1);
  return owned;
}

int FillResult(const std::vector<rpocr::Result>& source, rpocr_result* result) {
  result->items = nullptr;
  result->count = 0;
  if (source.empty()) return RPOCR_OK;
  auto* items = AllocItems(source.size());
  if (!items) return Fail(RPOCR_ERR_NOMEM, "out of memory");
  auto* trailer = reinterpret_cast<ClsTrailer*>(items + source.size());
  for (std::size_t i = 0; i < source.size(); ++i) {
    const auto& src = source[i];
    trailer[i] = {src.cls_label, src.cls_score, src.rec_steps, src.crop_width, src.crop_height,
                  src.rec_max_wh_ratio, 0, nullptr, nullptr};
    bool chars_ok = true;
    const std::size_t n = std::min(src.char_cols.size(), src.char_probs.size());
    if (n > 0) {
      trailer[i].char_cols = static_cast<int*>(std::malloc(n * sizeof(int)));
      trailer[i].char_probs = static_cast<float*>(std::malloc(n * sizeof(float)));
      chars_ok = trailer[i].char_cols && trailer[i].char_probs;
      if (chars_ok) {
        std::memcpy(trailer[i].char_cols, src.char_cols.data(), n * sizeof(int));
        std::memcpy(trailer[i].char_probs, src.char_probs.data(), n * sizeof(float));
        trailer[i].char_count = n;
      }
    }
    items[i].text = chars_ok ? Dup(src.text) : nullptr;
    if (!items[i].text) {
      std::free(trailer[i].char_cols);
      std::free(trailer[i].char_probs);
      for (std::size_t j = 0; j < i; ++j) {
        std::free(trailer[j].char_cols);
        std::free(trailer[j].char_probs);
      }
      for (std::size_t j = 0; j < i; ++j) std::free(items[j].text);
      std::free(items);
      return Fail(RPOCR_ERR_NOMEM, "out of memory");
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
  return RPOCR_OK;
}

rpocr::Image Borrow(const rpocr_image& image) {
  rpocr::Image out;
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
int FillLite(const std::vector<rpocr::indra::LiteLine>& src, rpocr_result* result) {
  result->items = nullptr;
  result->count = 0;
  if (src.empty()) return RPOCR_OK;
  auto* items = AllocItems(src.size());
  if (!items) return Fail(RPOCR_ERR_NOMEM, "out of memory");
  for (std::size_t i = 0; i < src.size(); ++i) {
    const auto& l = src[i];
    items[i].text = Dup(l.text);
    if (!items[i].text) {
      for (std::size_t j = 0; j < i; ++j) std::free(items[j].text);
      std::free(items);
      return Fail(RPOCR_ERR_NOMEM, "out of memory");
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
  return RPOCR_OK;
}
}  // namespace

// Either the PP-OCRv6 engine or, when det_model is a lite folder (meta.json), the Indra lite engine.
struct rpocr_ocr {
  std::unique_ptr<rpocr::OCR> ocr;
  std::unique_ptr<rpocr::indra::LiteEngine> lite;
  std::string lite_note;
  explicit rpocr_ocr(rpocr::OCR engine_in) : ocr(std::make_unique<rpocr::OCR>(std::move(engine_in))) {}
  explicit rpocr_ocr(std::unique_ptr<rpocr::indra::LiteEngine> l) : lite(std::move(l)) { lite_note = lite->Note(); }
  rpocr::OCR& engine() { return *ocr; }
  const rpocr::OCR& engine() const { return *ocr; }
};

extern "C" {

void rpocr_options_init_ex(rpocr_options* options, size_t size) {
  if (!options || size < sizeof(int)) return;
  rpocr_options full;
  rpocr_options* out = options;
  options = &full;
  std::memset(options, 0, sizeof(*options));
  options->struct_size = static_cast<int>(sizeof(rpocr_options));
  options->backend = RPOCR_BACKEND_HYBRID;
  const rpocr::Options defaults;
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
  options->model_type = RPOCR_MODEL_AUTO;
  options->use_cls = defaults.use_cls ? 1 : 0;
  options->cls_thresh = defaults.cls_thresh;
  options->cls_batch_size = defaults.cls_batch_size;
  options->cls_height = defaults.cls_height;
  options->cls_width = defaults.cls_width;
  options->cls_crop_long_lines = defaults.cls_crop_long_lines ? 1 : 0;
  options->cls_model = nullptr;
  const size_t n = size < sizeof(full) ? size : sizeof(full);
  std::memcpy(out, &full, n);
  out->struct_size = static_cast<int>(n);
}

// Pre-model_type layout (binaries built against older headers).
void rpocr_options_init(rpocr_options* options) {
  rpocr_options_init_ex(options, offsetof(rpocr_options, model_type));
}

const char* rpocr_last_error(void) { return g_error.c_str(); }

rpocr_ocr* rpocr_create(const char* det_model, const char* rec_model,
                        const char* dictionary, const rpocr_options* options) {
  ClearError();
  int model_type = RPOCR_MODEL_AUTO;
  if (options && options->struct_size >= static_cast<int>(offsetof(rpocr_options, model_type) + sizeof(int)))
    model_type = options->model_type;
  if (model_type != RPOCR_MODEL_AUTO && model_type != RPOCR_MODEL_PPOCR && model_type != RPOCR_MODEL_LITE) {
    Fail(RPOCR_ERR_INVALID_ARGUMENT, "model_type must be RPOCR_MODEL_AUTO, RPOCR_MODEL_PPOCR or RPOCR_MODEL_LITE");
    return nullptr;
  }
  const bool lite_dir = det_model && rpocr::indra::LiteEngine::IsLiteDir(det_model);
  if (model_type == RPOCR_MODEL_LITE && !lite_dir) {
    Fail(det_model ? RPOCR_ERR_MODEL : RPOCR_ERR_INVALID_ARGUMENT,
         "model_type=lite needs det_model to be a lite model folder containing meta.json");
    return nullptr;
  }
  if (model_type == RPOCR_MODEL_PPOCR && lite_dir) {
    Fail(RPOCR_ERR_MODEL, "model_type=ppocr, but det_model is a lite model folder (meta.json); use RPOCR_MODEL_LITE or AUTO");
    return nullptr;
  }
  const bool lite = lite_dir;
  if (!det_model || (!lite && (!rec_model || !dictionary))) {
    Fail(RPOCR_ERR_INVALID_ARGUMENT, "model and dictionary paths are required");
    return nullptr;
  }
  try {
    if (lite) {  // model type "lite": det_model is the Indra folder; rec_model / dictionary are ignored
      const rpocr::Options o = ToOptions(options);
      const int hw = static_cast<int>(std::thread::hardware_concurrency());
      const int threads = std::max(1, hw > 0 ? std::min(o.rec_parallelism, hw) : o.rec_parallelism);
      return new rpocr_ocr(std::make_unique<rpocr::indra::LiteEngine>(det_model, threads));
    }
    auto* handle = new rpocr_ocr(rpocr::OCR(det_model, rec_model, dictionary, ToOptions(options)));
    return handle;
  } catch (const std::bad_alloc&) {
    Fail(RPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    const std::string message = ex.what();
    const int status = message.find("cannot open") != std::string::npos ? RPOCR_ERR_IO
        : message.find("GPU-only") != std::string::npos || message.find("Vulkan") != std::string::npos
            ? RPOCR_ERR_UNSUPPORTED
            : RPOCR_ERR_MODEL;
    Fail(status, message);
  }
  return nullptr;
}

void rpocr_destroy(rpocr_ocr* ocr) { delete ocr; }

int rpocr_set_det_thresholds(rpocr_ocr* ocr, float det_threshold, float det_box_threshold,
                             float det_unclip_ratio) {
  ClearError();
  if (!ocr) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "null OCR handle");
  try {
    if (ocr->lite) return RPOCR_OK;  // Indra's line decoder has fixed thresholds
    ocr->engine().SetDetectionThresholds(det_threshold, det_box_threshold, det_unclip_ratio);
    return RPOCR_OK;
  } catch (const std::exception& ex) {
    return Fail(RPOCR_ERR_INVALID_ARGUMENT, ex.what());
  }
}

int rpocr_recognize(rpocr_ocr* ocr, const rpocr_image* image, rpocr_result* result) {
  ClearError();
  if (!result) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "null result");
  result->items = nullptr;
  result->count = 0;
  if (!ocr || !image) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "null OCR handle or image");
  try {
    if (ocr->lite) {
      if (!image->rgb || image->width <= 0 || image->height <= 0) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "empty image");
      return FillLite(ocr->lite->Run(image->rgb, image->width, image->height, 3), result);
    }
    return FillResult(ocr->engine().Recognize(Borrow(*image)), result);
  } catch (const std::bad_alloc&) {
    return Fail(RPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    return Fail(RPOCR_ERR_RUNTIME, ex.what());
  }
}

int rpocr_run(rpocr_ocr* ocr, const rpocr_image* image, int tasks, rpocr_result* result) {
  ClearError();
  if (!result) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "null result");
  result->items = nullptr;
  result->count = 0;
  if (!ocr || !image) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "null OCR handle or image");
  const bool det = (tasks & RPOCR_TASK_DET) != 0, rec = (tasks & RPOCR_TASK_REC) != 0;
  if ((!det && !rec) || (tasks & ~(RPOCR_TASK_DET | RPOCR_TASK_REC | RPOCR_TASK_CLS)) != 0) {
    return Fail(RPOCR_ERR_INVALID_ARGUMENT, "tasks must be RPOCR_TASK_DET, RPOCR_TASK_REC or both");
  }
  if ((tasks & RPOCR_TASK_CLS) != 0 && !ocr->lite) return rpocr_run_ex(ocr, image, tasks, result);
  try {
    if (ocr->lite) {
      if (!image->rgb || image->width <= 0 || image->height <= 0) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "empty image");
      return FillLite(ocr->lite->Run(image->rgb, image->width, image->height, (det ? 1 : 0) | (rec ? 2 : 0)), result);
    }
    return FillResult(ocr->engine().Run(Borrow(*image), det, rec), result);
  } catch (const rpocr::UnsupportedError& ex) {
    return Fail(RPOCR_ERR_UNSUPPORTED, ex.what());
  } catch (const std::bad_alloc&) {
    return Fail(RPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    return Fail(RPOCR_ERR_RUNTIME, ex.what());
  }
}

int rpocr_run_ex(rpocr_ocr* ocr, const rpocr_image* image, int tasks, rpocr_result* result) {
  ClearError();
  if (!result) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "null result");
  result->items = nullptr;
  result->count = 0;
  if (!ocr || !image) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "null OCR handle or image");
  const bool det = (tasks & RPOCR_TASK_DET) != 0, rec = (tasks & RPOCR_TASK_REC) != 0,
             cls = (tasks & RPOCR_TASK_CLS) != 0;
  if ((!det && !rec && !cls) ||
      (tasks & ~(RPOCR_TASK_DET | RPOCR_TASK_REC | RPOCR_TASK_CLS)) != 0) {
    return Fail(RPOCR_ERR_INVALID_ARGUMENT,
                "tasks must combine RPOCR_TASK_DET, RPOCR_TASK_REC and RPOCR_TASK_CLS");
  }
  try {
    if (ocr->lite) {
      if (!det && !rec) return Fail(RPOCR_ERR_UNSUPPORTED, "the lite model has no orientation classifier");
      if (!image->rgb || image->width <= 0 || image->height <= 0) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "empty image");
      return FillLite(ocr->lite->Run(image->rgb, image->width, image->height, (det ? 1 : 0) | (rec ? 2 : 0)), result);
    }
    if (!det && !rec && !ocr->engine().ClsLoaded())
      return Fail(RPOCR_ERR_INVALID_ARGUMENT, "RPOCR_TASK_CLS needs a classifier model (cls_model)");
    return FillResult(ocr->engine().Run(Borrow(*image), det, rec, cls), result);
  } catch (const rpocr::UnsupportedError& ex) {
    return Fail(RPOCR_ERR_UNSUPPORTED, ex.what());
  } catch (const std::bad_alloc&) {
    return Fail(RPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    return Fail(RPOCR_ERR_RUNTIME, ex.what());
  }
}

int rpocr_result_cls(const rpocr_result* result, size_t index, int* label, float* score) {
  if (!result || !result->items || index >= result->count) return RPOCR_ERR_INVALID_ARGUMENT;
  const ClsTrailer& t = Trailer(result)[index];
  if (label) *label = t.label;
  if (score) *score = t.score;
  return RPOCR_OK;
}

int rpocr_result_chars(const rpocr_result* result, size_t index, rpocr_line_chars* chars) {
  if (!result || !result->items || index >= result->count || !chars)
    return RPOCR_ERR_INVALID_ARGUMENT;
  const ClsTrailer& t = Trailer(result)[index];
  chars->rec_steps = t.rec_steps;
  chars->crop_width = t.crop_width;
  chars->crop_height = t.crop_height;
  chars->rec_max_wh_ratio = t.rec_max_wh_ratio;
  chars->count = t.char_count;
  chars->cols = t.char_cols;
  chars->probs = t.char_probs;
  return RPOCR_OK;
}

int rpocr_set_rec_order(rpocr_ocr* ocr, rpocr_rec_order_fn fn, void* user) {
  ClearError();
  if (!ocr) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "null handle");
  if (ocr->lite) return RPOCR_OK;  // the lite pipeline has its own recognizer
  if (!fn) {
    ocr->engine().SetRecOrder({});
    return RPOCR_OK;
  }
  ocr->engine().SetRecOrder([fn, user](const std::vector<double>& ratios,
                                       std::vector<std::size_t>& order) {
    fn(ratios.data(), ratios.size(), order.data(), user);
  });
  return RPOCR_OK;
}

void rpocr_argsort(const double* values, size_t count, size_t* order) {
  if (!order || (!values && count > 0)) return;
  rpocr::detail::NumpyArgsort(values, count, order);
}

int rpocr_argsort_isa(void) { return static_cast<int>(rpocr::detail::NumpyArgsortIsa()); }

int rpocr_cls_loaded(const rpocr_ocr* ocr) {
  if (!ocr || ocr->lite) return 0;
  return ocr->engine().ClsLoaded() ? 1 : 0;
}

int rpocr_recognize_batch(rpocr_ocr* ocr, const rpocr_image* images, size_t count,
                          rpocr_result* outs) {
  ClearError();
  if (!ocr || (!images && count > 0) || !outs) {
    return Fail(RPOCR_ERR_INVALID_ARGUMENT, "null batch argument");
  }
  for (size_t i = 0; i < count; ++i) {
    outs[i].items = nullptr;
    outs[i].count = 0;
  }
  try {
    if (ocr->lite) {
      std::vector<rpocr::indra::LiteEngine::BatchImage> ims;
      for (size_t i = 0; i < count; ++i) {
        if (!images[i].rgb || images[i].width <= 0 || images[i].height <= 0) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "empty image");
        ims.push_back({images[i].rgb, images[i].width, images[i].height});
      }
      auto pages = ocr->lite->RunBatch(ims);
      for (size_t i = 0; i < count; ++i) {
        const int status = FillLite(pages[i], &outs[i]);
        if (status != RPOCR_OK) {
          for (size_t j = 0; j < i; ++j) rpocr_result_free(&outs[j]);
          return status;
        }
      }
      return RPOCR_OK;
    }
    std::vector<rpocr::Image> owned;
    owned.reserve(count);
    for (size_t i = 0; i < count; ++i) owned.push_back(Borrow(images[i]));
    const auto batch = ocr->engine().RecognizeBatch(owned);
    if (batch.size() != count) return Fail(RPOCR_ERR_RUNTIME, "batch size mismatch");
    for (size_t i = 0; i < count; ++i) {
      const int status = FillResult(batch[i], &outs[i]);
      if (status != RPOCR_OK) {
        for (size_t j = 0; j < i; ++j) rpocr_result_free(&outs[j]);
        return status;
      }
    }
    return RPOCR_OK;
  } catch (const std::bad_alloc&) {
    return Fail(RPOCR_ERR_NOMEM, "out of memory");
  } catch (const std::exception& ex) {
    return Fail(RPOCR_ERR_RUNTIME, ex.what());
  }
}

void rpocr_result_free(rpocr_result* result) {
  if (!result) return;
  if (result->items) {
    ClsTrailer* trailer = Trailer(result);
    for (size_t i = 0; i < result->count; ++i) {
      std::free(result->items[i].text);
      std::free(trailer[i].char_cols);
      std::free(trailer[i].char_probs);
    }
    std::free(result->items);
  }
  result->items = nullptr;
  result->count = 0;
}

int rpocr_load_ppm(const char* path, rpocr_image_buffer* image) {
  ClearError();
  if (!image) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "null image");
  image->width = 0;
  image->height = 0;
  image->rgb = nullptr;
  if (!path) return Fail(RPOCR_ERR_INVALID_ARGUMENT, "null path");
  try {
    rpocr::Image loaded = rpocr::LoadPPM(path);
    const std::size_t bytes = loaded.rgb.size();
    image->rgb = static_cast<uint8_t*>(std::malloc(bytes));
    if (!image->rgb) return Fail(RPOCR_ERR_NOMEM, "out of memory");
    std::memcpy(image->rgb, loaded.rgb.data(), bytes);
    image->width = loaded.width;
    image->height = loaded.height;
    return RPOCR_OK;
  } catch (const std::exception& ex) {
    return Fail(RPOCR_ERR_IO, ex.what());
  }
}

void rpocr_image_free(rpocr_image_buffer* image) {
  if (!image) return;
  std::free(image->rgb);
  image->rgb = nullptr;
  image->width = 0;
  image->height = 0;
}

void rpocr_query_backend_info(rpocr_backend_info* info) {
  if (!info) return;
  std::memset(info, 0, sizeof(*info));
  try {
    const rpocr::BackendInfo source = rpocr::QueryBackendInfo();
    info->vulkan_loader_available = source.vulkan_loader_available ? 1 : 0;
    info->vulkan_compute_available = source.vulkan_compute_available ? 1 : 0;
    info->full_graph_gpu_available = source.full_graph_gpu_available ? 1 : 0;
    info->vulkan_runtime_generation = source.vulkan_runtime_generation;
    std::snprintf(info->device_name, sizeof(info->device_name), "%s", source.device_name.c_str());
  } catch (const std::exception& ex) {
    SetError(ex.what());
  }
}

void rpocr_request_vulkan_device(int device_index) {
  rpocr::detail::RequestVulkanDeviceIndex(device_index);
}

int rpocr_vulkan_validation_error_count(void) {
  return rpocr::detail::VulkanValidationErrorCount();
}

int rpocr_active_backend(const rpocr_ocr* ocr) {
  if (!ocr) return -1;
  if (ocr->lite) return RPOCR_BACKEND_CPU;
  switch (ocr->engine().ActiveBackend()) {
    case rpocr::Backend::cpu_only: return RPOCR_BACKEND_CPU;
    case rpocr::Backend::gpu_only: return RPOCR_BACKEND_GPU;
    case rpocr::Backend::hybrid: return RPOCR_BACKEND_HYBRID;
    case rpocr::Backend::metal: return RPOCR_BACKEND_METAL;
  }
  return -1;
}

const char* rpocr_backend_note(const rpocr_ocr* ocr) {
  if (ocr && ocr->lite) return ocr->lite_note.c_str();
  return ocr ? ocr->engine().BackendNote().c_str() : "";
}

int rpocr_metal_available(void) {
  try {
    return rpocr::MetalAvailable() ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

int rpocr_resize_linear_rgb(const unsigned char* src, int src_width, int src_height,
                            unsigned char* dst, int dst_width, int dst_height) {
  if (!src || !dst || src_width <= 0 || src_height <= 0 || dst_width <= 0 || dst_height <= 0)
    return 1;
  rpocr::detail::cvc::ResizeLinear8u(src, src_width, src_height, std::size_t(src_width) * 3, 3,
                                     dst, dst_width, dst_height, std::size_t(dst_width) * 3);
  return 0;
}

void rpocr_query_cpu_info(rpocr_cpu_info* info) {
  if (!info) return;
  std::memset(info, 0, sizeof(*info));
  const rpocr::CpuInfo source = rpocr::QueryCpuInfo();
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
