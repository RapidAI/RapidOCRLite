// C ABI for the native Indra recognizer (used by rapidocr._indra_lite via ctypes).
#include <cstring>
#include <exception>
#include <string>
#include "indra_rec.hpp"
#if defined(_WIN32)
#define INDRA_API extern "C" __declspec(dllexport)
#else
#define INDRA_API extern "C" __attribute__((visibility("default")))
#endif
namespace { thread_local std::string g_err; }
INDRA_API void* rpocr_indra_rec_create(const char* path, int threads) {
  try { return new rpocr::indra::RecNet(path, threads); } catch (const std::exception& e) { g_err = e.what(); return nullptr; }
}
INDRA_API void rpocr_indra_rec_destroy(void* h) { delete static_cast<rpocr::indra::RecNet*>(h); }
INDRA_API int rpocr_indra_rec_run(void* h, const float* x, int B, int H, int W, const int64_t* widths,
                                  int64_t* ids, float* conf, int64_t* lengths) {
  try { static_cast<rpocr::indra::RecNet*>(h)->Run(x, B, H, W, widths, ids, conf, lengths); return 0; }
  catch (const std::exception& e) { g_err = e.what(); return 1; }
}
INDRA_API int rpocr_indra_rec_steps(void* h, int W) { return static_cast<rpocr::indra::RecNet*>(h)->steps(W); }
INDRA_API const char* rpocr_indra_isa(void* h) { return static_cast<rpocr::indra::RecNet*>(h)->isa(); }
INDRA_API const char* rpocr_indra_last_error() { return g_err.c_str(); }
// ---- detector ----
#include "indra_det.hpp"
namespace { struct DetH { rpocr::indra::DetNet net; std::vector<float> out; int c = 0, h = 0, w = 0;
  DetH(const char* p, int t) : net(p, t) {} }; }
INDRA_API void* rpocr_indra_det_create(const char* path, int threads) {
  try { return new DetH(path, threads); } catch (const std::exception& e) { g_err = e.what(); return nullptr; }
}
INDRA_API void rpocr_indra_det_destroy(void* h) { delete static_cast<DetH*>(h); }
// runs the detector; out dims via dims[3] = {C, H, W}; then copy with rpocr_indra_det_fetch
INDRA_API int rpocr_indra_det_run(void* h, const float* x, int H, int W, int* dims) {
  try { auto* d = static_cast<DetH*>(h); d->net.Run(x, H, W, d->out, d->c, d->h, d->w);
        dims[0] = d->c; dims[1] = d->h; dims[2] = d->w; return 0; }
  catch (const std::exception& e) { g_err = e.what(); return 1; }
}
INDRA_API void rpocr_indra_det_fetch(void* h, float* dst) {
  auto* d = static_cast<DetH*>(h); std::memcpy(dst, d->out.data(), d->out.size() * sizeof(float));
}
// ---- script router ----
#include "indra_script.hpp"
INDRA_API void* rpocr_indra_script_create(const char* path, int threads) {
  try { return new rpocr::indra::ScriptNet(path, threads); } catch (const std::exception& e) { g_err = e.what(); return nullptr; }
}
INDRA_API void rpocr_indra_script_destroy(void* h) { delete static_cast<rpocr::indra::ScriptNet*>(h); }
INDRA_API int rpocr_indra_script_run(void* h, const float* x, int B, int H, int W, const int64_t* widths, float* p) {
  try { static_cast<rpocr::indra::ScriptNet*>(h)->Run(x, B, H, W, widths, p); return 0; }
  catch (const std::exception& e) { g_err = e.what(); return 1; }
}
// ---- reading order ----
#include "indra_order.hpp"
INDRA_API void* rpocr_indra_order_create(const char* path, int threads) {
  try { return new rpocr::indra::OrderNet(path, threads); } catch (const std::exception& e) { g_err = e.what(); return nullptr; }
}
INDRA_API void rpocr_indra_order_destroy(void* h) { delete static_cast<rpocr::indra::OrderNet*>(h); }
INDRA_API int rpocr_indra_order_ncls(void* h) { return static_cast<rpocr::indra::OrderNet*>(h)->ncls(); }
INDRA_API int rpocr_indra_order_run(void* h, const float* feats, const uint8_t* mask, int N, float* a, float* same,
                                    float* cls, float* tab) {
  try { static_cast<rpocr::indra::OrderNet*>(h)->Run(feats, mask, N, a, same, cls, tab); return 0; }
  catch (const std::exception& e) { g_err = e.what(); return 1; }
}
