#include "indra_rec.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <thread>

#include "onnx_min.hpp"
#if defined(__APPLE__) && defined(PPOCR_INDRA_METAL)
#include "metal_head.hpp"
#include "metal_rec.hpp"
#endif
#if defined(__APPLE__) && defined(PPOCR_INDRA_ACCELERATE)
#include <Accelerate/Accelerate.h>
#include <cstdlib>
#define INDRA_ACCEL 1
#endif

namespace ppocr::indra {
namespace {
int RoundUp(int n, int m) { return (n + m - 1) / m * m; }
std::vector<float> Packed(const Kern* k, const std::vector<float>& W, int N, int K) {
  std::vector<float> out(size_t(RoundUp(N, k->nb)) * K);
  k->pack(W.data(), N, K, out.data());
  return out;
}
std::vector<float> PadBias(const std::vector<float>& b, int N, int Np, float fill) {
  std::vector<float> o(Np, fill);
  std::copy(b.begin(), b.begin() + N, o.begin());
  return o;
}
}  // namespace

RecNet::RecNet(const std::string& path, int threads) : k_(SelectKern()), threads_(std::max(1, threads)) {
  OModel m = LoadOnnx(path);
  auto W = [&](const std::string& n) -> const OTensor& {
    auto it = m.init.find(n);
    if (it == m.init.end()) throw std::runtime_error("indra rec: missing initializer " + n);
    return it->second;
  };
  std::vector<const ONode*> convs;
  const ONode* bn = nullptr;
  for (auto& n : m.nodes) {
    if (n.op == "Conv") convs.push_back(&n);
    if (n.op == "BatchNormalization" && !bn) bn = &n;
  }
  if (convs.size() < 7 || (convs.size() - 2) % 5 != 0 || !bn)
    throw std::runtime_error("indra rec: unexpected graph layout");
  // stem conv [C,1,kh,kw] stride (kh,kw) + BN folded
  const OTensor& sw = W(convs[0]->in[1]);
  const OTensor& sb = W(convs[0]->in[2]);
  C_ = int(sw.dims[0]); kh_ = int(sw.dims[2]); sw_ = int(sw.dims[3]);
  int K0 = kh_ * sw_;
  const auto& g = W(bn->in[1]).f; const auto& be = W(bn->in[2]).f;
  const auto& mu = W(bn->in[3]).f; const auto& var = W(bn->in[4]).f;
  float eps = bn->attr.count("epsilon") ? bn->attr.at("epsilon").f : 1e-5f;
  std::vector<float> stem(size_t(C_) * K0), stemb(C_);
  for (int c = 0; c < C_; ++c) {
    float s = g[c] / std::sqrt(var[c] + eps);
    for (int k = 0; k < K0; ++k) stem[size_t(c) * K0 + k] = sw.f[size_t(c) * K0 + k] * s;
    stemb[c] = (sb.f[c] - mu[c]) * s + be[c];
  }
  int nb = k_->nb;
  Cp_ = RoundUp(C_, nb);
  stem_ = Packed(k_, stem, C_, K0);
  stemb_ = PadBias(stemb, C_, Cp_, 0.f);
  size_t nblk = (convs.size() - 2) / 5;
  // PPOCR_INDRA_INT8: "head" quantizes the CTC head, "all" (or "1") also the expand / project 1x1 convs
  std::string q8 = std::getenv("PPOCR_INDRA_INT8") ? std::getenv("PPOCR_INDRA_INT8") : "";
  const bool q8all = q8 == "all" || q8 == "1", q8head = q8all || q8 == "head";
#if defined(__APPLE__) && defined(PPOCR_INDRA_METAL)
  std::vector<MetalRecBlock> mblocks;
#endif
  for (size_t i = 0; i < nblk; ++i) {
    const ONode* const* c = &convs[1 + i * 5];
    Block b;
    const OTensor& dw = W(c[0]->in[1]);
    if (dw.dims.back() != 9) throw std::runtime_error("indra rec: dw kernel != 9");
    b.dwT.resize(size_t(9) * C_);
    for (int ch = 0; ch < C_; ++ch)
      for (int j = 0; j < 9; ++j) b.dwT[size_t(j) * C_ + ch] = dw.f[size_t(ch) * 9 + j];
    b.dwb = W(c[0]->in[2]).f;
    const OTensor& ex = W(c[1]->in[1]); E_ = int(ex.dims[0]);
    const OTensor& f1 = W(c[2]->in[1]); S_ = int(f1.dims[0]);
    Ep_ = RoundUp(E_, nb); Sp_ = RoundUp(S_, nb);
    b.ex = Packed(k_, ex.f, E_, C_); b.exb = PadBias(W(c[1]->in[2]).f, E_, Ep_, 0.f);
    b.f1 = f1.f; b.f1b = W(c[2]->in[2]).f;
    b.f2 = W(c[3]->in[1]).f; b.f2b = W(c[3]->in[2]).f;
    b.pr = Packed(k_, W(c[4]->in[1]).f, C_, E_); b.prb = PadBias(W(c[4]->in[2]).f, C_, Cp_, 0.f);
    if (q8all) { b.qex = QuantizeRows(ex.f.data(), E_, C_); b.qpr = QuantizeRows(W(c[4]->in[1]).f.data(), C_, E_); }
#if defined(__APPLE__) && defined(PPOCR_INDRA_METAL)
    {
      MetalRecBlock mb;
      mb.dwT = b.dwT; mb.dwb = b.dwb; mb.ex = ex.f; mb.exb = W(c[1]->in[2]).f;
      mb.f1 = b.f1; mb.f1b = b.f1b; mb.f2 = b.f2; mb.f2b = b.f2b;
      mb.pr = W(c[4]->in[1]).f; mb.prb = W(c[4]->in[2]).f;
      mblocks.push_back(std::move(mb));
    }
#endif
    blocks_.push_back(std::move(b));
  }
  const OTensor& hw = W(convs.back()->in[1]);
  V_ = int(hw.dims[0]); Vp_ = RoundUp(V_, nb);
  head_ = Packed(k_, hw.f, V_, C_);
#ifdef INDRA_ACCEL
  if (!std::getenv("PPOCR_INDRA_NO_ACCEL")) headRaw_ = hw.f;
#endif
  headb_ = PadBias(W(convs.back()->in[2]).f, V_, Vp_, -1.0e30f);
  isa_ = k_->name;
  if (q8head) {
    qhead_ = QuantizeRows(hw.f.data(), V_, C_);
    headRaw_.clear();
    isa_ += std::string("+") + Int8Isa() + (q8all ? "(all)" : "(head)");
  }
#if defined(__APPLE__) && defined(PPOCR_INDRA_METAL)
  if (const char* e = std::getenv("PPOCR_INDRA_METAL"); e && e[0] == '2') {
    try {
      mrec_ = std::make_shared<MetalRec>(stem, stemb, K0, C_, E_, S_, mblocks, hw.f.data(), W(convs.back()->in[2]).f.data(), V_);
      metal_isa_ = std::string(k_->name) + "+metal-rec";
    } catch (const std::exception&) { mrec_.reset(); }
  }
  if (const char* e = std::getenv("PPOCR_INDRA_METAL"); e && e[0] == '1') {
    try {
      metal_ = std::make_shared<MetalHead>(hw.f.data(), W(convs.back()->in[2]).f.data(), V_, C_);
      metal_isa_ = std::string(k_->name) + "+metal-head";
    } catch (const std::exception&) { metal_.reset(); }  // no GPU: stay on the CPU head
  }
#endif
}

void RecNet::Line(const float* x, int H, int W, int L, int64_t* ids, float* conf, float* hout) const {
  if (L <= 0) return;
  const int K0 = kh_ * sw_;
  std::vector<float> patch(size_t(L) * K0), h(size_t(L) * Cp_), t(size_t(L) * Cp_), e(size_t(L) * Ep_);
  for (int l = 0; l < L; ++l)
    for (int r = 0; r < kh_; ++r)
      for (int j = 0; j < sw_; ++j) patch[size_t(l) * K0 + r * sw_ + j] = x[size_t(r) * W + l * sw_ + j];
  k_->gemm(patch.data(), L, K0, K0, stem_.data(), stemb_.data(), Cp_, h.data(), Cp_, 1);
  std::vector<float> s(E_), s1(S_), gate(E_);
  for (const Block& b : blocks_) {
    // depthwise on a compact [L][C] copy (h has stride Cp_ == C_ when C_ % nb == 0)
    k_->dw9(h.data(), L, Cp_, b.dwT.data(), b.dwb.data(), t.data());
    if (!b.qex.empty()) GemmQ(t.data(), L, C_, Cp_, b.qex, b.exb.data(), e.data(), Ep_, 1);
    else k_->gemm(t.data(), L, C_, Cp_, b.ex.data(), b.exb.data(), Ep_, e.data(), Ep_, 1);
    std::fill(s.begin(), s.end(), 0.f);
    for (int l = 0; l < L; ++l) for (int c = 0; c < E_; ++c) s[c] += e[size_t(l) * Ep_ + c];
    float inv = 1.f / std::max(1.f, float(L));
    for (int c = 0; c < E_; ++c) s[c] *= inv;
    for (int o = 0; o < S_; ++o) {
      float a = b.f1b[o];
      for (int c = 0; c < E_; ++c) a += b.f1[size_t(o) * E_ + c] * s[c];
      s1[o] = std::max(a, 0.f);
    }
    for (int o = 0; o < E_; ++o) {
      float a = b.f2b[o];
      for (int c = 0; c < S_; ++c) a += b.f2[size_t(o) * S_ + c] * s1[c];
      gate[o] = 1.f / (1.f + std::exp(-a));
    }
    for (int l = 0; l < L; ++l) for (int c = 0; c < E_; ++c) e[size_t(l) * Ep_ + c] *= gate[c];
    if (!b.qpr.empty()) GemmQ(e.data(), L, E_, Ep_, b.qpr, b.prb.data(), t.data(), Cp_, 0);
    else k_->gemm(e.data(), L, E_, Ep_, b.pr.data(), b.prb.data(), Cp_, t.data(), Cp_, 0);
    for (size_t i = 0; i < size_t(L) * Cp_; ++i) h[i] += t[i];
  }
  if (hout) {  // body only: hand the features to a batched (GPU) head
    for (int l = 0; l < L; ++l) std::copy(h.begin() + size_t(l) * Cp_, h.begin() + size_t(l) * Cp_ + C_, hout + size_t(l) * C_);
    return;
  }
#ifdef INDRA_ACCEL
  if (!headRaw_.empty()) {  // Apple: head GEMM through Accelerate (AMX/SME), bias pre-filled
    constexpr int RA = 128;
    std::vector<float> lg(size_t(std::min(L, RA)) * Vp_);
    for (int l0 = 0; l0 < L; l0 += RA) {
      int r = std::min(RA, L - l0);
      for (int i = 0; i < r; ++i) std::copy(headb_.begin(), headb_.end(), lg.begin() + size_t(i) * Vp_);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, r, V_, C_, 1.f, h.data() + size_t(l0) * Cp_, Cp_,
                  headRaw_.data(), C_, 1.f, lg.data(), Vp_);
      for (int i = 0; i < r; ++i) {
        int a; float c;
        k_->rowmax(lg.data() + size_t(i) * Vp_, Vp_, &a, &c);
        ids[l0 + i] = a; conf[l0 + i] = c;
      }
    }
    return;
  }
#endif
  if (!qhead_.empty()) {
    constexpr int RQ = 64;
    std::vector<float> lg(size_t(RQ) * Vp_, -1.0e30f);
    for (int l0 = 0; l0 < L; l0 += RQ) {
      int r = std::min(RQ, L - l0);
      GemmQ(h.data() + size_t(l0) * Cp_, r, C_, Cp_, qhead_, headb_.data(), lg.data(), Vp_, 0);
      for (int i = 0; i < r; ++i) {
        int a; float c;
        k_->rowmax(lg.data() + size_t(i) * Vp_, Vp_, &a, &c);
        ids[l0 + i] = a; conf[l0 + i] = c;
      }
    }
    return;
  }
  constexpr int RB = 32;  // rows per head pass: head weights (V x C) are streamed once per RB rows
  std::vector<float> logits(size_t(RB) * Vp_);
  for (int l0 = 0; l0 < L; l0 += RB) {
    int r = std::min(RB, L - l0);
    k_->gemm(h.data() + size_t(l0) * Cp_, r, C_, Cp_, head_.data(), headb_.data(), Vp_, logits.data(), Vp_, 0);
    for (int i = 0; i < r; ++i) {
      int a; float c;
      k_->rowmax(logits.data() + size_t(i) * Vp_, Vp_, &a, &c);
      ids[l0 + i] = a; conf[l0 + i] = c;
    }
  }
}

void RecNet::RunRagged(const std::vector<const float*>& x, const std::vector<int>& w, const std::vector<int>& L, int H,
                       int64_t* ids, float* conf) const {
#if defined(__APPLE__) && defined(PPOCR_INDRA_METAL)
  if (mrec_ && H == kh_) {
    const int K0 = kh_ * sw_, B = int(x.size());
    std::vector<int> st(B + 1, 0);
    for (int b = 0; b < B; ++b) st[b + 1] = st[b] + L[b];
    if (st[B] == 0) return;
    std::vector<float> patch(size_t(st[B]) * K0, 0.f);
    for (int b = 0; b < B; ++b)
      for (int l = 0; l < L[b]; ++l) {
        float* d = patch.data() + size_t(st[b] + l) * K0;
        int c0 = l * sw_, n = std::max(0, std::min(sw_, w[b] - c0));
        for (int r = 0; r < kh_ && n > 0; ++r) std::copy(x[b] + size_t(r) * w[b] + c0, x[b] + size_t(r) * w[b] + c0 + n, d + r * sw_);
      }
    mrec_->Run(patch.data(), st[B], st, ids, conf);
    return;
  }
#endif
  (void)x; (void)w; (void)L; (void)H; (void)ids; (void)conf;
  throw std::runtime_error("indra rec: ragged run needs the Metal whole-forward path");
}

void RecNet::Run(const float* x, int B, int H, int W, const int64_t* widths, int64_t* ids, float* conf,
                 int64_t* lengths) const {
  if (H != kh_ || C_ % k_->nb != 0) throw std::runtime_error("indra rec: unsupported input height/channels");
  const int T = W / sw_;
  std::fill(ids, ids + size_t(B) * T, 0);
  std::fill(conf, conf + size_t(B) * T, 0.f);
  for (int b = 0; b < B; ++b) {
    int64_t L = (widths[b] + sw_ - 1) / sw_;
    lengths[b] = std::clamp<int64_t>(L, 0, T);
  }
#if defined(__APPLE__) && defined(PPOCR_INDRA_METAL)
  if (mrec_) {  // whole forward on the GPU: CPU only gathers the stem patches
    const int K0 = kh_ * sw_;
    std::vector<int> st(B + 1, 0);
    for (int b = 0; b < B; ++b) st[b + 1] = st[b] + int(lengths[b]);
    std::vector<float> patch(size_t(st[B]) * K0);
    for (int b = 0; b < B; ++b) {
      const float* xb = x + size_t(b) * H * W;
      for (int l = 0; l < lengths[b]; ++l) {
        float* d = patch.data() + size_t(st[b] + l) * K0;
        for (int r = 0; r < kh_; ++r) std::copy(xb + size_t(r) * W + l * sw_, xb + size_t(r) * W + l * sw_ + sw_, d + r * sw_);
      }
    }
    std::vector<int64_t> gi(st[B]); std::vector<float> gc(st[B]);
    mrec_->Run(patch.data(), st[B], st, gi.data(), gc.data());
    for (int b = 0; b < B; ++b)
      for (int64_t l = 0; l < lengths[b]; ++l) { ids[size_t(b) * T + l] = gi[st[b] + l]; conf[size_t(b) * T + l] = gc[st[b] + l]; }
    return;
  }
  if (metal_) {
    std::vector<size_t> off(B + 1, 0);
    for (int b = 0; b < B; ++b) off[b + 1] = off[b] + size_t(lengths[b]);
    std::vector<float> hall(off[B] * C_);
    std::atomic<int> nx{0};
    auto body = [&] {
      for (int b; (b = nx.fetch_add(1)) < B;)
        Line(x + size_t(b) * H * W, H, W, int(lengths[b]), nullptr, nullptr, hall.data() + off[b] * C_);
    };
    int nt2 = std::min(threads_, B);
    std::vector<std::thread> pl;
    for (int i = 1; i < nt2; ++i) pl.emplace_back(body);
    body();
    for (auto& th : pl) th.join();
    std::vector<int64_t> gi(off[B]); std::vector<float> gc(off[B]);
    metal_->Run(hall.data(), int(off[B]), C_, gi.data(), gc.data());
    for (int b = 0; b < B; ++b)
      for (int64_t l = 0; l < lengths[b]; ++l) { ids[size_t(b) * T + l] = gi[off[b] + l]; conf[size_t(b) * T + l] = gc[off[b] + l]; }
    return;
  }
#endif
  std::atomic<int> next{0};
  auto work = [&] {
    for (int b; (b = next.fetch_add(1)) < B;)
      Line(x + size_t(b) * H * W, H, W, int(lengths[b]), ids + size_t(b) * T, conf + size_t(b) * T);
  };
  int nt = std::min(threads_, B);
  std::vector<std::thread> pool;
  for (int i = 1; i < nt; ++i) pool.emplace_back(work);
  work();
  for (auto& th : pool) th.join();
}
}  // namespace ppocr::indra
