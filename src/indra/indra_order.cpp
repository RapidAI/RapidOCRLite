#include "indra_order.hpp"
#include "fmath.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "onnx_min.hpp"
#if defined(__APPLE__) && defined(PPOCR_INDRA_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define ORDER_ACCEL 1
#endif

namespace rpocr::indra {
namespace {
int RoundUp(int n, int m) { return (n + m - 1) / m * m; }
inline float Gelu(float x) { return (fm::Erf(x / 1.41421354f) + 1.f) * x * 0.5f; }
}  // namespace

OrderNet::OrderNet(const std::string& path, int threads) : k_(SelectKern()), pool_(new Pool(threads)) {
  OModel m = LoadOnnx(path);
  auto init = [&](const std::string& n) -> const OTensor* { auto it = m.init.find(n); return it == m.init.end() ? nullptr : &it->second; };
  std::vector<Lin> lins;
  std::vector<std::pair<std::vector<float>, std::vector<float>>> lns;
  std::unordered_map<std::string, const ONode*> byOut;
  for (auto& n : m.nodes) byOut[n.out[0]] = &n;
  for (size_t i = 0; i < m.nodes.size(); ++i) {
    const ONode& n = m.nodes[i];
    if (n.op == "MatMul" && init(n.in[1])) {
      const OTensor& w = *init(n.in[1]);
      Lin l; l.K = int(w.dims[0]); l.N = int(w.dims[1]); l.Np = RoundUp(l.N, k_->nb);
      std::vector<float> t(size_t(l.N) * l.K);
      for (int k = 0; k < l.K; ++k) for (int j = 0; j < l.N; ++j) t[size_t(j) * l.K + k] = w.f[size_t(k) * l.N + j];
      l.w.resize(size_t(l.Np) * l.K); k_->pack(t.data(), l.N, l.K, l.w.data());
      l.raw = w.f;  // [K][N], Accelerate path
      l.b.assign(l.Np, 0.f);
      // bias: the next node is Add(initializer, matmul)
      for (size_t j = i + 1; j < m.nodes.size() && j < i + 3; ++j) {
        const ONode& a = m.nodes[j];
        if (a.op == "Add" && (a.in[0] == n.out[0] || a.in[1] == n.out[0])) {
          const OTensor* b = init(a.in[0] == n.out[0] ? a.in[1] : a.in[0]);
          if (b) std::copy(b->f.begin(), b->f.end(), l.b.begin());
          break;
        }
      }
      lins.push_back(std::move(l));
    } else if (n.op == "LayerNormalization") {
      lns.emplace_back(init(n.in[1])->f, init(n.in[2])->f);
      if (n.attr.count("epsilon")) lnEps_ = n.attr.at("epsilon").f;
    } else if (n.op == "Mul" && byOut.count(n.in[1]) && byOut[n.in[1]]->op == "Constant" && freq_.empty()) {
      const OTensor& t = byOut[n.in[1]]->attr.at("value").t;
      if (t.f.size() > 1) freq_ = t.f;
    } else if (n.op == "Div" && byOut.count(n.in[1]) && byOut[n.in[1]]->op == "Constant") {
      const OTensor& t = byOut[n.in[1]]->attr.at("value").t;
      if (t.f.size() == 1 && t.f[0] > 2.f) scale_ = t.f[0];
    }
  }
  // embed(2) + 4 per layer + final 7
  int nl = (int(lins.size()) - 2 - 7) / 4;
  if (nl < 1 || int(lins.size()) != 2 + 4 * nl + 7 || int(lns.size()) != 2 * nl + 1 || freq_.empty())
    throw std::runtime_error("indra order: unexpected graph layout");
  int p = 0;
  e1_ = std::move(lins[p++]); e2_ = std::move(lins[p++]);
  D_ = e2_.N; nfreq_ = int(freq_.size());
  npos_ = (e1_.K - 0) ;  // fixed below
  for (int i = 0; i < nl; ++i) {
    Layer L;
    L.g1 = lns[2 * i].first; L.b1 = lns[2 * i].second; L.g2 = lns[2 * i + 1].first; L.b2 = lns[2 * i + 1].second;
    L.qkv = std::move(lins[p++]); L.proj = std::move(lins[p++]); L.ff1 = std::move(lins[p++]); L.ff2 = std::move(lins[p++]);
    layers_.push_back(std::move(L));
  }
  gN_ = lns.back().first; bN_ = lns.back().second;
  q_ = std::move(lins[p++]); k_w_ = std::move(lins[p++]); bu_ = std::move(lins[p++]); bw_ = std::move(lins[p++]);
  tu_ = std::move(lins[p++]); tw_ = std::move(lins[p++]); cls_ = std::move(lins[p++]);
  ncls_ = cls_.N;
  // input width: F + npos * 2 * nfreq = e1.K, with npos = 4 sliced coordinates
  npos_ = 4; F_ = e1_.K - npos_ * 2 * nfreq_;
  if (F_ <= 0 || layers_[0].qkv.N != 3 * D_ || D_ % H_) throw std::runtime_error("indra order: dims");
}

void OrderNet::Linear(const Lin& l, const float* x, int rows, int ldx, float* y, int ldy, int act) const {
#ifdef ORDER_ACCEL
  for (int r = 0; r < rows; ++r) std::copy(l.b.begin(), l.b.begin() + l.N, y + size_t(r) * ldy);
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, l.N, l.K, 1.f, x, ldx, l.raw.data(), l.N, 1.f, y, ldy);
  if (act == 3)
    pool_->For(rows, 64, [&](int b, int e) {
      for (int r = b; r < e; ++r) k_->gelu_row(y + size_t(r) * ldy, l.N);
    });
  return;
#endif
  pool_->For(rows, 32, [&](int b, int e) {
    k_->gemm(x + size_t(b) * ldx, e - b, l.K, ldx, l.w.data(), l.b.data(), l.Np, y + size_t(b) * ldy, ldy, 0);
    if (act == 3)
      for (int r = b; r < e; ++r) k_->gelu_row(y + size_t(r) * ldy, l.N);
  });
}

// out[i][j] = (u_i . w_j) / scale, then sym: +1 -> out + out^T, -1 -> out - out^T
void OrderNet::Pair(const float* u, int ldu, const float* w, int ldw, int N, int D, float* out, int sym) const {
  int Np = RoundUp(N, k_->nb);
  std::vector<float> tmp(size_t(N) * Np);
#ifdef ORDER_ACCEL
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, D, 1.f, u, ldu, w, ldw, 0.f, tmp.data(), Np);
  pool_->For(N, 16, [&](int b, int e) {
    for (int i = b; i < e; ++i) for (int j = 0; j < N; ++j) tmp[size_t(i) * Np + j] /= scale_;
  });
#else
  std::vector<float> wt(size_t(N) * D), wp(size_t(Np) * D), zero(Np, 0.f);
  for (int j = 0; j < N; ++j) std::copy(w + size_t(j) * ldw, w + size_t(j) * ldw + D, wt.begin() + size_t(j) * D);
  k_->pack(wt.data(), N, D, wp.data());
  pool_->For(N, 16, [&](int b, int e) {
    k_->gemm(u + size_t(b) * ldu, e - b, D, ldu, wp.data(), zero.data(), Np, tmp.data() + size_t(b) * Np, Np, 0);
    for (int i = b; i < e; ++i) for (int j = 0; j < N; ++j) tmp[size_t(i) * Np + j] /= scale_;
  });
#endif
  constexpr int TB = 32;  // cache-blocked transpose-add
  pool_->For((N + TB - 1) / TB, 1, [&](int b, int e) {
    for (int bi = b; bi < e; ++bi)
      for (int j0 = 0; j0 < N; j0 += TB)
        for (int i = bi * TB; i < std::min(N, bi * TB + TB); ++i) {
          const float* ti = tmp.data() + size_t(i) * Np;
          float* o = out + size_t(i) * N;
          int j1 = std::min(N, j0 + TB);
          if (sym > 0) for (int j = j0; j < j1; ++j) o[j] = ti[j] + tmp[size_t(j) * Np + i];
          else for (int j = j0; j < j1; ++j) o[j] = ti[j] - tmp[size_t(j) * Np + i];
        }
  });
}

void OrderNet::Run(const float* feats, const uint8_t* mask, int N, float* a, float* same, float* cls, float* tab) const {
  if (N <= 0) return;
  const int D = D_, Dp = RoundUp(D, k_->nb), dh = D / H_;
  // embedding input: [feats(F) | sin(x*f) (4 x nf) | cos(x*f) (4 x nf)] per position, matching
  // Concat(Sin, Cos) over [N,4,nf] then reshape -> [4][2nf]
  int K0 = e1_.K;
  std::vector<float> x0(size_t(N) * K0);
  for (int i = 0; i < N; ++i) {
    const float* f = feats + size_t(i) * F_;
    float* d = x0.data() + size_t(i) * K0;
    std::copy(f, f + F_, d);
    for (int c = 0; c < npos_; ++c)
      for (int q = 0; q < nfreq_; ++q) {
        float v = f[c] * freq_[q];
        d[F_ + c * 2 * nfreq_ + q] = fm::Sin(v);
        d[F_ + c * 2 * nfreq_ + nfreq_ + q] = fm::Cos(v);
      }
  }
  std::vector<float> t1(size_t(N) * RoundUp(std::max({e1_.N, 3 * D, 2 * D, layers_[0].ff1.N}), k_->nb));
  std::vector<float> x(size_t(N) * Dp), h(size_t(N) * Dp), att(size_t(N) * Dp), ff(size_t(N) * Dp);
  Linear(e1_, x0.data(), N, K0, t1.data(), e1_.Np, 3);
  Linear(e2_, t1.data(), N, e1_.Np, x.data(), Dp, 0);
  auto LN = [&](const float* in, float* out, const std::vector<float>& g, const std::vector<float>& b) {
    pool_->For(N, 64, [&](int s, int e) {
      for (int i = s; i < e; ++i) {
        const float* r = in + size_t(i) * Dp; float* o = out + size_t(i) * Dp;
        float mu = 0.f; for (int j = 0; j < D; ++j) mu += r[j]; mu /= float(D);
        float var = 0.f; for (int j = 0; j < D; ++j) { float z = r[j] - mu; var += z * z; } var /= float(D);
        float inv = 1.f / std::sqrt(var + lnEps_);
        for (int j = 0; j < D; ++j) o[j] = (r[j] - mu) * inv * g[j] + b[j];
      }
    });
  };
  const float qs = std::sqrt(1.f / std::sqrt(float(dh)));
  int Np = RoundUp(N, k_->nb);
  std::vector<float> zero(std::max(Np, Dp), 0.f);
  for (const Layer& L : layers_) {
    LN(x.data(), h.data(), L.g1, L.b1);
    int ldq = L.qkv.Np;
    Linear(L.qkv, h.data(), N, Dp, t1.data(), ldq, 0);
    // per head: S = (q*qs)(k*qs)^T + mask; softmax; @ v
    for (int hd = 0; hd < H_; ++hd) {
      std::vector<float> qh(size_t(N) * dh), kh(size_t(N) * dh), vT(size_t(dh) * N);
      for (int i = 0; i < N; ++i)
        for (int c = 0; c < dh; ++c) {
          qh[size_t(i) * dh + c] = t1[size_t(i) * ldq + hd * dh + c] * qs;
          kh[size_t(i) * dh + c] = t1[size_t(i) * ldq + D + hd * dh + c] * qs;
          vT[size_t(c) * N + i] = t1[size_t(i) * ldq + 2 * D + hd * dh + c];
        }
#ifndef ORDER_ACCEL
      std::vector<float> kp(size_t(Np) * dh), vp(size_t(RoundUp(dh, k_->nb)) * N);
      k_->pack(kh.data(), N, dh, kp.data());
      k_->pack(vT.data(), dh, N, vp.data());
#endif
      int dhp = RoundUp(dh, k_->nb);
#ifdef ORDER_ACCEL
      std::vector<float> vh(size_t(N) * dh);
      for (int i = 0; i < N; ++i) for (int c = 0; c < dh; ++c) vh[size_t(i) * dh + c] = vT[size_t(c) * N + i];
#endif
      pool_->For(N, 16, [&](int b, int e) {
        std::vector<float> S(size_t(e - b) * Np), o(size_t(e - b) * dhp);
#ifdef ORDER_ACCEL
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, e - b, N, dh, 1.f, qh.data() + size_t(b) * dh, dh, kh.data(), dh,
                    0.f, S.data(), Np);
#else
        k_->gemm(qh.data() + size_t(b) * dh, e - b, dh, dh, kp.data(), zero.data(), Np, S.data(), Np, 0);
#endif
        for (int i = 0; i < e - b; ++i) {
          float* s = S.data() + size_t(i) * Np;
          if (mask) for (int j = 0; j < N; ++j) if (!mask[j]) s[j] = -INFINITY;
          k_->softmax_row(s, N);
          for (int j = N; j < Np; ++j) s[j] = 0.f;
        }
#ifdef ORDER_ACCEL
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, e - b, dh, N, 1.f, S.data(), Np, vh.data(), dh, 0.f, o.data(), dhp);
#else
        k_->gemm(S.data(), e - b, N, Np, vp.data(), zero.data(), dhp, o.data(), dhp, 0);
#endif
        for (int i = 0; i < e - b; ++i)
          std::copy(o.begin() + size_t(i) * dhp, o.begin() + size_t(i) * dhp + dh, att.begin() + size_t(b + i) * Dp + hd * dh);
      });
    }
    Linear(L.proj, att.data(), N, Dp, ff.data(), Dp, 0);
    for (size_t i = 0; i < size_t(N) * Dp; ++i) x[i] += ff[i];
    LN(x.data(), h.data(), L.g2, L.b2);
    Linear(L.ff1, h.data(), N, Dp, t1.data(), L.ff1.Np, 3);
    Linear(L.ff2, t1.data(), N, L.ff1.Np, ff.data(), Dp, 0);
    for (size_t i = 0; i < size_t(N) * Dp; ++i) x[i] += ff[i];
  }
  LN(x.data(), h.data(), gN_, bN_);
  std::vector<float> u(size_t(N) * Dp), w(size_t(N) * Dp);
  Linear(q_, h.data(), N, Dp, u.data(), Dp, 0); Linear(k_w_, h.data(), N, Dp, w.data(), Dp, 0);
  Pair(u.data(), Dp, w.data(), Dp, N, D, a, -1);
  Linear(bu_, h.data(), N, Dp, u.data(), Dp, 0); Linear(bw_, h.data(), N, Dp, w.data(), Dp, 0);
  Pair(u.data(), Dp, w.data(), Dp, N, D, same, +1);
  if (tab) {
    std::vector<float> U(size_t(N) * tu_.Np), Wt(size_t(N) * tw_.Np);
    Linear(tu_, h.data(), N, Dp, U.data(), tu_.Np, 0); Linear(tw_, h.data(), N, Dp, Wt.data(), tw_.Np, 0);
    int g = tu_.N / D;
    for (int t = 0; t < g; ++t) Pair(U.data() + t * D, tu_.Np, Wt.data() + t * D, tw_.Np, N, D, tab + size_t(t) * N * N, +1);
  }
  std::vector<float> c(size_t(N) * cls_.Np);
  Linear(cls_, h.data(), N, Dp, c.data(), cls_.Np, 0);
  for (int i = 0; i < N; ++i) std::copy(c.begin() + size_t(i) * cls_.Np, c.begin() + size_t(i) * cls_.Np + ncls_, cls + size_t(i) * ncls_);
}
}  // namespace rpocr::indra
