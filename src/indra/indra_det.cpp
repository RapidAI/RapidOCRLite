#include "indra_det.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>

namespace ppocr::indra {
namespace {
int RoundUp(int n, int m) { return (n + m - 1) / m * m; }
struct FT {  // NHWC, channel stride Cs (multiple of nb), padded channels are 0
  int C = 0, H = 0, W = 0, Cs = 0;
  std::vector<float> d;
  float* px(int y, int x) { return d.data() + (size_t(y) * W + x) * Cs; }
  const float* px(int y, int x) const { return d.data() + (size_t(y) * W + x) * Cs; }
};
struct Val {
  bool isf = false;
  FT f;
  std::vector<int64_t> i;  // int tensor (1-D or scalar)
  float scalar = 0.f; bool fscalar = false;
};
// buffer recycling: big fresh allocations cost more in page faults than the convs themselves
thread_local std::vector<std::vector<float>>* g_free = nullptr;
thread_local size_t g_live = 0, g_peak = 0;
std::vector<float> Take(size_t n, bool zero) {
  std::vector<float> v;
  if (g_free) {
    int best = -1;
    for (int i = 0; i < int(g_free->size()); ++i)
      if ((*g_free)[i].capacity() >= n && (best < 0 || (*g_free)[i].capacity() < (*g_free)[best].capacity())) best = i;
    if (best >= 0) { v = std::move((*g_free)[best]); g_free->erase(g_free->begin() + best); }
  }
  if (v.capacity() < n && std::getenv("PPOCR_INDRA_DEBUG_ALLOC")) std::fprintf(stderr, "alloc miss %zu MB\n", n * 4 >> 20);
  if (zero) v.assign(n, 0.f); else v.resize(n);
  g_live += v.capacity() * sizeof(float); g_peak = std::max(g_peak, g_live);
  return v;
}
void Give(std::vector<float>&& v) {
  g_live -= std::min(g_live, v.capacity() * sizeof(float));
  if (g_free && v.capacity() > 4096) g_free->push_back(std::move(v));
}
FT Make(int C, int H, int W, int nb, bool zero = false) {
  FT t; t.C = C; t.H = H; t.W = W; t.Cs = RoundUp(C, nb); t.d = Take(size_t(H) * W * t.Cs, zero); return t;
}
}  // namespace

DetNet::DetNet(const std::string& path, int threads) : k_(SelectKern()), pool_(new Pool(threads)), m_(LoadOnnx(path)) {
  const int nb = k_->nb;
  kn_ = k_;
  if (std::string(k_->name) == "avx512" && KernAvx2()) kn_ = KernAvx2();
  std::unordered_map<std::string, int>& uses = uses_;
  for (auto& n : m_.nodes) for (auto& s : n.in) uses[s]++;
  skip_.assign(m_.nodes.size(), 0);
  for (size_t ni = 0; ni < m_.nodes.size(); ++ni) {
    const ONode& n = m_.nodes[ni];
    if (n.op != "Conv") continue;
    const OTensor& w = m_.init.at(n.in[1]);
    ConvW c;
    c.cout = int(w.dims[0]); c.cin = int(w.dims[1]); c.k = int(w.dims[2]);
    if (w.dims[3] != c.k) throw std::runtime_error("indra det: non-square kernel");
    c.s = n.attr.count("strides") ? int(n.attr.at("strides").ints[0]) : 1;
    c.p = n.attr.count("pads") ? int(n.attr.at("pads").ints[0]) : 0;
    c.group = n.attr.count("group") ? int(n.attr.at("group").i) : 1;
    if (n.attr.count("dilations") && n.attr.at("dilations").ints[0] != 1) throw std::runtime_error("indra det: dilation");
    c.act = 0;
    // fuse a following HardSwish that is the conv output's only consumer
    if (ni + 1 < m_.nodes.size() && m_.nodes[ni + 1].op == "HardSwish" && m_.nodes[ni + 1].in[0] == n.out[0] &&
        uses[n.out[0]] == 1) {
      c.act = 1; skip_[ni + 1] = 1; fusedOut_[m_.nodes[ni + 1].out[0]] = int(ni);
    }
    int Cs = RoundUp(c.cout, nb);
    c.b.assign(Cs, 0.f);
    std::copy(m_.init.at(n.in[2]).f.begin(), m_.init.at(n.in[2]).f.end(), c.b.begin());
    int kk = c.k * c.k;
    if (c.group == 1 && c.k == 1 && c.s == 1 && c.p == 0) {
      c.w.resize(size_t(Cs) * c.cin);
      k_->pack(w.f.data(), c.cout, c.cin, c.w.data());
      if (c.cout * 2 <= nb) {
        c.wn.resize(size_t(RoundUp(c.cout, kn_->nb)) * c.cin);
        kn_->pack(w.f.data(), c.cout, c.cin, c.wn.data());
        if (c.cout <= 4) {
          c.w4.assign(size_t(c.cin) * 4, 0.f); c.b4.assign(4, 0.f);
          for (int n = 0; n < c.cout; ++n) { c.b4[n] = c.b[n]; for (int k = 0; k < c.cin; ++k) c.w4[size_t(k) * 4 + n] = w.f[size_t(n) * c.cin + k]; }
        }
      }
    } else if (c.cin == 1 && (c.group == c.cout || (c.group == 1 && c.k <= 8))) {
      // depthwise, or dense conv on a 1-channel image: [k*k][stride] layouts for both channel strides
      int Cv = RoundUp(c.cout, nb / 2);
      c.w.assign(size_t(kk) * Cs, 0.f); c.w2.assign(size_t(kk) * Cv, 0.f);
      for (int ch = 0; ch < c.cout; ++ch) for (int j = 0; j < kk; ++j) {
        c.w[size_t(j) * Cs + ch] = w.f[size_t(ch) * kk + j];
        c.w2[size_t(j) * Cv + ch] = w.f[size_t(ch) * kk + j];
      }
    } else {
      throw std::runtime_error("indra det: unsupported Conv");
    }
    conv_[int(ni)] = std::move(c);
  }
}

void DetNet::Run(const float* x, int H, int W, std::vector<float>& out, int& OC, int& OH, int& OW) const {
  const int nb = k_->nb;
  std::unordered_map<std::string, Val> v;
  {
    Val in; in.isf = true; in.f = Make(1, H, W, 1); in.f.Cs = 1;
    in.f.d.assign(x, x + size_t(H) * W);
    v["x"] = std::move(in);
  }
  auto F = [&](const std::string& s) -> FT& {
    auto it = v.find(s);
    if (it == v.end() || !it->second.isf) throw std::runtime_error("indra det: missing float " + s);
    return it->second.f;
  };
  auto I = [&](const std::string& s) -> const std::vector<int64_t>& {
    auto it = v.find(s);
    if (it != v.end()) return it->second.i;
    auto jt = m_.init.find(s);
    if (jt != m_.init.end()) return jt->second.i;
    throw std::runtime_error("indra det: missing int " + s);
  };
  Pool& pool = *pool_;
  std::lock_guard<std::mutex> runlock(mu_);
  g_free = &free_; g_live = 0; g_peak = 0;
  struct Reset {
    std::vector<std::vector<float>>& f;
    ~Reset() {  // bound the recycled pool (keeps steady-state pages fault-free without hoarding memory)
      g_free = nullptr;
      std::sort(f.begin(), f.end(), [](auto& a, auto& b) { return a.capacity() > b.capacity(); });
      size_t total = 0, keep = 0;
      // keep about one run's peak working set so the next page of a similar size is fault-free
      size_t cap = std::max(g_peak * 2, size_t(64) << 20);
      for (; keep < f.size() && (total += f[keep].capacity() * sizeof(float)) <= cap; ++keep) {}
      f.resize(keep);
    }
  } reset_guard{free_};
  std::unordered_map<std::string, int> left = uses_;
  auto release = [&](const ONode& q) {
    for (auto& s : q.in) {
      auto it = left.find(s);
      if (it != left.end() && --it->second == 0) {
        auto vt = v.find(s);
        if (vt != v.end() && vt->second.isf) { Give(std::move(vt->second.f.d)); v.erase(vt); }
      }
    }
  };
  static const bool prof = std::getenv("PPOCR_INDRA_PROFILE") != nullptr;
  std::map<std::string, double> pt;
  auto t0 = std::chrono::steady_clock::now();
  std::string lastop;
  auto tick = [&](const std::string& op) {
    if (!prof) return;
    auto t1 = std::chrono::steady_clock::now();
    if (!lastop.empty()) pt[lastop] += std::chrono::duration<double, std::milli>(t1 - t0).count();
    t0 = t1; lastop = op;
  };
  // serial 1x1 conv on rows [0,rows) of A (lda) into O (ldo); output stride ldo per pw conventions
  auto pw1x1 = [&](const ConvW& c, const float* A, int rows, int lda, float* O, int ldo) {
    if (c.cout * 2 <= nb) {
      if (!c.w4.empty()) { k_->gemm_n4(A, rows, c.cin, lda, c.w4.data(), c.b4.data(), c.cout, O, ldo, c.act); return; }
      int cb = RoundUp(c.cout, kn_->nb);
      constexpr int CH = 128;
      thread_local std::vector<float> tmp;
      tmp.resize(size_t(CH) * cb);
      for (int l0 = 0; l0 < rows; l0 += CH) {
        int rr = std::min(CH, rows - l0);
        kn_->gemm(A + size_t(l0) * lda, rr, c.cin, lda, c.wn.data(), c.b.data(), cb, tmp.data(), cb, c.act);
        for (int i = 0; i < rr; ++i)
          std::copy(tmp.begin() + size_t(i) * cb, tmp.begin() + size_t(i) * cb + c.cout, O + size_t(l0 + i) * ldo);
      }
    } else {
      k_->gemm(A, rows, c.cin, lda, c.w.data(), c.b.data(), ldo, O, ldo, c.act);
    }
  };
  static const bool nofuse = std::getenv("PPOCR_INDRA_NOFUSE") != nullptr;
  for (size_t ni = 0; ni < m_.nodes.size(); ++ni) {
    if (skip_[ni]) continue;
    {
      std::string tag = m_.nodes[ni].op;
      if (tag == "Conv") { const ConvW& c = conv_.at(int(ni)); tag += c.group > 1 ? "_dw" + std::to_string(c.k) : (c.k == 1 ? "_1x1" : "_stem"); if (std::getenv("PPOCR_INDRA_PROFILE_FINE")) tag += "_" + std::to_string(c.cin) + ">" + std::to_string(c.cout) + "@" + std::to_string(F(m_.nodes[ni].in[0]).H); }
      tick(tag);
    }
    const ONode& n = m_.nodes[ni];
    const std::string& op = n.op;
    const std::string& oname = (conv_.count(int(ni)) && conv_.at(int(ni)).act && ni + 1 < m_.nodes.size())
                                   ? m_.nodes[ni + 1].out[0] : n.out[0];
    Val r;
    if (op == "Conv") {
      const ConvW& c = conv_.at(int(ni));
      const FT& a = F(n.in[0]);
      int Ho = (a.H + 2 * c.p - c.k) / c.s + 1, Wo = (a.W + 2 * c.p - c.k) / c.s + 1;
      // depthwise / 1-channel convs only need channel stride % VW == 0 (gemm outputs need % nb)
      bool vwOut = !(c.group == 1 && c.k == 1);
      bool narrow = c.group == 1 && c.k == 1 && c.s == 1 && c.p == 0 && c.cout * 2 <= nb;
      int stride = narrow ? c.cout : vwOut ? (c.group == c.cout ? a.Cs : RoundUp(c.cout, nb / 2)) : RoundUp(c.cout, nb);
      r.isf = true; r.f.C = c.cout; r.f.H = Ho; r.f.W = Wo; r.f.Cs = stride; r.f.d = Take(size_t(Ho) * Wo * stride, false);
      FT& o = r.f;
      if (c.group == 1 && c.k == 1 && c.s == 1 && c.p == 0 && c.cout * 2 <= nb) {
        // narrow output (e.g. 4 channels): keep the tensor compact instead of padding to nb lanes

        int L = a.H * a.W, cb = RoundUp(c.cout, kn_->nb);
        if (!c.w4.empty()) {
          pool.For(L, 256, [&](int b, int e) {
            k_->gemm_n4(a.d.data() + size_t(b) * a.Cs, e - b, c.cin, a.Cs, c.w4.data(), c.b4.data(), c.cout,
                        o.d.data() + size_t(b) * o.Cs, o.Cs, c.act);
          });
        } else
        pool.For(L, 256, [&](int b, int e) {
          constexpr int CH = 128;
          std::vector<float> tmp(size_t(CH) * cb);
          for (int l0 = b; l0 < e; l0 += CH) {
            int rr = std::min(CH, e - l0);
            kn_->gemm(a.d.data() + size_t(l0) * a.Cs, rr, c.cin, a.Cs, c.wn.data(), c.b.data(), cb, tmp.data(), cb, c.act);
            for (int i = 0; i < rr; ++i)
              std::copy(tmp.begin() + size_t(i) * cb, tmp.begin() + size_t(i) * cb + c.cout, o.d.begin() + size_t(l0 + i) * o.Cs);
          }
        });
      } else if (c.group == 1 && c.k == 1 && c.s == 1 && c.p == 0) {
        int L = a.H * a.W;
        pool.For(L, 256, [&](int b, int e) {
          k_->gemm(a.d.data() + size_t(b) * a.Cs, e - b, c.cin, a.Cs, c.w.data(), c.b.data(), o.Cs,
                   o.d.data() + size_t(b) * o.Cs, o.Cs, c.act);
        });
      } else if (c.group == c.cout && c.cin == 1) {
        if (a.Cs != o.Cs || (a.Cs != RoundUp(c.cout, nb) && a.Cs != RoundUp(c.cout, nb / 2)))
          throw std::runtime_error("indra det: dw channel stride");
        size_t j = ni + 1;
        while (j < m_.nodes.size() && skip_[j]) ++j;
        if (!nofuse && j < m_.nodes.size() && m_.nodes[j].op == "Conv" && conv_.count(int(j)) && uses_.count(oname) &&
            uses_.at(oname) == 1 && m_.nodes[j].in[0] == oname) {
          const ConvW& q = conv_.at(int(j));
          if (q.group == 1 && q.k == 1 && q.s == 1 && q.p == 0 && q.cin == c.cout) {
            // fused depthwise -> pointwise: the dw rows live only in a per-thread cache-sized buffer
            const std::string& oname2 = (q.act && j + 1 < m_.nodes.size()) ? m_.nodes[j + 1].out[0] : m_.nodes[j].out[0];
            bool narrow2 = q.cout * 2 <= nb;
            int stride2 = narrow2 ? q.cout : RoundUp(q.cout, nb);
            Give(std::move(o.d));
            Val r2; r2.isf = true; r2.f.C = q.cout; r2.f.H = Ho; r2.f.W = Wo; r2.f.Cs = stride2;
            r2.f.d = Take(size_t(Ho) * Wo * stride2, false);
            FT& o2 = r2.f; const float* wdw = (a.Cs == RoundUp(c.cout, nb) ? c.w : c.w2).data();
            int RB = std::max(1, int(16384 / (size_t(Wo) * a.Cs + 1)));
            pool.For(Ho, RB, [&](int b, int e) {
              thread_local std::vector<float> t;
              for (int y0 = b; y0 < e; y0 += RB) {
                int y1 = std::min(e, y0 + RB);
                t.resize(size_t(y1 - y0) * Wo * a.Cs);
                k_->dw2d(a.d.data(), a.H, a.W, a.Cs, wdw, c.b.data(), c.k, c.s, c.p, t.data() - size_t(y0) * Wo * a.Cs, Wo, y0, y1, c.act);
                pw1x1(q, t.data(), (y1 - y0) * Wo, a.Cs, o2.d.data() + size_t(y0) * Wo * stride2, stride2);
              }
            });
            release(n);
            v[oname2] = std::move(r2);
            release(m_.nodes[j]);
            ni = j;
            continue;
          }
        }
        pool.For(Ho, 4, [&](int b, int e) {
          k_->dw2d(a.d.data(), a.H, a.W, a.Cs, (a.Cs == RoundUp(c.cout, nb) ? c.w : c.w2).data(), c.b.data(), c.k, c.s, c.p, o.d.data(), Wo, b, e, c.act);
        });
      } else {  // dense stem, 1 input channel (a.Cs == 1)
        pool.For(Ho, 4, [&](int b, int e) {
          k_->c1conv(a.d.data(), a.H, a.W, o.Cs, c.w2.data(), c.b.data(), c.k, c.s, c.p, o.d.data(), Wo, b, e, c.act);
        });
      }
    } else if (op == "HardSwish" || op == "Softplus") {
      r.isf = true; r.f = F(n.in[0]);
      for (float& z : r.f.d)
        z = op == "HardSwish" ? z * std::min(1.f, std::max(0.f, z / 6.f + 0.5f))
                              : (z > 20.f ? z : std::log1p(std::exp(z)));
      if (op == "Softplus") {  // keep padded channels at 0 (softplus(0) != 0)
        FT& t = r.f;
        for (size_t p = 0; p < size_t(t.H) * t.W; ++p) std::fill(t.d.begin() + p * t.Cs + t.C, t.d.begin() + (p + 1) * t.Cs, 0.f);
      }
    } else if (op == "Add") {
      FT& a = F(n.in[0]); const FT& b = F(n.in[1]);
      if (a.C != b.C || a.H != b.H || a.W != b.W || a.Cs != b.Cs) throw std::runtime_error("indra det: Add shape");
      r.isf = true;
      bool steal = left.count(n.in[0]) && left.at(n.in[0]) == 1 && n.in[0] != n.in[1];
      if (steal) r.f = std::move(a);  // last use of a: add in place
      else { r.f = Make(a.C, a.H, a.W, nb); r.f.Cs = a.Cs; }
      {
        float* o = r.f.d.data(); const float* pa = steal ? o : a.d.data(); const float* pb = b.d.data();
        int rows = r.f.H;
        size_t rw = size_t(r.f.W) * r.f.Cs;
        pool.For(rows, 16, [&](int y0, int y1) {
          for (size_t p = size_t(y0) * rw; p < size_t(y1) * rw; ++p) o[p] = pa[p] + pb[p];
        });
      }
    } else if (op == "Mul") {
      const FT& a = F(n.in[0]);
      auto it = v.find(n.in[1]);
      if (it == v.end() || !it->second.fscalar) throw std::runtime_error("indra det: Mul expects scalar");
      r.isf = true; r.f = a;
      for (float& z : r.f.d) z *= it->second.scalar;
    } else if (op == "Constant") {
      const OTensor& t = n.attr.at("value").t;
      if (t.dtype == 1) { r.fscalar = true; r.scalar = t.f.at(0); }
      else r.i = t.i;
    } else if (op == "Shape") {
      const FT& a = F(n.in[0]);
      r.i = {1, a.C, a.H, a.W};
    } else if (op == "Gather") {
      const auto& d = I(n.in[0]); const auto& idx = I(n.in[1]);
      for (auto j : idx) r.i.push_back(d.at(size_t(j < 0 ? j + int64_t(d.size()) : j)));
    } else if (op == "Unsqueeze" && !v.count(n.in[0])) {
      throw std::runtime_error("indra det: Unsqueeze input missing");
    } else if (op == "Unsqueeze" && v.at(n.in[0]).isf) {
      // stem_lat space-to-depth: Unsqueeze(3) -> Reshape -> Reshape -> Transpose(0,1,3,5,2,4) -> Reshape -> Squeeze
      size_t j = ni + 1;
      while (j < m_.nodes.size() && m_.nodes[j].op != "Squeeze") {
        const ONode& q = m_.nodes[j];
        if (q.op == "Transpose") {
          const auto& pm = q.attr.at("perm").ints;
          if (pm != std::vector<int64_t>{0, 1, 3, 5, 2, 4}) throw std::runtime_error("indra det: unexpected s2d perm");
        } else if (q.op != "Reshape" && q.op != "Constant") throw std::runtime_error("indra det: unexpected s2d op " + q.op);
        ++j;
      }
      if (j == m_.nodes.size()) throw std::runtime_error("indra det: s2d without Squeeze");
      const FT& a = F(n.in[0]);
      if (a.H % 2 || a.W % 2) throw std::runtime_error("indra det: s2d odd size");
      int Ho = a.H / 2, Wo = a.W / 2;
      r.isf = true; r.f = Make(a.C * 4, Ho, Wo, nb, false);  // padded lanes never read (gemm consumers read K = C)
      FT& rf = r.f;
      pool.For(Ho, 16, [&](int yb, int ye) {
      for (int y = yb; y < ye; ++y) for (int xx = 0; xx < Wo; ++xx) {
        float* o = rf.px(y, xx);
        for (int c = 0; c < a.C; ++c) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx)
          o[c * 4 + dy * 2 + dx] = a.px(2 * y + dy, 2 * xx + dx)[c];
      }
      });
      v[m_.nodes[j].out[0]] = std::move(r);
      release(n);
      ni = j;
      continue;
    } else if (op == "Unsqueeze" || op == "Cast") {
      r.i = I(n.in[0]);
    } else if (op == "Concat") {
      size_t j = ni + 1;
      while (j < m_.nodes.size() && (skip_[j] || m_.nodes[j].op == "Constant")) ++j;
      bool fl = v.count(n.in[0]) && v.at(n.in[0]).isf;
      if (fl && !nofuse && j < m_.nodes.size() && m_.nodes[j].op == "Conv" && conv_.count(int(j)) &&
          uses_.count(n.out[0]) && uses_.at(n.out[0]) == 1 && m_.nodes[j].in[0] == n.out[0]) {
        const ConvW& q = conv_.at(int(j));
        const FT& a0 = F(n.in[0]);
        bool ok = q.group == 1 && q.k == 1 && q.s == 1 && q.p == 0 && q.cout * 2 > nb;
        int C = 0;
        for (auto& s2 : n.in) { const FT& t = F(s2); C += t.C; ok = ok && t.H == a0.H && t.W == a0.W; }
        if (ok && C == q.cin) {
          // Concat -> 1x1 conv: accumulate per-input GEMMs (same k order as the concatenated sum)
          auto& parts = splitw_[int(j)];
          if (parts.empty()) {
            const auto& W = m_.init.at(m_.nodes[j].in[1]).f;
            int off = 0;
            for (auto& s2 : n.in) {
              int ci = F(s2).C;
              std::vector<float> ws(size_t(q.cout) * ci), pk(size_t(RoundUp(q.cout, nb)) * ci);
              for (int o2 = 0; o2 < q.cout; ++o2) for (int k = 0; k < ci; ++k) ws[size_t(o2) * ci + k] = W[size_t(o2) * q.cin + off + k];
              k_->pack(ws.data(), q.cout, ci, pk.data());
              parts.push_back(std::move(pk)); off += ci;
            }
          }
          const std::string& oname2 = (q.act && j + 1 < m_.nodes.size()) ? m_.nodes[j + 1].out[0] : m_.nodes[j].out[0];
          int Cs2 = RoundUp(q.cout, nb), L = a0.H * a0.W;
          Val r2; r2.isf = true; r2.f.C = q.cout; r2.f.H = a0.H; r2.f.W = a0.W; r2.f.Cs = Cs2;
          r2.f.d = Take(size_t(L) * Cs2, false);
          FT& o2 = r2.f;
          std::vector<const FT*> ins;
          for (auto& s2 : n.in) ins.push_back(&F(s2));
          pool.For(L, 256, [&](int b, int e) {
            for (int l0 = b; l0 < e; l0 += 96) {
              int rr = std::min(96, e - l0);
              float* O = o2.d.data() + size_t(l0) * Cs2;
              for (size_t t = 0; t < ins.size(); ++t) {
                const FT& in = *ins[t];
                int act = t + 1 == ins.size() ? q.act : 0;
                if (t == 0) k_->gemm(in.d.data() + size_t(l0) * in.Cs, rr, in.C, in.Cs, parts[t].data(), q.b.data(), Cs2, O, Cs2, act);
                else k_->gemm_acc(in.d.data() + size_t(l0) * in.Cs, rr, in.C, in.Cs, parts[t].data(), Cs2, O, Cs2, act);
              }
            }
          });
          release(n);
          v[oname2] = std::move(r2);
          release(m_.nodes[j]);
          ni = j;
          continue;
        }
      }
      if (fl) {
        int C = 0; const FT& a0 = F(n.in[0]);
        for (auto& s : n.in) C += F(s).C;
        r.isf = true; r.f = Make(C, a0.H, a0.W, nb, false);
        int off = 0;
        for (auto& s : n.in) {
          const FT& a = F(s);
          if (a.H != a0.H || a.W != a0.W) throw std::runtime_error("indra det: Concat shape");
          FT& rf = r.f;
          pool.For(a.H, 16, [&](int y0, int y1) {
            for (size_t p = size_t(y0) * a.W; p < size_t(y1) * a.W; ++p)
              std::copy(a.d.begin() + p * a.Cs, a.d.begin() + p * a.Cs + a.C, rf.d.begin() + p * rf.Cs + off);
          });
          off += a.C;
        }
      } else {
        for (auto& s : n.in) { const auto& d = I(s); r.i.insert(r.i.end(), d.begin(), d.end()); }
      }
    } else if (op == "Slice") {
      int64_t st = I(n.in[1]).at(0), en = I(n.in[2]).at(0);
      int64_t ax = n.in.size() > 3 ? I(n.in[3]).at(0) : 0;
      if (n.in.size() > 4 && I(n.in[4]).at(0) != 1) throw std::runtime_error("indra det: Slice step");
      if (v.count(n.in[0]) && v.at(n.in[0]).isf) {
        const FT& a = F(n.in[0]);
        if (ax != 1) throw std::runtime_error("indra det: float Slice axis");
        int64_t b = std::clamp<int64_t>(st < 0 ? st + a.C : st, 0, a.C), e = std::clamp<int64_t>(en < 0 ? en + a.C : en, 0, a.C);
        r.isf = true; r.f = Make(int(e - b), a.H, a.W, nb, true);
        for (size_t p = 0; p < size_t(a.H) * a.W; ++p)
          std::copy(a.d.begin() + p * a.Cs + b, a.d.begin() + p * a.Cs + e, r.f.d.begin() + p * r.f.Cs);
      } else {
        const auto& d = I(n.in[0]); int64_t sz = int64_t(d.size());
        int64_t b = std::clamp<int64_t>(st < 0 ? st + sz : st, 0, sz), e = std::clamp<int64_t>(en < 0 ? en + sz : en, 0, sz);
        r.i.assign(d.begin() + b, d.begin() + std::max(b, e));
      }
    } else if (op == "Resize") {
      const FT& a = F(n.in[0]);
      const auto& sz = I(n.in[3]);
      int oh = int(sz.at(2)), ow = int(sz.at(3));
      r.isf = true; r.f = Make(a.C, oh, ow, nb);
      float sy = float(oh) / float(a.H), sx = float(ow) / float(a.W);
      std::vector<int> xi(ow);
      for (int xx = 0; xx < ow; ++xx) xi[xx] = std::min(a.W - 1, int(std::floor(double(xx) / double(sx))));
      FT& rf = r.f;
      pool.For(oh, 16, [&](int yb, int ye) {
        for (int y = yb; y < ye; ++y) {
          int iy = std::min(a.H - 1, int(std::floor(double(y) / double(sy))));
          for (int xx = 0; xx < ow; ++xx) std::copy(a.px(iy, xi[xx]), a.px(iy, xi[xx]) + a.Cs, rf.px(y, xx));
        }
      });
    } else if (op == "DepthToSpace" && n.attr.at("blocksize").i == 2 && ni + 1 < m_.nodes.size()) {
      // fused tail: DepthToSpace(CRD, 2) -> out = concat(ch[0:1], softplus(ch[1:]) * k), written as NCHW
      const FT& a = F(n.in[0]);
      float mul = 1.f; int split = -1;
      for (size_t j = ni + 1; j < m_.nodes.size(); ++j) {
        const ONode& q = m_.nodes[j];
        if (q.op == "Constant") { const OTensor& t = q.attr.at("value").t; if (t.dtype == 1) mul = t.f.at(0); continue; }
        if (q.op == "Slice") continue;
        if (q.op == "Softplus" || q.op == "Mul") continue;
        if (q.op == "Concat" && j + 1 == m_.nodes.size()) continue;
        throw std::runtime_error("indra det: unexpected tail op " + q.op);
      }
      // the first Slice keeps channel [0, split); read its end from the Slice's 'ends' constant
      for (size_t j = ni + 1; j < m_.nodes.size(); ++j)
        if (m_.nodes[j].op == "Slice") {
          for (auto& q : m_.nodes)
            if (q.op == "Constant" && q.out[0] == m_.nodes[j].in[2]) split = int(q.attr.at("value").t.i.at(0));
          break;
        }
      if (a.C % 4 || split < 0) throw std::runtime_error("indra det: tail");
      int C = a.C / 4, oh = a.H * 2, ow = a.W * 2;
      OC = C; OH = oh; OW = ow;
      out.assign(size_t(C) * oh * ow, 0.f);
      pool.For(a.H, 8, [&](int yb, int ye) {
        for (int y = yb; y < ye; ++y) for (int xx = 0; xx < a.W; ++xx) {
          const float* s = a.px(y, xx);
          for (int c = 0; c < C; ++c) for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
            out[(size_t(c) * oh + 2 * y + dy) * ow + 2 * xx + dx] = s[c * 4 + dy * 2 + dx];
          }
        }
      });
      {
        size_t plane = size_t(oh) * ow, tot = plane * (C - split);
        float* base = out.data() + plane * split;
        pool.For(int((tot + 65535) / 65536), 1, [&](int b, int e) {
          size_t s0 = size_t(b) * 65536, s1 = std::min(tot, size_t(e) * 65536);
          k_->softplus_mul(base + s0, s1 - s0, mul);
        });
      }
      tick("output");
      if (prof) { for (auto& [k, ms] : pt) std::fprintf(stderr, "  %-14s %8.2f ms\n", k.c_str(), ms); }
      for (auto& kv : v) if (kv.second.isf) Give(std::move(kv.second.f.d));
      return;
    } else if (op == "DepthToSpace") {
      const FT& a = F(n.in[0]);
      int bs = int(n.attr.at("blocksize").i);
      if (bs != 2 || a.C % 4) throw std::runtime_error("indra det: DepthToSpace");
      int C = a.C / 4;
      r.isf = true; r.f = Make(C, a.H * 2, a.W * 2, nb, true);  // CRD: in channel c*4 + dy*2 + dx
      for (int y = 0; y < a.H; ++y) for (int xx = 0; xx < a.W; ++xx) {
        const float* s = a.px(y, xx);
        for (int dy = 0; dy < 2; ++dy) for (int dx = 0; dx < 2; ++dx) {
          float* o = r.f.px(2 * y + dy, 2 * xx + dx);
          for (int c = 0; c < C; ++c) o[c] = s[c * 4 + dy * 2 + dx];
        }
      }
    } else {
      throw std::runtime_error("indra det: unsupported op " + op);
    }
    if (static const char* dd = std::getenv("PPOCR_INDRA_DUMP"); dd && r.isf) {
      std::string fn = std::string(dd) + "/" + std::to_string(ni) + ".bin";
      if (FILE* fp = std::fopen(fn.c_str(), "wb")) {
        int hd[4] = {r.f.C, r.f.H, r.f.W, r.f.Cs};
        std::fwrite(hd, 4, 4, fp); std::fwrite(r.f.d.data(), 4, r.f.d.size(), fp); std::fclose(fp);
      }
    }
    v[oname] = std::move(r);
    release(n);
    if (conv_.count(int(ni)) && conv_.at(int(ni)).act) release(m_.nodes[ni + 1]);
  }
  tick("output");
  if (prof) { for (auto& [k, ms] : pt) std::fprintf(stderr, "  %-14s %8.2f ms\n", k.c_str(), ms); }
  const FT& o = F(m_.nodes.back().out[0]);
  OC = o.C; OH = o.H; OW = o.W;
  out.resize(size_t(OC) * OH * OW);
  for (int c = 0; c < OC; ++c)
    for (size_t p = 0; p < size_t(OH) * OW; ++p) out[size_t(c) * OH * OW + p] = o.d[p * o.Cs + c];
}
}  // namespace ppocr::indra
