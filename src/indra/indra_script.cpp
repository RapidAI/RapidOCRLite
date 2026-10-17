#include "indra_script.hpp"
#include "fmath.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "onnx_min.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
static double g_sp[16];
static struct SPD { ~SPD() { if (std::getenv("PPOCR_INDRA_PROFILE")) for (int i = 0; i < 10; ++i) std::fprintf(stderr, "script stage %d %.1f ms\n", i, g_sp[i]); } } g_spd;

namespace rpocr::indra {
namespace {
int RoundUp(int n, int m) { return (n + m - 1) / m * m; }
}

ScriptNet::ScriptNet(const std::string& path, int threads) : k_(SelectKern()), pool_(new Pool(threads)) {
  OModel m = LoadOnnx(path);
  std::vector<std::string> seq;
  for (auto& n : m.nodes) {
    if (n.op == "AveragePool" || n.op == "Conv" || n.op == "MaxPool" || n.op == "Relu" || n.op == "Gemm") seq.push_back(n.op);
    if (n.op == "AveragePool" || n.op == "MaxPool") {
      const auto& ks = n.attr.at("kernel_shape").ints; const auto& st = n.attr.at("strides").ints;
      if (ks != std::vector<int64_t>{2, 2} || st != std::vector<int64_t>{2, 2}) throw std::runtime_error("indra script: pool");
    }
    if (n.op == "Conv") {
      const OTensor& w = m.init.at(n.in[1]);
      if (w.dims[2] != 3 || w.dims[3] != 3 || n.attr.at("pads").ints[0] != 1 ||
          (n.attr.count("group") && n.attr.at("group").i != 1))
        throw std::runtime_error("indra script: conv");
      L l; l.cout = int(w.dims[0]); l.cin = int(w.dims[1]); l.cs = RoundUp(l.cout, k_->nb / 2);
      // ONNX weight [cout][cin][3][3] -> per ky: [cout][kx][cin], matching 3 neighbouring compact NHWC pixels
      for (int ky = 0; ky < 3; ++ky) {
        std::vector<float> t(size_t(l.cout) * 3 * l.cin);
        for (int o = 0; o < l.cout; ++o) for (int c = 0; c < l.cin; ++c) for (int kx = 0; kx < 3; ++kx)
          t[(size_t(o) * 3 + kx) * l.cin + c] = w.f[(size_t(o) * l.cin + c) * 9 + ky * 3 + kx];
        l.w[ky].resize(size_t(RoundUp(l.cout, k_->nb)) * 3 * l.cin);
        k_->pack(t.data(), l.cout, 3 * l.cin, l.w[ky].data());
      }
      if (l.cin > 1 && l.cin <= 64 && l.cout <= 64 && std::getenv("PPOCR_INDRA_WINO")) {  // Winograd F(2x2,3x3): U = G g G^T per (cout, cin)
        const float G[4][3] = {{1, 0, 0}, {.5f, .5f, .5f}, {.5f, -.5f, .5f}, {0, 0, 1}};
        std::vector<float> U(size_t(16) * l.cout * l.cin);
        for (int o = 0; o < l.cout; ++o) for (int c = 0; c < l.cin; ++c) {
          const float* g = &w.f[(size_t(o) * l.cin + c) * 9];
          float t[4][3];
          for (int i = 0; i < 4; ++i) for (int j = 0; j < 3; ++j) t[i][j] = G[i][0] * g[j] + G[i][1] * g[3 + j] + G[i][2] * g[6 + j];
          for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
            U[(size_t(i * 4 + j) * l.cout + o) * l.cin + c] = t[i][0] * G[j][0] + t[i][1] * G[j][1] + t[i][2] * G[j][2];
        }
        for (int e = 0; e < 16; ++e) {
          l.u[e].resize(size_t(RoundUp(l.cout, k_->nb)) * l.cin);
          k_->pack(&U[size_t(e) * l.cout * l.cin], l.cout, l.cin, l.u[e].data());
        }
      }
      if (l.cin == 1) {  // direct 1-channel conv: channel stride of one vector multiple
        l.cs = RoundUp(l.cout, k_->nb / 2);
        l.w1.assign(size_t(9) * l.cs, 0.f);
        for (int o = 0; o < l.cout; ++o) for (int t = 0; t < 9; ++t) l.w1[size_t(t) * l.cs + o] = w.f[size_t(o) * 9 + t];
      }
      l.b.assign(RoundUp(l.cout, k_->nb), 0.f);
      const auto& b = m.init.at(n.in[2]).f; std::copy(b.begin(), b.end(), l.b.begin());
      layers_.push_back(std::move(l));
    }
    if (n.op == "Gemm") {
      const OTensor& w = m.init.at(n.in[1]);
      if (w.dims[0] != 1) throw std::runtime_error("indra script: gemm");
      fc_ = w.f; fcb_ = m.init.at(n.in[2]).f.at(0);
    }
  }
  std::vector<std::string> want = {"AveragePool"};
  for (int i = 0; i < 4; ++i) { want.push_back("Conv"); want.push_back("Relu"); want.push_back("MaxPool"); }
  want.push_back("Gemm");
  if (seq != want || layers_.empty() || int(fc_.size()) != layers_.back().cout)
    throw std::runtime_error("indra script: unexpected graph layout");
}

float ScriptNet::Item(const float* x, int H, int W, int64_t width) const {
  // Only the first max(1, width/32) output columns are averaged; their receptive field ends well
  // inside valid*32 + 128 input pixels (3x3 convs at strides 2..16 widen it by < 32 px), so the
  // padded tail of a batch can be skipped without changing the result.
  const int stride = W;
  {
    int64_t valid = std::max<int64_t>(1, width / 32);
    W = int(std::min<int64_t>(W, (valid * 32 + 128 + 31) / 32 * 32));
  }
  // a: compact NHWC with a zero row above/below and a zero column left/right: (h+2) x (w+2) x cin
  int h = H / 2, w = W / 2;
  auto idx = [](int y, int xx, int wp, int c) { return (size_t(y + 1) * wp + xx + 1) * c; };
  std::vector<float> a(size_t(h + 2) * (w + 2), 0.f);
  for (int y = 0; y < h; ++y) for (int xx = 0; xx < w; ++xx) {
    const float* r0 = x + size_t(2 * y) * stride + 2 * xx; const float* r1 = r0 + stride;
    a[idx(y, xx, w + 2, 1)] = (r0[0] + r0[1] + r1[0] + r1[1]) * 0.25f;
  }
  std::vector<float> c, nxt;
  int cs = 1;
  int li = 0; auto tp = std::chrono::steady_clock::now();
  auto lap = [&](int i) { auto t = std::chrono::steady_clock::now(); g_sp[i] += std::chrono::duration<double, std::milli>(t - tp).count(); tp = t; };
  for (const L& l : layers_) {
    int K = 3 * l.cin, wp = w + 2;
    int h2 = h / 2, w2 = w / 2, co = l.cout;
    nxt.resize(size_t(h2 + 2) * (w2 + 2) * co);
    // zero only the border (interior is fully written below)
    std::fill(nxt.begin(), nxt.begin() + size_t(w2 + 2) * co, 0.f);
    std::fill(nxt.end() - size_t(w2 + 2) * co, nxt.end(), 0.f);
    for (int y = 0; y < h2; ++y) {
      std::fill_n(nxt.begin() + size_t(y + 1) * (w2 + 2) * co, co, 0.f);
      std::fill_n(nxt.begin() + (size_t(y + 1) * (w2 + 2) + w2 + 1) * co, co, 0.f);
    }
    if (!l.u[0].empty()) {
      // Winograd F(2x2,3x3); each 2x2 output tile is exactly one 2x2 max-pool window
      int T = h2 * w2, ci = l.cin, Np = RoundUp(co, k_->nb);
      thread_local std::vector<float> V, M, zb;
      V.resize(size_t(16) * T * ci); M.resize(size_t(16) * T * Np);
      if (int(zb.size()) < Np) zb.assign(Np, 0.f);
      for (int ty = 0; ty < h2; ++ty) for (int tx = 0; tx < w2; ++tx) {
        int t = ty * w2 + tx;
        const float* r[4];
        for (int i = 0; i < 4; ++i) r[i] = a.data() + (size_t(2 * ty + i) * wp + 2 * tx) * ci;
        // channel-vectorised transforms: rows of 4 pixels x ci contiguous channels
        float q[4][4][64];
        for (int j = 0; j < 4; ++j) {
          const float *d0 = r[0] + j * ci, *d1 = r[1] + j * ci, *d2 = r[2] + j * ci, *d3 = r[3] + j * ci;
          for (int c2 = 0; c2 < ci; ++c2) {
            q[0][j][c2] = d0[c2] - d2[c2]; q[1][j][c2] = d1[c2] + d2[c2];
            q[2][j][c2] = d2[c2] - d1[c2]; q[3][j][c2] = d1[c2] - d3[c2];
          }
        }
        size_t st = size_t(T) * ci;
        for (int i = 0; i < 4; ++i) {
          float* o = V.data() + (size_t(i * 4) * T + t) * ci;
          const float *q0 = q[i][0], *q1 = q[i][1], *q2 = q[i][2], *q3 = q[i][3];
          for (int c2 = 0; c2 < ci; ++c2) {
            o[c2] = q0[c2] - q2[c2]; o[st + c2] = q1[c2] + q2[c2];
            o[2 * st + c2] = q2[c2] - q1[c2]; o[3 * st + c2] = q1[c2] - q3[c2];
          }
        }
      }
      for (int e = 0; e < 16; ++e)
        k_->gemm(V.data() + size_t(e) * T * ci, T, ci, ci, l.u[e].data(), zb.data(), Np, M.data() + size_t(e) * T * Np, Np, 0);
      for (int ty = 0; ty < h2; ++ty) for (int tx = 0; tx < w2; ++tx) {
        int t = ty * w2 + tx;
        float* dst = nxt.data() + idx(ty, tx, w2 + 2, co);
        const float* m = M.data() + size_t(t) * Np;
        size_t st = size_t(T) * Np;
        float p[2][4][64];
        for (int j = 0; j < 4; ++j) {
          const float *m0 = m + (0 + j) * st, *m1 = m + (4 + j) * st, *m2 = m + (8 + j) * st, *m3 = m + (12 + j) * st;
          for (int o = 0; o < co; ++o) { p[0][j][o] = m0[o] + m1[o] + m2[o]; p[1][j][o] = m1[o] - m2[o] - m3[o]; }
        }
        for (int o = 0; o < co; ++o) {
          float y00 = p[0][0][o] + p[0][1][o] + p[0][2][o], y01 = p[0][1][o] - p[0][2][o] - p[0][3][o];
          float y10 = p[1][0][o] + p[1][1][o] + p[1][2][o], y11 = p[1][1][o] - p[1][2][o] - p[1][3][o];
          float mx = std::max(std::max(y00, y01), std::max(y10, y11)) + l.b[o];
          dst[o] = mx > 0.f ? mx : 0.f;
        }
      }
      lap(2 * li);
      a.swap(nxt); h = h2; w = w2; cs = co;
      lap(2 * li + 1); ++li;
      continue;
    }
    // conv two rows at a time into a small buffer, then 2x2 max-pool straight into the next padded input
    c.resize(size_t(2) * w * l.cs);
    const float* Wp[3] = {l.w[0].data(), l.w[1].data(), l.w[2].data()};
    for (int y2 = 0; y2 < h2; ++y2) {
      if (!l.w1.empty()) {
        k_->c1conv(a.data(), h + 2, wp, l.cs, l.w1.data(), l.b.data(), 3, 1, 0, c.data() - size_t(2 * y2) * w * l.cs, w, 2 * y2, 2 * y2 + 2, 2);
      } else {
        for (int yy = 0; yy < 2; ++yy) {
          int y = 2 * y2 + yy;
          const float* A[3] = {a.data() + size_t(y) * wp * l.cin, a.data() + size_t(y + 1) * wp * l.cin,
                               a.data() + size_t(y + 2) * wp * l.cin};
          k_->gemm3(A, w, K, l.cin, Wp, l.b.data(), l.cs, c.data() + size_t(yy) * w * l.cs, l.cs, 2);
        }
      }
      for (int xx = 0; xx < w2; ++xx) {
        float* d = nxt.data() + idx(y2, xx, w2 + 2, co);
        const float* p00 = c.data() + size_t(2 * xx) * l.cs;
        const float* p01 = p00 + l.cs; const float* p10 = p00 + size_t(w) * l.cs; const float* p11 = p10 + l.cs;
        for (int ch = 0; ch < co; ++ch) d[ch] = std::max(std::max(p00[ch], p01[ch]), std::max(p10[ch], p11[ch]));
      }
    }
    lap(2 * li);
    a.swap(nxt); h = h2; w = w2; cs = co;
    lap(2 * li + 1); ++li;
  }
  // mean over H, then masked mean over the first max(1, width/32) columns (truncating int64 division)
  int C = layers_.back().cout;
  int64_t valid = std::max<int64_t>(1, width / 32);
  std::vector<double> acc(C, 0.0);
  std::vector<float> col1(C);
  int cnt = 0;
  for (int xx = 0; xx < w; ++xx) {
    std::fill(col1.begin(), col1.end(), 0.f);
    for (int y = 0; y < h; ++y) for (int ch = 0; ch < C; ++ch) col1[ch] += a[idx(y, xx, w + 2, cs)  + ch];
    if (xx < valid) { for (int ch = 0; ch < C; ++ch) acc[ch] += col1[ch] / float(h); ++cnt; }
  }
  float z = fcb_;
  for (int ch = 0; ch < C; ++ch) z += fc_[ch] * float(acc[ch] / std::max(1, cnt));
  return 1.f / (1.f + fm::Exp(-z));
}

void ScriptNet::Run(const float* x, int B, int H, int W, const int64_t* widths, float* p) const {
  if (H % 32) throw std::runtime_error("indra script: height must be a multiple of 32");
  pool_->For(B, 1, [&](int b, int e) {
    for (int i = b; i < e; ++i) p[i] = Item(x + size_t(i) * H * W, H, W, widths[i]);
  });
}
}  // namespace rpocr::indra
