// Indra lite pipeline in C++. Ported from indra-ocr 0.3.2 (Apache-2.0, license text in src/indra/LICENSE-indra-ocr): ocr pipeline.py,
// det_post.py, rec_preproc.py, order_post.py, vocab.py. The arithmetic follows the Python
// reference (numpy float32 / float64 semantics, numpy's pairwise float32 sums and its
// x86 SIMD exp/log, Python banker's rounding) so pages come out identical. Built with
// -ffp-contract=off: every fused multiply-add below is explicit.
#include "lite_pipeline.hpp"
#include "fmath.hpp"
#include <chrono>
#include <thread>
#include <atomic>
#include <cstdio>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <future>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#if defined(_MSC_VER)
#include <immintrin.h>
#include <intrin.h>
#endif
#if defined(_WIN32) && !defined(S_ISDIR)
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif

#include "cpu_x86.hpp"
#include "indra_det.hpp"
#include "indra_order.hpp"
#include "indra_rec.hpp"
#include "indra_script.hpp"

extern "C" {
int rpocr_img_resize_u8(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh, int interp, int flags);
int rpocr_img_warp_affine(const void* src, int sw, int sh, void* dst, int dw, int dh, const double* M, int is_float,
                          int border, double bval, int uf, int flags);
void rpocr_img_rotation_matrix(double cx, double cy, double angle, double scale, int flags, double* M);
void rpocr_img_set_local_threads(int n);
int rpocr_img_get_threads(void);
void rpocr_img_rgb2gray(const uint8_t* rgb, int w, int h, int stride, uint8_t* dst);
int rpocr_img_cc4(const uint8_t* mask, int w, int h, int32_t* labels, int32_t* stats, int max_n);
void rpocr_img_set_threads(int n);
}

namespace rpocr::indra {
namespace {

#include "unicode_tab.inc"

template <size_t N>
bool InTab(const uint32_t (&t)[N][2], uint32_t c) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    size_t m = (lo + hi) / 2;
    if (c > t[m][1]) lo = m + 1;
    else if (c < t[m][0]) hi = m;
    else return true;
  }
  return false;
}
bool IsSpace(uint32_t c) { return InTab(kSpace, c); }
bool IsDigit(uint32_t c) { return InTab(kDigit, c); }
bool IsUpper(uint32_t c) { return InTab(kUpper, c); }
bool IsAlpha(uint32_t c) { return InTab(kAlpha, c); }

// ---------------------------------------------------------------- UTF-8
std::u32string Utf8To32(const std::string& s) {
  std::u32string out;
  for (size_t i = 0; i < s.size();) {
    unsigned char c = s[i];
    uint32_t cp; int n;
    if (c < 0x80) { cp = c; n = 1; }
    else if ((c >> 5) == 6) { cp = c & 0x1f; n = 2; }
    else if ((c >> 4) == 14) { cp = c & 0x0f; n = 3; }
    else { cp = c & 0x07; n = 4; }
    for (int k = 1; k < n && i + k < s.size(); k++) cp = (cp << 6) | (s[i + k] & 0x3f);
    out.push_back(cp); i += n;
  }
  return out;
}
void Put32(std::string& o, uint32_t cp) {
  if (cp < 0x80) o += (char)cp;
  else if (cp < 0x800) { o += (char)(0xc0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3f)); }
  else if (cp < 0x10000) { o += (char)(0xe0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3f)); o += (char)(0x80 | (cp & 0x3f)); }
  else { o += (char)(0xf0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3f)); o += (char)(0x80 | ((cp >> 6) & 0x3f)); o += (char)(0x80 | (cp & 0x3f)); }
}
std::string Utf32To8(const std::u32string& s) { std::string o; for (auto c : s) Put32(o, c); return o; }
std::u32string PyStrip(const std::u32string& s) {
  size_t a = 0, b = s.size();
  while (a < b && IsSpace(s[a])) a++;
  while (b > a && IsSpace(s[b - 1])) b--;
  return s.substr(a, b - a);
}

// ---------------------------------------------------------------- minimal JSON
struct J {
  enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
  double num = 0; bool b = false; std::string str; std::vector<J> arr; std::map<std::string, J> obj;
  const J& operator[](const std::string& k) const {
    auto it = obj.find(k); if (it == obj.end()) throw std::runtime_error("meta.json: missing key " + k); return it->second;
  }
  bool has(const std::string& k) const { return obj.count(k) != 0; }
};
struct JP {
  const std::string& s; size_t i = 0;
  explicit JP(const std::string& x) : s(x) {}
  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) i++; }
  [[noreturn]] void bad() { throw std::runtime_error("meta.json: parse error at " + std::to_string(i)); }
  uint32_t hex4() { if (i + 4 > s.size()) bad(); uint32_t v = (uint32_t)std::stoul(s.substr(i, 4), nullptr, 16); i += 4; return v; }
  J parse() {
    ws(); if (i >= s.size()) bad();
    J j; char c = s[i];
    if (c == '{') {
      j.t = J::Obj; i++; ws();
      if (s[i] == '}') { i++; return j; }
      for (;;) { ws(); J k = parse(); if (k.t != J::Str) bad(); ws(); if (s[i++] != ':') bad();
        j.obj[k.str] = parse(); ws(); if (s[i] == ',') { i++; continue; } if (s[i++] != '}') bad(); return j; }
    }
    if (c == '[') {
      j.t = J::Arr; i++; ws();
      if (s[i] == ']') { i++; return j; }
      for (;;) { j.arr.push_back(parse()); ws(); if (s[i] == ',') { i++; continue; } if (s[i++] != ']') bad(); return j; }
    }
    if (c == '"') {
      j.t = J::Str; i++;
      while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\') {
          char e = s[++i]; i++;
          switch (e) {
            case 'n': j.str += '\n'; break; case 't': j.str += '\t'; break; case 'r': j.str += '\r'; break;
            case 'b': j.str += '\b'; break; case 'f': j.str += '\f'; break;
            case 'u': { uint32_t cp = hex4();
              if (cp >= 0xD800 && cp < 0xDC00 && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') {
                i += 2; uint32_t lo = hex4(); cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); }
              Put32(j.str, cp); break; }
            default: j.str += e;
          }
        } else j.str += s[i++];
      }
      i++; return j;
    }
    if (s.compare(i, 4, "true") == 0) { i += 4; j.t = J::Bool; j.b = true; return j; }
    if (s.compare(i, 5, "false") == 0) { i += 5; j.t = J::Bool; return j; }
    if (s.compare(i, 4, "null") == 0) { i += 4; return j; }
    size_t st = i; while (i < s.size() && (std::strchr("+-.eE", s[i]) || (s[i] >= '0' && s[i] <= '9'))) i++;
    if (st == i) bad();
    j.t = J::Num; j.num = std::strtod(s.substr(st, i - st).c_str(), nullptr); return j;
  }
};
J ParseJson(const std::string& s) { JP p(s); return p.parse(); }

// ---------------------------------------------------------------- numeric helpers (numpy / Python semantics)
// numpy float32 add.reduce of a contiguous run: identity 0, pairwise blocks of 128 with 8 accumulators
float PairSum(const float* a, size_t n) {
  if (n < 8) { float r = 0.f; for (size_t i = 0; i < n; i++) r += a[i]; return r; }
  if (n <= 128) {
    float r[8]; for (int j = 0; j < 8; j++) r[j] = a[j];
    size_t i = 8;
    for (; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += a[i + j];
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  }
  size_t n2 = n / 2; n2 -= n2 % 8;
  return PairSum(a, n2) + PairSum(a + n2, n - n2);
}
float SumF(const float* a, size_t n) { return 0.f + PairSum(a, n); }
float MeanF(const float* a, size_t n) { return SumF(a, n) / (float)n; }
float MedianF(std::vector<float> v) {  // np.median of float32
  size_t n = v.size(); std::sort(v.begin(), v.end());
  if (n % 2) return v[n / 2];
  return (v[n / 2 - 1] + v[n / 2]) / 2.f;
}
double PyRound(double x, int nd) {  // round(x, nd): correctly rounded, ties to even
  char buf[64]; std::snprintf(buf, sizeof buf, "%.*f", nd, x); return std::strtod(buf, nullptr);
}
long PyRoundInt(double x) { return (long)std::nearbyint(x); }

bool NumpySimdExpLog() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
  static const bool v = cpu::Avx512F() || cpu::Avx2Fma();
  return v;
#else
  return false;  // numpy calls libm expf / logf here
#endif
}
#if (defined(__x86_64__) || defined(__i386__)) && !(defined(_MSC_VER) && !defined(__clang__))
#define NP_FMA_TARGET __attribute__((target("fma")))
#else
#define NP_FMA_TARGET
#endif
// numpy simd_exp_FLOAT / simd_log_FLOAT (AVX2 / AVX512F), one lane
NP_FMA_TARGET float NpExpSimd(float x) {
  if (x != x) return x;
  if (x >= 88.72283935546875f) return std::numeric_limits<float>::infinity();
  if (x <= -103.97208404541015625f) return 0.f;
  float q = x * 1.442695040888963407359924681001892137F;
  q = q + 0x1.800000p+23f; q = q - 0x1.800000p+23f;
  float r = std::fma(q, -6.93145752e-1f, x);
  r = std::fma(q, -1.42860677e-6f, r);
  r = std::fma(q, 0.0f, r);
  float num = std::fma(5.082762527590693718096e-04f, r, 6.757896990527504603057e-03f);
  num = std::fma(num, r, 5.114512081637298353406e-02f);
  num = std::fma(num, r, 2.473615434895520810817e-01f);
  num = std::fma(num, r, 7.257664613233124478488e-01f);
  num = std::fma(num, r, 9.999999999980870924916e-01f);
  float den = std::fma(2.159509375685829852307e-02f, r, -2.742335390411667452936e-01f);
  den = std::fma(den, r, 1.0f);
  return std::ldexp(num / den, (int)q);
}
NP_FMA_TARGET float NpLogSimd(float x) {
  if (x != x) return x;
  if (x < 0.f) return -std::numeric_limits<float>::quiet_NaN();
  if (x == 0.f) return -std::numeric_limits<float>::infinity();
  if (std::isinf(x)) return x;
  int e; float m = std::frexp(x, &e);
  float ef = (float)e;
  if (m <= 0.707106781186547524400844362104849039F) { m = m + m; ef = ef - 1.f; }
  m = m - 1.f;
  float num = std::fma(2.589979117907922693523e-02f, m, 3.808837741388407920751e-01f);
  num = std::fma(num, m, 1.480000633576506585156e+00f);
  num = std::fma(num, m, 2.112677543073053063722e+00f);
  num = std::fma(num, m, 9.999999999999998702752e-01f);
  num = std::fma(num, m, 0.0f);
  float den = std::fma(5.875095403124574342950e-03f, m, 1.546476374983906719538e-01f);
  den = std::fma(den, m, 9.864942958519418960339e-01f);
  den = std::fma(den, m, 2.453006071784736363091e+00f);
  den = std::fma(den, m, 2.612677543073109236779e+00f);
  den = std::fma(den, m, 1.0f);
  return std::fma(ef, 0.693147180559945309417232121458176568F, num / den);
}
float NpExp(float x) { return NumpySimdExpLog() ? NpExpSimd(x) : fm::Exp(x); }
float NpLog(float x) { return NumpySimdExpLog() ? NpLogSimd(x) : fm::Log(x); }
float NpLogAddExp(float x, float y) {  // npy_logaddexpf (libm expf / log1pf)
  if (x == y) return x + 0.693147180559945309417232121458176568F;
  float t = x - y;
  if (t > 0) return x + fm::Log1p(fm::Exp(-t));
  if (t <= 0) return y + fm::Log1p(fm::Exp(t));
  return t;
}
// numpy.arange(start, stop, step) for float64
std::vector<double> Arange(double start, double stop, double step) {
  double lenf = std::ceil((stop - start) / step);
  long n = lenf > 0 ? (long)lenf : 0;
  std::vector<double> v((size_t)n);
  if (n > 0) v[0] = start;
  if (n > 1) { v[1] = start + step; double d = v[1] - v[0]; for (long i = 2; i < n; i++) v[(size_t)i] = start + (double)i * d; }
  return v;
}

// ---------------------------------------------------------------- vocab
struct Vocab {
  std::vector<std::string> chars;
  std::vector<std::u32string> chars32;
  bool has_commands = false;
  long space_id = -1;
  void Load(const std::string& js) {
    J d = ParseJson(js);
    if (d.has("blank") && d["blank"].num != 0) throw std::runtime_error("only blank=0 vocabularies are supported");
    for (auto& c : d["chars"].arr) {
      chars.push_back(c.str); chars32.push_back(Utf8To32(c.str));
      if (chars32.back().size() > 1) has_commands = true;
      if (c.str == " " && space_id < 0) space_id = (long)chars.size();
    }
  }
  std::string Decode(const std::vector<int64_t>& ids) const {
    std::vector<const std::u32string*> toks;
    for (auto i : ids) if (i > 0 && i <= (int64_t)chars.size()) toks.push_back(&chars32[(size_t)i - 1]);
    std::u32string out;
    for (size_t k = 0; k < toks.size(); k++) {
      const auto& t = *toks[k]; out += t;
      if (!has_commands) continue;
      bool alpha_tail = t.size() > 2 && t[0] == U'\\';
      for (size_t q = 1; alpha_tail && q < t.size(); q++) alpha_tail = IsAlpha(t[q]);
      if (alpha_tail && k + 1 < toks.size() && !toks[k + 1]->empty() && IsAlpha((*toks[k + 1])[0])) out += U' ';
    }
    return Utf32To8(out);
  }
};

// ---------------------------------------------------------------- data
struct Box5 { float v[5]; };
struct Tok { bool is_char; int64_t id; std::u32string ch; double x; float p = 0.f; double step = 0; };
struct RecOut { std::string text; double conf = 0; std::vector<Tok> toks; };

// Per-character positions of a line for return_word_box: tokens as the text
// shows them (" ".join(text.split()): leading / trailing spaces dropped,
// space runs collapsed), positions relative to [b0, b2]. Leaves `l` empty
// when the token count does not match the text's code points.
inline void LiteChars(LiteLine& l, const std::vector<Tok>& toks, double b0, double b2,
                      const std::u32string& text, bool (*is_space)(const Tok&, const void*), const void* ctx) {
  l.char_x.clear(); l.char_probs.clear(); l.step = 0;
  const double width = b2 - b0;
  if (width <= 0 || toks.empty()) return;
  std::vector<const Tok*> kept; bool pending_space = false;
  for (const auto& t : toks) {
    if (is_space(t, ctx)) { if (!kept.empty()) pending_space = true; continue; }
    if (pending_space) { kept.push_back(nullptr); pending_space = false; }
    kept.push_back(&t);
  }
  if (kept.size() != text.size()) return;
  double step = 0;
  for (const auto* t : kept) if (t) { step = t->step; break; }
  if (step <= 0) return;
  for (size_t k = 0; k < kept.size(); ++k) {
    const Tok* t = kept[k];
    if (!t) {  // a collapsed space: between its neighbours, as CTC emits it
      const Tok* a = kept[k - 1]; const Tok* b = kept[k + 1];
      l.char_x.push_back((float)(((a->x + b->x) / 2 - b0) / width));
      l.char_probs.push_back(std::min(a->p, b->p));
      continue;
    }
    l.char_x.push_back((float)((t->x - b0) / width));
    l.char_probs.push_back(t->p);
  }
  l.step = (float)(step / width);
}
struct Crop { int w = 0; std::vector<float> d; };  // [H][w] ink map

constexpr double kAdaptiveMinH = 9.0, kAdaptiveTargetH = 12.0, kSplitGap = 3.0, kGutterGap = 1.6;
constexpr int kGutterEdges = 3, kAdaptiveMaxSide = 2048;
constexpr double kMinConf = 0.5, kDeskewMax = 5.0, kDeskewMin = 0.3;
constexpr double kPadX = 0.3, kPadY = 0.2;

}  // namespace

struct LiteEngine::Impl {
  std::unique_ptr<DetNet> det;
  std::unique_ptr<RecNet> rec, rec_cjk;
  std::unique_ptr<ScriptNet> script;
  std::unique_ptr<OrderNet> order;
  Vocab vocab, vocab_cjk;
  int long_side = 1024, height = 32, threads = 1;
  bool box_round = true;
  mutable std::mutex mu;

  // ---- detection
  std::vector<Box5> DetectAt(const std::vector<uint8_t>& gray, int w, int h, int ls, double& scale) const {
    scale = (double)ls / (double)std::max(h, w);
    long nh = std::max(1L, PyRoundInt(h * scale)), nw = std::max(1L, PyRoundInt(w * scale));
    std::vector<uint8_t> g;
    const uint8_t* src = gray.data();
    if (nh != h || nw != w) {
      g.resize((size_t)nh * nw);
      rpocr_img_resize_u8(gray.data(), w, h, g.data(), (int)nw, (int)nh, scale < 1 ? 3 : 1, -1);
      src = g.data();
    }
    int ph = (int)std::ceil(nh / 32.0) * 32, pw = (int)std::ceil(nw / 32.0) * 32;
    std::vector<float> x((size_t)ph * pw, 0.f);
    for (long r = 0; r < nh; r++)
      for (long c = 0; c < nw; c++) x[(size_t)r * pw + c] = 1.0f - (float)src[r * nw + c] / 255.0f;
    std::vector<float> out; int C = 0, oh = 0, ow = 0;
    det->Run(x.data(), ph, pw, out, C, oh, ow);
    if (const char* d = std::getenv("PPOCR_LITE_DUMP")) {
      FILE* f = std::fopen((std::string(d) + "_detin.bin").c_str(), "wb"); std::fwrite(&ph, 4, 1, f); std::fwrite(&pw, 4, 1, f);
      std::fwrite(x.data(), 4, x.size(), f); std::fclose(f);
      f = std::fopen((std::string(d) + "_detout.bin").c_str(), "wb"); std::fwrite(&C, 4, 1, f); std::fwrite(&oh, 4, 1, f); std::fwrite(&ow, 4, 1, f);
      std::fwrite(out.data(), 4, out.size(), f); std::fclose(f);
    }
    return Decode(out.data(), oh, ow);
  }
  static std::vector<Box5> Decode(const float* out, int h, int w) {
    size_t n = (size_t)h * w;
    std::vector<float> prob(n); std::vector<uint8_t> mask(n);
    for (size_t i = 0; i < n; i++) { prob[i] = 1.0f / (1.0f + NpExp(-out[i])); mask[i] = prob[i] > 0.5f; }
    std::vector<int32_t> lab(n); std::vector<int32_t> st(5 * 1024);
    int nc = rpocr_img_cc4(mask.data(), w, h, lab.data(), st.data(), 1024);
    if (nc > 1024) { st.resize((size_t)5 * nc); nc = rpocr_img_cc4(mask.data(), w, h, lab.data(), st.data(), nc); }
    std::vector<Box5> boxes;
    if (nc <= 1) return boxes;
    std::vector<double> sy0(nc, 0), sy1(nc, 0), ssc(nc, 0); std::vector<long> cnt(nc, 0);
    const float* o1 = out + n; const float* o2 = out + 2 * n;
    for (int r = 0; r < h; r++) {
      float ys = ((float)r + 0.5f) * 2.0f;
      for (int c = 0; c < w; c++) {
        size_t i = (size_t)r * w + c; int l = lab[i];
        if (l <= 0) continue;
        sy0[l] += (double)(ys - o1[i]); sy1[l] += (double)(ys + o2[i]); ssc[l] += (double)prob[i]; cnt[l]++;
      }
    }
    for (int i = 1; i < nc; i++) {
      if (st[5 * i + 4] < 2) continue;
      double c = (double)std::max(1.0f, (float)cnt[i]);
      double y0 = sy0[i] / c, y1 = sy1[i] / c, sc = ssc[i] / c;
      double lh = std::max(1.0, y1 - y0);
      double ex = std::min(0.15 * lh, 0.5 * lh);
      double x0 = (double)(st[5 * i] * 2) - ex, x1 = (double)((st[5 * i] + st[5 * i + 2]) * 2) + ex;
      boxes.push_back({{(float)(x0 - 0.0 * lh), (float)(y0 - 0.0 * lh), (float)(x1 + 0.0 * lh), (float)(y1 + 0.0 * lh), (float)sc}});
    }
    return boxes;
  }
  std::vector<Box5> Detect(const std::vector<uint8_t>& gray, int w, int h) const {
    double s;
    auto boxes = DetectAt(gray, w, h, long_side, s);
    if (boxes.size() >= 8) {
      std::vector<float> hh; for (auto& b : boxes) hh.push_back(b.v[3] - b.v[1]);
      double med = (double)MedianF(hh);
      if (med < kAdaptiveMinH) {
        long ls = (long)std::min((double)kAdaptiveMaxSide, long_side * kAdaptiveTargetH / std::max(med, 1.0)) / 32 * 32;
        if (ls > long_side) boxes = DetectAt(gray, w, h, (int)ls, s);
      }
    }
    float fs = (float)s;
    for (auto& b : boxes) {
      for (int k = 0; k < 4; k++) b.v[k] = b.v[k] / fs;
      b.v[0] = std::min(std::max(b.v[0], 0.f), (float)w); b.v[2] = std::min(std::max(b.v[2], 0.f), (float)w);
      b.v[1] = std::min(std::max(b.v[1], 0.f), (float)h); b.v[3] = std::min(std::max(b.v[3], 0.f), (float)h);
    }
    return boxes;
  }

  // ---- recognition
  Crop CropLine(const std::vector<uint8_t>& gray, int W, int H, const float* box) const {
    double x0 = box[0], y0 = box[1], x1 = box[2], y1 = box[3];
    double h = std::max(1.0, y1 - y0);
    double X0 = x0 - kPadX * h, X1 = x1 + kPadX * h, Y0 = y0 - kPadY * h, Y1 = y1 + kPadY * h;
    long ix0 = (long)std::floor(X0), iy0 = (long)std::floor(Y0), ix1 = (long)std::ceil(X1), iy1 = (long)std::ceil(Y1);
    long cw = std::max(1L, ix1 - ix0), ch = std::max(1L, iy1 - iy0);
    std::vector<uint8_t> crop((size_t)cw * ch, 255);
    long sx0 = std::max(0L, ix0), sy0 = std::max(0L, iy0), sx1 = std::min((long)W, ix1), sy1 = std::min((long)H, iy1);
    if (sx1 > sx0 && sy1 > sy0)
      for (long r = sy0; r < sy1; r++)
        std::memcpy(&crop[(size_t)(r - iy0) * cw + (sx0 - ix0)], &gray[(size_t)r * W + sx0], (size_t)(sx1 - sx0));
    long ow = std::max(8L, std::min(4096L, PyRoundInt((double)cw * height / (double)ch)));
    std::vector<uint8_t> rs((size_t)ow * height);
    rpocr_img_resize_u8(crop.data(), (int)cw, (int)ch, rs.data(), (int)ow, height, ch > height ? 3 : 1, -1);
    Crop c; c.w = (int)ow; c.d.resize(rs.size());
    for (size_t i = 0; i < rs.size(); i++) c.d[i] = 1.0f - (float)rs[i] / 255.0f;
    return c;
  }
  static std::vector<std::vector<int>> WidthBatches(const std::vector<int>& widths, int max_batch = 64) {
    // PPOCR_LITE_BUCKET=ratio,max_batch,max_area (experiment; default 1.3,64,40960 = indra-ocr)
    static const std::array<double, 3> cfg = [] {
      std::array<double, 3> c{1.3, 64, 64 * 640};
      if (const char* e = std::getenv("PPOCR_LITE_BUCKET")) std::sscanf(e, "%lf,%lf,%lf", &c[0], &c[1], &c[2]);
      return c;
    }();
    if (max_batch == 64) max_batch = (int)cfg[1];
    std::vector<int> order(widths.size()); std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return widths[a] < widths[b]; });
    std::vector<std::vector<int>> out;
    for (int i : order) {
      if (!out.empty() && (int)out.back().size() < max_batch &&
          (double)std::max(widths[i], 64) <= cfg[0] * (double)std::max(widths[out.back()[0]], 64) &&
          (double)(out.back().size() + 1) * std::max(widths[i], 64) <= cfg[2])
        out.back().push_back(i);
      else out.push_back({i});
    }
    return out;
  }
  void PadBatch(const std::vector<Crop>& crops, const std::vector<int>& idx, std::vector<float>& arr,
                std::vector<int64_t>& widths, int& Wm) const {
    int mx = 0; for (int j : idx) mx = std::max(mx, crops[j].w);
    Wm = (int)std::ceil(mx / 8.0) * 8;
    arr.assign((size_t)idx.size() * height * Wm, 0.f); widths.resize(idx.size());
    for (size_t k = 0; k < idx.size(); k++) {
      const Crop& c = crops[idx[k]];
      for (int r = 0; r < height; r++) std::memcpy(&arr[(k * height + r) * Wm], &c.d[(size_t)r * c.w], sizeof(float) * c.w);
      widths[k] = c.w;
    }
  }
  std::vector<RecOut> Recognize(const std::vector<uint8_t>& gray, int W, int H, const std::vector<Box5>& boxes) const {
    std::vector<Crop> crops; crops.reserve(boxes.size());
    for (auto& b : boxes) crops.push_back(CropLine(gray, W, H, b.v));
    std::vector<RecOut> out(crops.size());
    std::vector<int> all(crops.size()); std::iota(all.begin(), all.end(), 0);
    struct G { const RecNet* net; const Vocab* voc; std::vector<int> sel; bool as_chars; };
    std::vector<G> groups{{rec.get(), &vocab, all, false}};
    if (script && !crops.empty()) {
      std::vector<char> hz(crops.size(), 0);
      std::vector<int> cw; for (auto& c : crops) cw.push_back(c.w);
      for (auto& idx : WidthBatches(cw)) {
        std::vector<float> arr; std::vector<int64_t> wd; int Wm;
        PadBatch(crops, idx, arr, wd, Wm);
        std::vector<float> p(idx.size());
        script->Run(arr.data(), (int)idx.size(), height, Wm, wd.data(), p.data());
        for (size_t k = 0; k < idx.size(); k++) hz[idx[k]] = p[k] > 0.5f;
      }
      G a{rec.get(), &vocab, {}, false}, b{rec_cjk.get(), &vocab_cjk, {}, true};
      for (int j : all) (hz[j] ? b.sel : a.sel).push_back(j);
      groups = {a, b};
    }
    static const bool one_submit = [] { const char* e = std::getenv("PPOCR_LITE_GPU_ONE_SUBMIT"); return !(e && e[0] == '0'); }();
    for (auto& g : groups) {
      std::vector<int> cw; for (int j : g.sel) cw.push_back(crops[j].w);
      auto parts = WidthBatches(cw);
      // Per part: line indices, padded width Wm (fixes T and the token stride exactly as the batched run).
      struct Part { std::vector<int> idx; int Wm = 0, T = 0; std::vector<int64_t> ids, len; std::vector<float> pm; };
      std::vector<Part> ps(parts.size());
      for (size_t q = 0; q < parts.size(); q++) {
        Part& P = ps[q]; int mx = 0;
        for (int i : parts[q]) { P.idx.push_back(g.sel[i]); mx = std::max(mx, crops[g.sel[i]].w); }
        P.Wm = (int)std::ceil(mx / 8.0) * 8; P.T = g.net->steps(P.Wm);
        P.ids.assign(P.idx.size() * (size_t)P.T, 0); P.pm.assign(P.idx.size() * (size_t)P.T, 0.f); P.len.assign(P.idx.size(), 0);
      }
      if (one_submit && g.net->ragged_gpu()) {  // item 2: one GPU submission per recognizer per page
        std::vector<const float*> xs; std::vector<int> ws, Ls;
        for (auto& P : ps)
          for (size_t k = 0; k < P.idx.size(); k++) {
            const Crop& c = crops[P.idx[k]];
            int sw = g.net->stride(), L = std::min(P.T, (c.w + sw - 1) / sw);
            xs.push_back(c.d.data()); ws.push_back(c.w); Ls.push_back(L); P.len[k] = L;
          }
        size_t tot = 0; for (int L : Ls) tot += (size_t)L;
        std::vector<int64_t> gi(tot); std::vector<float> gc(tot);
        g.net->RunRagged(xs, ws, Ls, height, gi.data(), gc.data());
        size_t o = 0;
        for (auto& P : ps)
          for (size_t k = 0; k < P.idx.size(); k++)
            for (int64_t l = 0; l < P.len[k]; l++, o++) { P.ids[k * P.T + l] = gi[o]; P.pm[k * P.T + l] = gc[o]; }
      } else {
        for (auto& P : ps) {
          std::vector<float> arr; std::vector<int64_t> wd; int Wm;
          PadBatch(crops, P.idx, arr, wd, Wm);
          g.net->Run(arr.data(), (int)P.idx.size(), height, Wm, wd.data(), P.ids.data(), P.pm.data(), P.len.data());
        }
      }
      for (auto& P : ps) {
        const std::vector<int>& idx = P.idx; int B = (int)idx.size(), T = P.T, Wm = P.Wm;
        double stride = (double)Wm / (double)std::max(T, 1);
        for (int k = 0; k < B; k++) {
          int j = idx[k]; long n = (long)P.len[k];
          const int64_t* seq = &P.ids[(size_t)k * T]; const float* p = &P.pm[(size_t)k * T];
          double x0 = boxes[j].v[0], y0 = boxes[j].v[1], x1 = boxes[j].v[2], y1 = boxes[j].v[3];
          double h = std::max(1.0, y1 - y0);
          double ix0 = std::floor(x0 - kPadX * h);
          double cwd = std::ceil(x1 + kPadX * h) - ix0;
          double f = cwd / (double)crops[j].w;
          std::vector<int64_t> kept; std::vector<float> kp; std::vector<Tok> toks;
          for (long t = 0; t < n; t++) {
            if (seq[t] == 0 || (t > 0 && seq[t] == seq[t - 1])) continue;
            kept.push_back(seq[t]); kp.push_back(p[t]);
            Tok tk; tk.x = ix0 + ((double)t + 0.5) * stride * f; tk.p = p[t]; tk.step = stride * f;
            if (g.as_chars) { tk.is_char = true; tk.id = 0; tk.ch = g.voc->chars32[(size_t)seq[t] - 1]; }
            else { tk.is_char = false; tk.id = seq[t]; }
            toks.push_back(std::move(tk));
          }
          double conf = kept.empty() ? 0.0 : (double)MeanF(kp.data(), kp.size());
          if (conf >= kMinConf) { out[j].text = g.voc->Decode(kept); out[j].conf = conf; out[j].toks = std::move(toks); }
          else { out[j].text.clear(); out[j].conf = conf; out[j].toks.clear(); }
        }
      }
    }
    return out;
  }

  // ---- line splitting
  static bool SpaceTok(const Tok& t, const void* self) { return static_cast<const Impl*>(self)->IsSpaceTok(t); }
  bool IsSpaceTok(const Tok& t) const {
    return t.is_char ? (t.ch.size() == 1 && t.ch[0] == U' ') : (t.id == vocab.space_id);
  }
  std::string DecodeToks(const std::vector<Tok>& toks) const {
    if (!toks.empty() && toks[0].is_char) {  // " ".join("".join(chars).split())
      std::u32string all; for (auto& t : toks) all += t.ch;
      std::u32string out, word; bool first = true;
      auto flush = [&] { if (!word.empty()) { if (!first) out += U' '; out += word; first = false; word.clear(); } };
      for (char32_t c : all) { if (IsSpace(c)) flush(); else word += c; }
      flush();
      return Utf32To8(out);
    }
    std::vector<int64_t> ids; for (auto& t : toks) ids.push_back(t.id);
    return vocab.Decode(ids);
  }
  std::vector<std::vector<double>> GutterCuts(const std::vector<Box5>& boxes, const std::vector<RecOut>& rec) const {
    std::vector<std::vector<double>> cuts;
    for (size_t i = 0; i < boxes.size(); i++) {
      double h = std::max(1.0, (double)(boxes[i].v[3] - boxes[i].v[1]));
      std::vector<double> xs; for (auto& t : rec[i].toks) if (!IsSpaceTok(t)) xs.push_back(t.x);
      std::vector<double> mine;
      for (size_t k = 0; k + 1 < xs.size(); k++) {
        double a = xs[k], c = xs[k + 1];
        if (c - a <= kGutterGap * h) continue;
        float gl = (float)(a + 0.35 * h), gr = (float)(c - 0.35 * h), lim = (float)(0.5 * h);
        int cnt = 0;
        for (size_t j = 0; j < boxes.size(); j++) {
          if (j == i) continue;
          if (std::fabs(boxes[j].v[2] - gl) < lim || std::fabs(boxes[j].v[0] - gr) < lim) cnt++;
        }
        if (cnt >= kGutterEdges) mine.push_back((a + c) / 2);
      }
      cuts.push_back(mine);
    }
    return cuts;
  }
  struct Seg { float b[4]; std::string text; std::vector<Tok> toks; };
  std::vector<Seg> SplitLine(const float* box, const std::vector<Tok>& toks, double gap, std::vector<double> at) const {
    double h = std::max(1.0, (double)(box[3] - box[1]));
    std::sort(at.begin(), at.end());
    std::vector<std::vector<Tok>> segs(1);
    bool have_prev = false; double prev = 0;
    for (auto& tk : toks) {
      double x = tk.x;
      if (!IsSpaceTok(tk)) {
        bool cut = false;
        if (have_prev) {
          cut = x - prev > gap * h;
          for (double c : at) if (prev < c && c < x) cut = true;
        }
        if (cut) segs.emplace_back();
        prev = x; have_prev = true;
      }
      segs.back().push_back(tk);
    }
    std::vector<Seg> out;
    if (segs.size() == 1) { Seg s; std::memcpy(s.b, box, sizeof s.b); s.text = DecodeToks(toks); s.toks = toks; out.push_back(std::move(s)); return out; }
    for (auto& sg : segs) {
      std::string text = Utf32To8(PyStrip(Utf8To32(DecodeToks(sg))));
      std::vector<double> xs; for (auto& t : sg) if (!IsSpaceTok(t)) xs.push_back(t.x);
      if (text.empty() || xs.empty()) continue;
      double mn = *std::min_element(xs.begin(), xs.end()), mx = *std::max_element(xs.begin(), xs.end());
      float l = (float)(mn - 0.35 * h), r = (float)(mx + 0.35 * h);
      Seg s; s.b[0] = l > box[0] ? l : box[0]; s.b[1] = box[1]; s.b[2] = r < box[2] ? r : box[2]; s.b[3] = box[3];
      s.text = std::move(text); s.toks = sg; out.push_back(std::move(s));
    }
    return out;
  }

  // ---- reading order
  static std::vector<int> TopDown(const std::vector<std::array<float, 4>>& b, const std::vector<int>& ids) {
    if (ids.empty()) return {};
    std::vector<float> hh; for (int i : ids) hh.push_back(b[i][3] - b[i][1]);
    float hm = MedianF(hh) * 0.5f;
    float den = (0.001f > hm) ? 0.001f : hm;
    std::vector<long> key(ids.size());
    for (size_t k = 0; k < ids.size(); k++) key[k] = PyRoundInt((double)(((b[ids[k]][1] + b[ids[k]][3]) / 2.f) / den));
    std::vector<int> o(ids.size()); std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&](int x, int y) {
      if (key[x] != key[y]) return key[x] < key[y];
      return b[ids[x]][0] < b[ids[y]][0];
    });
    std::vector<int> out; for (int k : o) out.push_back(ids[k]);
    return out;
  }
  static void BestCut(const std::vector<std::array<float, 4>>& b, const std::vector<int>& ids, int axis, float& gap,
                      int& kk, std::vector<int>& o) {
    int lo_i = axis == 1 ? 1 : 0, hi_i = axis == 1 ? 3 : 2;
    o.resize(ids.size()); std::iota(o.begin(), o.end(), 0);
    std::stable_sort(o.begin(), o.end(), [&](int x, int y) { return b[ids[x]][lo_i] < b[ids[y]][lo_i]; });
    float end = b[ids[o[0]]][hi_i]; gap = 0.f; kk = -1;
    for (size_t k = 1; k < o.size(); k++) {
      float g = b[ids[o[k]]][lo_i] - end;
      if (g > gap) { gap = g; kk = (int)k; }
      float hv = b[ids[o[k]]][hi_i];
      if (hv > end) end = hv;
    }
  }
  static std::vector<int> XyCut(const std::vector<std::array<float, 4>>& b, const std::vector<int>& ids) {
    if (ids.size() <= 1) return ids;
    float gy, gx; int ky, kx; std::vector<int> oy, ox;
    BestCut(b, ids, 1, gy, ky, oy); BestCut(b, ids, 0, gx, kx, ox);
    if (ky < 0 && kx < 0) return TopDown(b, ids);
    int k = gy >= gx ? ky : kx; const auto& o = gy >= gx ? oy : ox;
    std::vector<int> A, B;
    for (int q = 0; q < (int)o.size(); q++) (q < k ? A : B).push_back(ids[o[q]]);
    auto ra = XyCut(b, A), rb = XyCut(b, B);
    ra.insert(ra.end(), rb.begin(), rb.end());
    return ra;
  }
  // Tie-breaking in order decoding is explicit and platform independent: equal sort keys keep line index order
  // (stable sort), equal insertion gains take the first position, equal merge scores the lowest index.
  static std::vector<int> DecodeOrder(const std::vector<float>& a, int n, int sweeps = 4) {
    std::vector<int> order(n); std::iota(order.begin(), order.end(), 0);
    if (n <= 1) return order;
    std::vector<float> lp((size_t)n * n);
    for (size_t i = 0; i < lp.size(); i++) lp[i] = -NpLogAddExp(0.0f, -a[i]);
    std::vector<float> key(n), col(n, 0.f);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) col[j] = col[j] + lp[(size_t)i * n + j];
    for (int i = 0; i < n; i++) key[i] = -SumF(&lp[(size_t)i * n], n) + col[i];
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return key[x] < key[y]; });
    for (int sw = 0; sw < sweeps; sw++) {
      bool improved = false;
      std::vector<int> snap = order;
      for (int x : snap) {
        std::vector<int> rest; rest.reserve(n - 1);
        int cur = 0;
        for (int q = 0; q < n; q++) { if (order[q] == x) cur = q; else rest.push_back(order[q]); }
        int m = n - 1;
        std::vector<double> gains(m + 1);
        std::vector<float> cb(m), ca(m);
        float acc = 0.f;
        for (int q = 0; q < m; q++) { acc = (q == 0) ? lp[(size_t)rest[q] * n + x] : acc + lp[(size_t)rest[q] * n + x]; cb[q] = acc; }
        acc = 0.f;
        for (int q = m - 1; q >= 0; q--) { acc = (q == m - 1) ? lp[(size_t)x * n + rest[q]] : acc + lp[(size_t)x * n + rest[q]]; ca[q] = acc; }
        for (int q = 0; q <= m; q++) gains[q] = (q == 0 ? 0.0 : (double)cb[q - 1]) + (q < m ? (double)ca[q] : 0.0);
        int k = (int)(std::max_element(gains.begin(), gains.end()) - gains.begin());
        if (gains[k] > gains[cur] + 1e-6) {
          rest.insert(rest.begin() + k, x); order = rest; improved = true;
        }
      }
      if (!improved) break;
    }
    return order;
  }
  static std::vector<std::vector<int>> Cluster(const std::vector<float>& logits, int n) {
    const double NINF = -std::numeric_limits<double>::infinity();
    std::vector<double> S((size_t)n * n), M((size_t)n * n);
    for (size_t i = 0; i < S.size(); i++) S[i] = logits[i];
    for (int i = 0; i < n; i++) S[(size_t)i * n + i] = 0.0;
    std::vector<double> size(n, 1.0); std::vector<char> alive(n, 1);
    std::vector<std::vector<int>> members(n); for (int i = 0; i < n; i++) members[i] = {i};
    M = S; for (int i = 0; i < n; i++) M[(size_t)i * n + i] = NINF;
    auto argmax = [&](const double* r) { int b = 0; for (int j = 1; j < n; j++) if (r[j] > r[b]) b = j; return b; };
    std::vector<int> best(n); std::vector<double> bv(n);
    for (int i = 0; i < n; i++) { best[i] = argmax(&M[(size_t)i * n]); bv[i] = M[(size_t)i * n + best[i]]; }
    for (int it = 0; it < n - 1; it++) {
      int a = argmax(bv.data()), b = best[a];
      if (bv[a] <= 0) break;
      if (a > b) std::swap(a, b);
      for (int j = 0; j < n; j++) S[(size_t)a * n + j] += S[(size_t)b * n + j];
      for (int j = 0; j < n; j++) S[(size_t)j * n + a] += S[(size_t)j * n + b];
      S[(size_t)a * n + a] = 0.0;
      size[a] += size[b]; alive[b] = 0;
      members[a].insert(members[a].end(), members[b].begin(), members[b].end());
      std::vector<double> row(n);
      for (int j = 0; j < n; j++) row[j] = S[(size_t)a * n + j] / (size[a] * size[j]);
      for (int j = 0; j < n; j++) if (!alive[j]) row[j] = NINF;
      row[a] = NINF;
      for (int j = 0; j < n; j++) { M[(size_t)a * n + j] = row[j]; M[(size_t)j * n + a] = row[j]; }
      for (int j = 0; j < n; j++) { M[(size_t)b * n + j] = NINF; M[(size_t)j * n + b] = NINF; }
      bv[b] = NINF;
      best[a] = argmax(row.data()); bv[a] = row[best[a]];
      for (int j = 0; j < n; j++) {
        if (!alive[j] || j == a || !(best[j] == a || best[j] == b)) continue;
        best[j] = argmax(&M[(size_t)j * n]); bv[j] = M[(size_t)j * n + best[j]];
      }
      for (int j = 0; j < n; j++) if (alive[j] && j != a && row[j] > bv[j]) { best[j] = a; bv[j] = row[j]; }
    }
    std::vector<std::vector<int>> out;
    for (int i = 0; i < n; i++) if (alive[i]) { auto m = members[i]; std::sort(m.begin(), m.end()); out.push_back(m); }
    return out;
  }
  std::vector<int> DecodeBlocksOrder(const std::vector<float>& a, const std::vector<float>& same,
                                     const std::vector<std::array<float, 4>>& b, const std::vector<float>& cls, int ncls,
                                     int n) const {
    std::vector<int> id(n); std::iota(id.begin(), id.end(), 0);
    if (n <= 1) return id;
    auto blocks = Cluster(same, n);
    std::vector<std::vector<int>> inner;
    for (auto& bl : blocks) {
      std::vector<float> s(ncls, 0.f);
      for (int i : bl) for (int c = 0; c < ncls; c++) s[c] = s[c] + cls[(size_t)i * ncls + c];
      int am = 0; for (int c = 1; c < ncls; c++) if (s[c] > s[am]) am = c;
      bool grid = ncls > 0 && (am == 5 || am == 8 || am == 9);  // Table, Table-Of-Contents, Form
      inner.push_back(grid ? TopDown(b, bl) : XyCut(b, bl));
    }
    int m = (int)blocks.size();
    std::vector<double> T((size_t)m * n, 0.0);
    for (int p = 0; p < m; p++) {
      double g = 1.0 / (double)blocks[p].size();
      for (int i : blocks[p]) for (int j = 0; j < n; j++) T[(size_t)p * n + j] += g * (double)a[(size_t)i * n + j];
    }
    std::vector<double> up((size_t)m * m, 0.0);
    for (int p = 0; p < m; p++)
      for (int q = p + 1; q < m; q++) {
        double g = 1.0 / (double)blocks[q].size(), s = 0.0;
        for (int j : blocks[q]) s += T[(size_t)p * n + j] * g;
        up[(size_t)p * m + q] = s;
      }
    std::vector<float> ab((size_t)m * m);
    for (int p = 0; p < m; p++) for (int q = 0; q < m; q++) ab[(size_t)p * m + q] = (float)(up[(size_t)p * m + q] - up[(size_t)q * m + p]);
    std::vector<int> out;
    for (int k : DecodeOrder(ab, m)) out.insert(out.end(), inner[k].begin(), inner[k].end());
    return out;
  }
  std::vector<float> LineFeatures(const std::vector<std::array<float, 4>>& b, int W, int H,
                                  const std::vector<std::string>& texts) const {
    int n = (int)b.size(); const int F = 16;
    std::vector<float> f((size_t)n * F, 0.f);
    std::vector<float> wv(n), hv(n);
    for (int i = 0; i < n; i++) { wv[i] = std::max(b[i][2] - b[i][0], 0.001f); hv[i] = std::max(b[i][3] - b[i][1], 0.001f); }
    float med = MedianF(hv);
    float fW = (float)W, fH = (float)H, f2W = (float)(2 * W), f2H = (float)(2 * H);
    float col9 = (float)((double)med / (double)H * 20.0);
    for (int i = 0; i < n; i++) {
      float* r = &f[(size_t)i * F];
      r[0] = b[i][0] / fW; r[1] = b[i][1] / fH; r[2] = b[i][2] / fW; r[3] = b[i][3] / fH;
      r[4] = (b[i][0] + b[i][2]) / f2W; r[5] = (b[i][1] + b[i][3]) / f2H; r[6] = wv[i] / fW;
      r[7] = NpLog(hv[i] / med); r[8] = NpLog(wv[i] / hv[i]) / 4.0f; r[9] = col9;
      std::u32string t = PyStrip(Utf8To32(texts[i]));
      if (t.empty()) continue;
      double L = (double)t.size();
      r[10] = (float)(std::log1p(L) / 5.0);
      int nd = 0; for (auto c : t) nd += IsDigit(c);
      r[11] = (float)((double)nd / L);
      static const std::u32string bullets = U"\u2022-\u2013\u25aa\u25e6*\u00b7\u2192";
      bool lead = bullets.find(t[0]) != std::u32string::npos;
      if (!lead && t.size() > 3) {
        std::u32string p = t.substr(0, 3);
        while (!p.empty() && (p.back() == U'.' || p.back() == U')')) p.pop_back();
        bool all = !p.empty(); for (auto c : p) all = all && IsDigit(c);
        lead = all;
      }
      r[12] = lead ? 1.f : 0.f;
      r[13] = t.back() == U'-' ? 1.f : 0.f;
      r[14] = std::u32string(U".!?:;").find(t.back()) != std::u32string::npos ? 1.f : 0.f;
      r[15] = IsUpper(t[0]) ? 1.f : 0.f;
    }
    return f;
  }
  std::vector<int> ReadingOrder(std::vector<std::array<float, 4>> b, const std::vector<std::string>& texts, int W, int H) const {
    if (box_round) for (auto& x : b) for (auto& v : x) v = std::nearbyint(v);
    int n = (int)b.size();
    std::vector<int> ids(n); std::iota(ids.begin(), ids.end(), 0);
    if (n < 2 || !order) return n ? XyCut(b, ids) : ids;
    auto feats = LineFeatures(b, W, H, texts);
    std::vector<uint8_t> mask(n, 1);
    int nc = order->ncls();
    std::vector<float> a((size_t)n * n), same((size_t)n * n), cls((size_t)n * std::max(nc, 1));
    order->Run(feats.data(), mask.data(), n, a.data(), same.data(), cls.data(), nullptr);
    if (nc <= 0) return DecodeOrder(a, n);
    return DecodeBlocksOrder(a, same, b, cls, nc, n);
  }

  // ---- deskew
  double EstimateSkew(const std::vector<uint8_t>& gray, int w, int h) const {
    double s = 800.0 / (double)std::max(h, w);
    int sw = (int)std::max(1L, PyRoundInt(w * s)), sh = (int)std::max(1L, PyRoundInt(h * s));
    std::vector<uint8_t> small((size_t)sw * sh);
    rpocr_img_resize_u8(gray.data(), w, h, small.data(), sw, sh, 3, -1);
    std::vector<float> ink(small.size());
    for (size_t i = 0; i < ink.size(); i++) ink[i] = (255.0f - (float)small[i]) / 255.0f;
    if (MeanF(ink.data(), ink.size()) < 0.003f) return 0.0;
    double cx = sw / 2.0, cy = sh / 2.0;
    // Sharpness of each candidate angle is independent: evaluate them in parallel (one warp per thread,
    // image ops single-threaded inside), then pick the first maximum in order as before.
    auto sharp = [&](double a, std::vector<float>& r, std::vector<float>& p, std::vector<float>& d2) {
      double M[6]; rpocr_img_rotation_matrix(cx, cy, a, 1.0, -1, M);
      rpocr_img_warp_affine(ink.data(), sw, sh, r.data(), sw, sh, M, 1, 0, 0.0, 0, -1);
      for (int y = 0; y < sh; y++) p[y] = SumF(&r[(size_t)y * sw], sw);
      for (int y = 0; y + 1 < sh; y++) { float d = p[y + 1] - p[y]; d2[y] = d * d; }
      return (double)SumF(d2.data(), d2.size());
    };
    auto argbest = [&](const std::vector<double>& v) {
      std::vector<double> q(v.size());
      int nt = std::max(1, std::min((int)v.size(), rpocr_img_get_threads()));
      std::atomic<size_t> next{0};
      auto work = [&] {
        if (nt > 1) rpocr_img_set_local_threads(1);
        std::vector<float> r(ink.size()), p(sh), d2(sh > 0 ? sh - 1 : 0);
        for (size_t i; (i = next++) < v.size();) q[i] = sharp(v[i], r, p, d2);
        if (nt > 1) rpocr_img_set_local_threads(0);
      };
      std::vector<std::thread> th;
      for (int t = 1; t < nt; t++) th.emplace_back(work);
      work();
      for (auto& t : th) t.join();
      double bestv = 0, besta = v[0]; bool first = true;
      for (size_t i = 0; i < v.size(); i++) if (first || q[i] > bestv) { bestv = q[i]; besta = v[i]; first = false; }
      return besta;
    };
    double best = argbest(Arange(-kDeskewMax, kDeskewMax + 1e-6, 0.5));
    return argbest(Arange(best - 0.5, best + 0.5 + 1e-6, 0.1));
  }

  // ---- page
  // PPOCR_LITE_PROF=1: per-stage wall time totals, printed to stderr when the engine is destroyed.
  mutable double prof[9] = {};
  bool prof_on = std::getenv("PPOCR_LITE_PROF") != nullptr;
  static double NowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
  // Page = Front (deskew + detection) then Back (recognition, splitting, order). Batches overlap Front of page i+1
  // with Back of page i (DetNet and the recognizers are only ever used by one stage at a time).
  struct Front { std::vector<uint8_t> gray; int w = 0, h = 0; double skew = 0; std::vector<Box5> boxes; };
  std::vector<LiteLine> Page(std::vector<uint8_t> gray, int w, int h) const { return Back(MakeFront(std::move(gray), w, h)); }
  Front MakeFront(std::vector<uint8_t> gray, int w, int h) const {
    double t0 = NowMs();
    double skew = EstimateSkew(gray, w, h);
    double t1 = NowMs();
    if (std::fabs(skew) >= kDeskewMin) {
      double M[6]; rpocr_img_rotation_matrix(w / 2.0, h / 2.0, skew, 1.0, -1, M);
      std::vector<uint8_t> rot(gray.size());
      rpocr_img_warp_affine(gray.data(), w, h, rot.data(), w, h, M, 0, 1, 0.0, 0, -1);
      gray.swap(rot);
    } else skew = 0.0;
    double t2 = NowMs();
    auto boxes = Detect(gray, w, h);
    double t3 = NowMs();
    if (prof_on) { prof[0] += t1 - t0; prof[1] += t2 - t1; prof[2] += t3 - t2; }
    Front f; f.gray = std::move(gray); f.w = w; f.h = h; f.skew = skew; f.boxes = std::move(boxes);
    return f;
  }
  std::vector<LiteLine> Back(Front fr) const {
    const std::vector<uint8_t>& gray = fr.gray; const int w = fr.w, h = fr.h; const double skew = fr.skew;
    const std::vector<Box5>& boxes = fr.boxes;
    double t3 = NowMs();
    auto rec = Recognize(gray, w, h, boxes);
    double t4 = NowMs();
    std::vector<std::array<float, 4>> nb; std::vector<std::string> texts; std::vector<double> confs;
    std::vector<std::vector<Tok>> seg_toks;
    auto cuts = boxes.empty() ? std::vector<std::vector<double>>() : GutterCuts(boxes, rec);
    for (size_t i = 0; i < boxes.size(); i++)
      for (auto& sg : SplitLine(boxes[i].v, rec[i].toks, kSplitGap, cuts[i])) {
        nb.push_back({sg.b[0], sg.b[1], sg.b[2], sg.b[3]}); texts.push_back(sg.text); confs.push_back(rec[i].conf);
        seg_toks.push_back(std::move(sg.toks));
      }
    double t5 = NowMs();
    auto order_ = ReadingOrder(nb, texts, w, h);
    double t6 = NowMs();
    if (prof_on) { prof[3] += t4 - t3; prof[4] += t5 - t4; prof[5] += t6 - t5; }
    std::vector<LiteLine> out;
    double Mi[6];
    if (skew) rpocr_img_rotation_matrix(w / 2.0, h / 2.0, -skew, 1.0, -1, Mi);
    for (int i : order_) {
      if (texts[i].empty()) continue;
      double v[4]; for (int k = 0; k < 4; k++) v[k] = PyRound((double)nb[i][k], 1);
      if (skew) {
        double xs[4] = {v[0], v[2], v[0], v[2]}, ys[4] = {v[1], v[1], v[3], v[3]};
        double mnx = 1e300, mny = 1e300, mxx = -1e300, mxy = -1e300;
        for (int k = 0; k < 4; k++) {
          double cxk = Mi[0] * xs[k] + Mi[1] * ys[k] + Mi[2] * 1.0, cyk = Mi[3] * xs[k] + Mi[4] * ys[k] + Mi[5] * 1.0;
          mnx = std::min(mnx, cxk); mxx = std::max(mxx, cxk); mny = std::min(mny, cyk); mxy = std::max(mxy, cyk);
        }
        auto clip = [](double x, double hi) { return std::min(std::max(x, 0.0), hi); };
        v[0] = PyRound(clip(mnx, w), 1); v[1] = PyRound(clip(mny, h), 1); v[2] = PyRound(clip(mxx, w), 1); v[3] = PyRound(clip(mxy, h), 1);
      }
      LiteLine l; l.x0 = (float)v[0]; l.y0 = (float)v[1]; l.x1 = (float)v[2]; l.y1 = (float)v[3];
      l.text = texts[i]; l.conf = (float)PyRound(confs[i], 4);
      // Fractions of the (deskewed) segment width carry over to the output box.
      LiteChars(l, seg_toks[i], nb[i][0], nb[i][2], Utf8To32(l.text), &Impl::SpaceTok, this);
      out.push_back(std::move(l));
    }
    return out;
  }
};

bool LiteEngine::IsLiteDir(const std::string& path) {
  struct stat st{};
  if (path.empty() || stat(path.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return false;
  return stat((path + "/meta.json").c_str(), &st) == 0;
}

LiteEngine::LiteEngine(const std::string& dir, int threads) : p_(std::make_unique<Impl>()) {
  auto exists = [&](const char* n) { struct stat st{}; return stat((dir + "/" + n).c_str(), &st) == 0; };
  auto need = [&](const char* n) {
    if (!exists(n)) throw std::runtime_error("cannot open lite model file " + dir + "/" + n);
    return dir + "/" + n;
  };
  for (const char* n : {"rec.int8.onnx", "mask.onnx", "orient.onnx"})
    if (exists(n)) throw std::runtime_error(dir + "/" + n + ": not supported by the native lite pipeline");
  int t = std::max(1, threads);
  p_->threads = t;
  rpocr_img_set_threads(t);
  std::ifstream f(need("meta.json"), std::ios::binary);
  std::stringstream ss; ss << f.rdbuf();
  J meta = ParseJson(ss.str());
  p_->det = std::make_unique<DetNet>(need("det.onnx"), t);
  p_->rec = std::make_unique<RecNet>(need("rec.onnx"), t);
  if (exists("order.onnx")) p_->order = std::make_unique<OrderNet>(dir + "/order.onnx", t);
  p_->vocab.Load(meta["vocab"].str);
  if (exists("script.onnx") && exists("rec_cjk.onnx")) {
    p_->script = std::make_unique<ScriptNet>(dir + "/script.onnx", t);
    p_->rec_cjk = std::make_unique<RecNet>(dir + "/rec_cjk.onnx", t);
    p_->vocab_cjk.Load(meta["vocab_cjk"].str);
  }
  p_->long_side = (int)meta["det_long_side"].num;
  p_->height = (int)meta["rec_height"].num;
  if (const char* e = std::getenv("PPOCR_INDRA_BOX_ROUND")) p_->box_round = std::strcmp(e, "0") != 0;
}
LiteEngine::~LiteEngine() {
  if (p_ && p_->prof_on)
    std::fprintf(stderr, "lite_prof_ms skew=%.1f rotate=%.1f det=%.1f rec=%.1f split=%.1f order=%.1f gray=%.1f\n", p_->prof[0], p_->prof[1],
                 p_->prof[2], p_->prof[3], p_->prof[4], p_->prof[5], p_->prof[6]);
}

std::string LiteEngine::Note() const {
  return std::string("bundled Indra OCR models (model_type=lite): native C++ pipeline, rec[") + p_->rec->isa() + "]";
}

std::vector<LiteLine> LiteEngine::Run(const std::uint8_t* rgb, int w, int h, int tasks) const {
  if (!rgb || w <= 0 || h <= 0) throw std::invalid_argument("empty image");
  std::lock_guard<std::mutex> lock(p_->mu);
  rpocr_img_set_threads(p_->threads);
  std::vector<uint8_t> g((size_t)w * h);
  double tg = p_->prof_on ? Impl::NowMs() : 0;
  rpocr_img_rgb2gray(rgb, w, h, w * 3, g.data());
  if (p_->prof_on) p_->prof[6] += Impl::NowMs() - tg;
  std::vector<LiteLine> out;
  if (tasks == 1) {
    for (auto& b : p_->Detect(g, w, h)) { LiteLine l; l.x0 = b.v[0]; l.y0 = b.v[1]; l.x1 = b.v[2]; l.y1 = b.v[3]; l.conf = b.v[4]; out.push_back(l); }
    return out;
  }
  if (tasks == 2) {
    int p = std::max(16, h / 2), W = w + 2 * p, H = h + 2 * p;
    std::vector<uint8_t> gp((size_t)W * H, 255);
    for (int r = 0; r < h; r++) std::memcpy(&gp[(size_t)(r + p) * W + p], &g[(size_t)r * w], (size_t)w);
    Box5 b{{(float)p, (float)p, (float)(p + w), (float)(p + h), 0.f}};
    auto r = p_->Recognize(gp, W, H, {b});
    LiteLine l; l.x0 = 0; l.y0 = 0; l.x1 = (float)w; l.y1 = (float)h; l.text = r[0].text; l.conf = (float)r[0].conf;
    LiteChars(l, r[0].toks, (double)p, (double)(p + w), Utf8To32(l.text), &Impl::SpaceTok, p_.get());
    out.push_back(l);
    return out;
  }
  return p_->Page(std::move(g), w, h);
}

std::vector<std::vector<LiteLine>> LiteEngine::RunBatch(const std::vector<BatchImage>& images) const {
  for (auto& im : images) if (!im.rgb || im.w <= 0 || im.h <= 0) throw std::invalid_argument("empty image");
  std::lock_guard<std::mutex> lock(p_->mu);
  rpocr_img_set_threads(p_->threads);
  std::vector<std::vector<LiteLine>> out(images.size());
  auto front = [&](size_t i) {
    std::vector<uint8_t> g((size_t)images[i].w * images[i].h);
    rpocr_img_rgb2gray(images[i].rgb, images[i].w, images[i].h, images[i].w * 3, g.data());
    return p_->MakeFront(std::move(g), images[i].w, images[i].h);
  };
  const char* e = std::getenv("PPOCR_LITE_PIPELINE");
  if (images.size() < 2 || (e && e[0] == '0')) {
    for (size_t i = 0; i < images.size(); i++) out[i] = p_->Back(front(i));
    return out;
  }
  std::future<Impl::Front> next = std::async(std::launch::async, front, 0);
  for (size_t i = 0; i < images.size(); i++) {
    Impl::Front cur = next.get();
    if (i + 1 < images.size()) next = std::async(std::launch::async, front, i + 1);
    try { out[i] = p_->Back(std::move(cur)); }
    catch (...) { if (next.valid()) next.wait(); throw; }
  }
  return out;
}

}  // namespace rpocr::indra
