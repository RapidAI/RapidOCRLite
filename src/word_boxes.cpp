// Word and character boxes (rapidocr main return_word_box), in C++ for the
// C / C++ / Java APIs. A line-for-line port of python/src/rapidocr/
// cal_rec_boxes.py (itself a port of main's CTCLabelDecode.get_word_info and
// CalRecBoxes at RapidAI/RapidOCR 8ac45f4e) for one line, with the
// float32 / float64 / int steps numpy takes there; tools/word_parity/
// c_words_check.py compares the two on every parity page.
#include "rapidocrlite/rapidocrlite.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include "np_argsort.hpp"
#include "word_boxes.hpp"

namespace rpocr {
namespace {

std::u32string Utf8To32(const std::string& s) {
  std::u32string out;
  for (std::size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    char32_t cp = c;
    int n = 1;
    if (c >= 0xF0) { cp = c & 0x07; n = 4; }
    else if (c >= 0xE0) { cp = c & 0x0F; n = 3; }
    else if (c >= 0xC0) { cp = c & 0x1F; n = 2; }
    if (i + n > s.size()) n = 1, cp = c;
    for (int k = 1; k < n; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    out.push_back(cp);
    i += n;
  }
  return out;
}

std::string Utf32To8(const std::u32string& s) {
  std::string out;
  for (char32_t c : s) {
    if (c < 0x80) out.push_back(static_cast<char>(c));
    else if (c < 0x800) { out.push_back(char(0xC0 | (c >> 6))); out.push_back(char(0x80 | (c & 0x3F))); }
    else if (c < 0x10000) {
      out.push_back(char(0xE0 | (c >> 12))); out.push_back(char(0x80 | ((c >> 6) & 0x3F)));
      out.push_back(char(0x80 | (c & 0x3F)));
    } else {
      out.push_back(char(0xF0 | (c >> 18))); out.push_back(char(0x80 | ((c >> 12) & 0x3F)));
      out.push_back(char(0x80 | ((c >> 6) & 0x3F))); out.push_back(char(0x80 | (c & 0x3F)));
    }
  }
  return out;
}

// Python str.isspace.
bool IsSpace(char32_t c) {
  return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 ||
         c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
         c == 0x202F || c == 0x205F || c == 0x3000;
}

bool IsChinese(char32_t c) {
  return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3000 && c <= 0x303F) || (c >= 0xFF00 && c <= 0xFFEF);
}

// Python round(x, digits): correctly rounded decimal (printf rounds the
// exact binary value the same way), back to the nearest double.
double PyRound(double x, int digits) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.*f", digits, x);
  return std::strtod(buf, nullptr);
}

enum class WType { cn, en_num };

struct Info {
  std::vector<std::u32string> words;
  std::vector<std::vector<int>> cols;
  std::vector<WType> types;
  double line_txt_len = 0;
  std::vector<double> confs;
};

Info GetWordInfo(const std::u32string& text, const std::vector<int>& valid) {
  Info info;
  if (valid.empty() || text.empty()) return info;
  std::vector<double> col_width(valid.size(), 0.0);
  for (std::size_t k = 1; k < valid.size(); ++k) col_width[k] = valid[k] - valid[k - 1];
  col_width[0] = std::min(IsChinese(text[0]) ? 3 : 2, valid[0]);
  bool have_state = false;
  WType state = WType::en_num;
  std::u32string word;
  std::vector<int> word_cols;
  const auto flush = [&] {
    info.words.push_back(word);
    info.cols.push_back(word_cols);
    info.types.push_back(state);
    word.clear();
    word_cols.clear();
  };
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char32_t ch = text[i];
    if (IsSpace(ch)) {
      if (!word.empty()) flush();
      continue;
    }
    if (i >= valid.size()) break;  // main raises IndexError here
    const WType c_state = IsChinese(ch) ? WType::cn : WType::en_num;
    if (!have_state) { state = c_state; have_state = true; }
    if (state != c_state || col_width[i] > 5) {
      if (!word.empty()) flush();
      state = c_state;
    }
    word.push_back(ch);
    word_cols.push_back(valid[i]);
  }
  if (!word.empty()) flush();
  return info;
}

using Quad = std::array<std::array<double, 2>, 4>;

std::array<double, 4> RectOf(const std::vector<Quad>& quads) {
  double x0 = quads[0][0][0], y0 = quads[0][0][1], x1 = x0, y1 = y0;
  for (const auto& q : quads)
    for (const auto& p : q) {
      x0 = std::min(x0, p[0]); x1 = std::max(x1, p[0]);
      y0 = std::min(y0, p[1]); y1 = std::max(y1, p[1]);
    }
  return {x0, y0, x1, y1};
}

std::vector<Quad> CalcBox(const std::vector<int>& cols, double avg_char_width, double avg_col_width,
                          const std::array<double, 4>& b) {
  const double x0 = b[0], y0 = b[1], x1 = b[2], y1 = b[3];
  std::vector<Quad> out;
  for (const int col : cols) {
    const double center_x = (col + 0.5) * avg_col_width;
    const double cx0 = std::max(std::trunc(center_x - avg_char_width / 2), 0.0) + x0;
    const double cx1 = std::min(std::trunc(center_x + avg_char_width / 2), x1 - x0) + x0;
    out.push_back({{{cx0, y0}, {cx1, y0}, {cx1, y1}, {cx0, y1}}});
  }
  std::stable_sort(out.begin(), out.end(), [](const Quad& a, const Quad& c) { return a[0][0] < c[0][0]; });
  return out;
}

float Norm32(float dx, float dy) {
  volatile float a = dx * dx;
  volatile float c = dy * dy;
  return std::sqrt(static_cast<float>(a + c));
}

bool LuSolve(std::array<std::array<double, 8>, 8>& a, std::array<double, 8>& b) {
  const int m = 8;
  const double eps = std::numeric_limits<double>::epsilon() * 100;
  for (int i = 0; i < m; ++i) {
    int k = i;
    for (int j = i + 1; j < m; ++j)
      if (std::fabs(a[j][i]) > std::fabs(a[k][i])) k = j;
    if (std::fabs(a[k][i]) < eps) return false;
    if (k != i) { std::swap(a[i], a[k]); std::swap(b[i], b[k]); }
    const double d = -1.0 / a[i][i];
    for (int j = i + 1; j < m; ++j) {
      const double alpha = a[j][i] * d;
      for (int c = i + 1; c < m; ++c) a[j][c] += alpha * a[i][c];
      b[j] += alpha * b[i];
    }
  }
  for (int i = m - 1; i >= 0; --i) {
    double s = b[i];
    for (int c = i + 1; c < m; ++c) s -= a[i][c] * b[c];
    b[i] = s / a[i][i];
  }
  return true;
}

bool Perspective(const float src[4][2], const float dst[4][2], double m[3][3]) {
  std::array<std::array<double, 8>, 8> a{};
  std::array<double, 8> b{};
  for (int i = 0; i < 4; ++i) {
    const float sx = src[i][0], sy = src[i][1], dx = dst[i][0], dy = dst[i][1];
    a[i][0] = a[i + 4][3] = sx;
    a[i][1] = a[i + 4][4] = sy;
    a[i][2] = a[i + 4][5] = 1.0;
    volatile float p0 = -sx * dx, p1 = -sy * dx, p2 = -sx * dy, p3 = -sy * dy;
    a[i][6] = p0; a[i][7] = p1; a[i + 4][6] = p2; a[i + 4][7] = p3;
    b[i] = dx;
    b[i + 4] = dy;
  }
  if (!LuSolve(a, b)) return false;
  for (int k = 0; k < 8; ++k) m[k / 3][k % 3] = b[k];
  m[2][2] = 1.0;
  return true;
}

void Invert3(const double s[3][3], double o[3][3]) {
  double d = s[0][0] * (s[1][1] * s[2][2] - s[1][2] * s[2][1]) -
             s[0][1] * (s[1][0] * s[2][2] - s[1][2] * s[2][0]) +
             s[0][2] * (s[1][0] * s[2][1] - s[1][1] * s[2][0]);
  for (int r = 0; r < 3; ++r) for (int c = 0; c < 3; ++c) o[r][c] = 0;
  if (d == 0.0) return;
  d = 1.0 / d;
  o[0][0] = (s[1][1] * s[2][2] - s[1][2] * s[2][1]) * d;
  o[0][1] = (s[0][2] * s[2][1] - s[0][1] * s[2][2]) * d;
  o[0][2] = (s[0][1] * s[1][2] - s[0][2] * s[1][1]) * d;
  o[1][0] = (s[1][2] * s[2][0] - s[1][0] * s[2][2]) * d;
  o[1][1] = (s[0][0] * s[2][2] - s[0][2] * s[2][0]) * d;
  o[1][2] = (s[0][2] * s[1][0] - s[0][0] * s[1][2]) * d;
  o[2][0] = (s[1][0] * s[2][1] - s[1][1] * s[2][0]) * d;
  o[2][1] = (s[0][1] * s[2][0] - s[0][0] * s[2][1]) * d;
  o[2][2] = (s[0][0] * s[1][1] - s[0][1] * s[1][0]) * d;
}

// np.dot(IM, [x, y, 1]) is a 3x3 float64 gemv in OpenBLAS. Its kernel on
// FMA hardware computes fma(a0, x, a1 * y) + a2 (verified bit for bit
// against numpy 2.5 / scipy-openblas 0.3.34 on x86-64 CI: mode 7); without
// FMA plain left-to-right sums. PPOCR_WORDBOX_DOT forces a mode
// (diagnostics): 0 plain, 1-8 the variants below.
int DotMode() {
  static const int mode = [] {
    if (const char* e = std::getenv("PPOCR_WORDBOX_DOT")) return std::atoi(e);
#if defined(__aarch64__) || defined(_M_ARM64)
    return 7;
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // NumPy's X86_V3 group (AVX2 + FMA ...) is what OpenBLAS's Haswell+ kernels need.
    return detail::NumpyArgsortIsa() != detail::ArgsortIsa::scalar ? 7 : 0;
#else
    return 0;
#endif
  }();
  return mode;
}

double Dot3(const double* r, double px, double py) {
  switch (DotMode()) {
    case 1: return std::fma(r[2], 1.0, std::fma(r[1], py, r[0] * px));
    case 2: return std::fma(r[0], px, std::fma(r[1], py, r[2]));
    case 3: return std::fma(r[1], py, std::fma(r[0], px, r[2]));
    case 4: { volatile double a = r[0] * px, c = r[2] * 1.0, b = r[1] * py; volatile double t = a + c; return t + b; }
    case 5: { volatile double a = r[0] * px, b = r[1] * py, c = r[2]; volatile double t = b + c; return a + t; }
    case 6: return std::fma(r[0], px, std::fma(r[1], py, r[2] * 1.0));
    case 7: { volatile double t = std::fma(r[0], px, r[1] * py); return t + r[2]; }
    case 8: { volatile double t = std::fma(r[1], py, r[0] * px); return t + r[2]; }
    default: {
      volatile double a = r[0] * px;
      volatile double b = r[1] * py;
      volatile double c = a + b;
      return c + r[2];
    }
  }
}

using IPt = std::array<long long, 2>;

// main's CalRecBoxes.order_points on integer points.
std::array<IPt, 4> OrderPoints(const std::array<IPt, 4>& box) {
  const double cx = (double(box[0][0]) + double(box[1][0]) + double(box[2][0]) + double(box[3][0])) / 4;
  const double cy = (double(box[0][1]) + double(box[1][1]) + double(box[2][1]) + double(box[3][1])) / 4;
  bool any_x = false, any_y = false, all_x = true, all_y_ne = true;
  for (const auto& p : box) {
    any_x |= double(p[0]) == cx; any_y |= double(p[1]) == cy;
    all_x &= double(p[0]) == cx; all_y_ne &= double(p[1]) != cy;
  }
  const auto first = [&](auto pred, const IPt*& out) {
    for (const auto& p : box) if (pred(p)) { out = &p; return; }
  };
  const IPt *p1 = nullptr, *p2 = nullptr, *p3 = nullptr, *p4 = nullptr;
  long long mnx = box[0][0], mxx = mnx, mny = box[0][1], mxy = mny;
  for (const auto& p : box) {
    mnx = std::min(mnx, p[0]); mxx = std::max(mxx, p[0]);
    mny = std::min(mny, p[1]); mxy = std::max(mxy, p[1]);
  }
  if (any_x && any_y) {
    first([&](const IPt& p) { return p[0] == mnx; }, p1);
    first([&](const IPt& p) { return p[1] == mny; }, p2);
    first([&](const IPt& p) { return p[0] == mxx; }, p3);
    first([&](const IPt& p) { return p[1] == mxy; }, p4);
  } else if (all_x) {
    double ys[4];
    for (int k = 0; k < 4; ++k) ys[k] = double(box[k][1]);
    std::size_t o[4];
    detail::NumpyArgsort(ys, 4, o);  // np.argsort on int64 takes the same networks
    return {box[o[0]], box[o[1]], box[o[2]], box[o[3]]};
  } else {
    const bool split_y = any_x && all_y_ne;
    // split_y: p12 = y < cy, p34 = y > cy; else p14 = x < cx, p23 = x > cx.
    const auto lo = [&](const IPt& p) { return split_y ? double(p[1]) < cy : double(p[0]) < cx; };
    const auto hi = [&](const IPt& p) { return split_y ? double(p[1]) > cy : double(p[0]) > cx; };
    const int key_axis = split_y ? 0 : 1;
    long long lo_min = 0, lo_max = 0, hi_min = 0, hi_max = 0;
    bool lo_any = false, hi_any = false;
    for (const auto& p : box) {
      if (lo(p)) { if (!lo_any) lo_min = lo_max = p[key_axis]; lo_min = std::min(lo_min, p[key_axis]); lo_max = std::max(lo_max, p[key_axis]); lo_any = true; }
      if (hi(p)) { if (!hi_any) hi_min = hi_max = p[key_axis]; hi_min = std::min(hi_min, p[key_axis]); hi_max = std::max(hi_max, p[key_axis]); hi_any = true; }
    }
    if (!lo_any || !hi_any) return box;  // main fails to reshape here
    if (split_y) {
      first([&](const IPt& p) { return lo(p) && p[0] == lo_min; }, p1);
      first([&](const IPt& p) { return lo(p) && p[0] == lo_max; }, p2);
      first([&](const IPt& p) { return hi(p) && p[0] == hi_max; }, p3);
      first([&](const IPt& p) { return hi(p) && p[0] == hi_min; }, p4);
    } else {
      first([&](const IPt& p) { return lo(p) && p[1] == lo_min; }, p1);
      first([&](const IPt& p) { return lo(p) && p[1] == lo_max; }, p4);
      first([&](const IPt& p) { return hi(p) && p[1] == hi_min; }, p2);
      first([&](const IPt& p) { return hi(p) && p[1] == hi_max; }, p3);
    }
  }
  if (!p1 || !p2 || !p3 || !p4) return box;
  return {*p1, *p2, *p3, *p4};
}

}  // namespace

std::vector<Word> WordBoxes(const Result& line, bool single_char) {
  float q[4][2];
  for (int k = 0; k < 4; ++k) { q[k][0] = float(line.box[k][0]); q[k][1] = float(line.box[k][1]); }
  return detail::WordBoxesAt(line, q, single_char);
}

namespace detail {
std::vector<Word> WordBoxesAt(const Result& line, const float box[4][2], bool single_char) {
  std::vector<Word> out;
  const std::u32string text = Utf8To32(line.text);
  bool blank = true;
  for (char32_t c : text) blank &= IsSpace(c);
  if (blank || line.rec_steps <= 0 || line.crop_width <= 0 || line.crop_height <= 0 ||
      line.rec_max_wh_ratio <= 0)
    return out;
  // build_word_info
  std::vector<int> valid(line.char_cols.begin(), line.char_cols.end());
  std::sort(valid.begin(), valid.end());
  valid.erase(std::unique(valid.begin(), valid.end()), valid.end());
  Info info = GetWordInfo(text, valid);
  info.line_txt_len = line.rec_steps * (line.crop_width * 1.0 / line.crop_height) / line.rec_max_wh_ratio;
  for (const float p : line.char_probs) info.confs.push_back(PyRound(static_cast<double>(p), 5));
  if (info.confs.empty()) info.confs.push_back(0.0);
  if (info.line_txt_len == 0) return out;

  // cal_ocr_word_box
  const std::array<double, 4> bp{0.0, 0.0, double(line.crop_width), double(line.crop_height)};
  const double avg_col_width = (bp[2] - bp[0]) / info.line_txt_len;
  bool all_en = true;
  for (const auto t : info.types) all_en &= t == WType::en_num;
  const bool en_words = all_en && !single_char;
  std::vector<std::vector<int>> word_cols;
  std::vector<int> flat_cols;
  std::vector<std::string> contents;
  std::vector<double> char_widths;
  for (std::size_t w = 0; w < info.words.size(); ++w) {
    const auto& wc = info.cols[w];
    if (en_words) {
      word_cols.push_back(wc);
      contents.push_back(Utf32To8(info.words[w]));
    } else {
      flat_cols.insert(flat_cols.end(), wc.begin(), wc.end());
      for (char32_t c : info.words[w]) contents.push_back(Utf32To8(std::u32string(1, c)));
    }
    if (wc.size() == 1) continue;
    char_widths.push_back((wc.back() - wc.front()) * avg_col_width / double(wc.size() - 1));
  }
  double avg_char_width = 0.0;
  if (!char_widths.empty()) {
    double sum = 0.0;
    for (double v : char_widths) sum += v;
    avg_char_width = sum / double(char_widths.size());
  } else {
    avg_char_width = (bp[2] - bp[0]) / double(text.size());
  }
  std::vector<Quad> boxes;
  if (en_words) {
    for (const auto& wc : word_cols) {
      const auto cells = CalcBox(wc, avg_char_width, avg_col_width, bp);
      const auto r = RectOf(cells);
      boxes.push_back({{{r[0], r[1]}, {r[2], r[1]}, {r[2], r[3]}, {r[0], r[3]}}});
    }
  } else {
    boxes = CalcBox(flat_cols, avg_char_width, avg_col_width, bp);
  }
  // adjust_box_overlap
  for (std::size_t i = 0; i + 1 < boxes.size(); ++i) {
    Quad& cur = boxes[i];
    Quad& nxt = boxes[i + 1];
    if (cur[1][0] > nxt[0][0]) {
      const double distance = std::fabs(cur[1][0] - nxt[0][0]);
      cur[1][0] -= distance / 2;
      cur[2][0] -= distance / 2;
      nxt[0][0] += distance - distance / 2;
      nxt[3][0] += distance - distance / 2;
    }
  }
  // get_box_direction on the line box (float32)
  float q[4][2];
  for (int k = 0; k < 4; ++k) { q[k][0] = box[k][0]; q[k][1] = box[k][1]; }
  const double e0 = Norm32(q[0][0] - q[1][0], q[0][1] - q[1][1]);
  const double e1 = Norm32(q[1][0] - q[2][0], q[1][1] - q[2][1]);
  const double e2 = Norm32(q[2][0] - q[3][0], q[2][1] - q[3][1]);
  const double e3 = Norm32(q[3][0] - q[0][0], q[3][1] - q[0][1]);
  const double bw = std::max(e0, e2), bh = std::max(e1, e3);
  const bool vertical = bw < 1e-6 || PyRound(bh / bw, 2) >= 1.5;
  // reverse_rotate_crop_image
  float mnx = q[0][0], mny = q[0][1];
  for (int k = 1; k < 4; ++k) { mnx = std::min(mnx, q[k][0]); mny = std::min(mny, q[k][1]); }
  const long long left = static_cast<long long>(mnx), top = static_cast<long long>(mny);
  for (int k = 0; k < 4; ++k) { q[k][0] = q[k][0] - float(left); q[k][1] = q[k][1] - float(top); }
  const int crop_w = static_cast<int>(Norm32(q[0][0] - q[1][0], q[0][1] - q[1][1]));
  const int crop_h = static_cast<int>(Norm32(q[0][0] - q[3][0], q[0][1] - q[3][1]));
  const float dst[4][2] = {{0.F, 0.F}, {float(crop_w), 0.F}, {float(crop_w), float(crop_h)}, {0.F, float(crop_h)}};
  double m[3][3], im[3][3];
  if (!Perspective(q, dst, m)) return out;
  Invert3(m, im);
  const double ang = -90.0 * (3.141592653589793 / 180.0);  // CPython math.radians(-90)
  const double ca = std::cos(ang), sa = std::sin(ang);
  const std::size_t n = std::min({contents.size(), info.confs.size(), boxes.size()});
  for (std::size_t w = 0; w < n; ++w) {
    std::array<IPt, 4> pts{};
    for (int k = 0; k < 4; ++k) {
      double px = boxes[w][k][0], py = boxes[w][k][1];
      if (vertical) {
        const double rx = px * ca + py * sa;
        const double ry = py * ca - px * sa;
        px = rx + crop_w;
        py = ry;
      }
      const double x = Dot3(im[0], px, py), y = Dot3(im[1], px, py), z = Dot3(im[2], px, py);
      pts[k] = {static_cast<long long>(std::trunc(x / z + double(left))),
                static_cast<long long>(std::trunc(y / z + double(top)))};
    }
    const auto ordered = OrderPoints(pts);
    Word word;
    word.text = contents[w];
    word.confidence = info.confs[w];
    for (int k = 0; k < 4; ++k) word.box[k] = {int(ordered[k][0]), int(ordered[k][1])};
    out.push_back(std::move(word));
  }
  return out;
}
}  // namespace detail

}  // namespace rpocr
