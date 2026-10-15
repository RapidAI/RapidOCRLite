// rapidocr 3.x DBPostProcess geometry, bit-exact. This translation unit is
// compiled without -ffast-math / with /fp:precise: it reproduces OpenCV's
// float32 cv2.minAreaRect / cv2.boxPoints / cv2.fillPoly (ported from
// OpenCV, Apache-2.0) and pyclipper's integer polygon offset (vendored
// Clipper 6.4.2, Boost Software License, src/clipper/).
#include "db_geometry.hpp"

#include "clipper/clipper.hpp"

// Port of the OpenCV (Apache-2.0) geometry used by rapidocr's DB
// postprocess: convexHull (Sklansky, integer input, counter-clockwise,
// returnPoints), the rotating-calipers cv::minAreaRect and
// RotatedRect::points (cv2.boxPoints), reproduced in the same float32
// arithmetic and index order so the boxes match cv2 bit for bit.
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <climits>
#include <cstdio>
#include <utility>
#include <vector>

namespace ppocr { namespace detail { namespace cvgeom {

constexpr double kPi = 3.1415926535897932384626433832795;  // CV_PI

struct IPt { int x, y; };
struct FPt { float x, y; };
struct RRect { FPt center; float w, h, angle; };

inline int Sign(long long v) { return (v > 0) - (v < 0); }

inline int Sklansky(const IPt* const* array, int start, int end, int* stack, int nsign,
                    int sign2) {
  int incr = end > start ? 1 : -1;
  int pprev = start, pcur = pprev + incr, pnext = pcur + incr;
  int stacksize = 3;
  if (start == end || (array[start]->x == array[end]->x && array[start]->y == array[end]->y)) {
    stack[0] = start;
    return 1;
  }
  stack[0] = pprev; stack[1] = pcur; stack[2] = pnext;
  end += incr;
  while (pnext != end) {
    const int cury = array[pcur]->y, nexty = array[pnext]->y, by = nexty - cury;
    if (Sign(by) != nsign) {
      const int ax = array[pcur]->x - array[pprev]->x, ay = cury - array[pprev]->y;
      const int bx = array[pnext]->x - array[pcur]->x;
      const long long convexity = (long long)ay * bx - (long long)ax * by;
      if (Sign(convexity) == sign2 && (ax != 0 || ay != 0)) {
        pprev = pcur; pcur = pnext; pnext += incr;
        stack[stacksize] = pnext; stacksize++;
      } else if (pprev == start) {
        pcur = pnext; stack[1] = pcur; pnext += incr; stack[2] = pnext;
      } else {
        stack[stacksize - 2] = pnext;
        pcur = pprev;
        pprev = stack[stacksize - 4];
        stacksize--;
      }
    } else {
      pnext += incr;
      stack[stacksize - 1] = pnext;
    }
  }
  return --stacksize;
}

// cv::convexHull(points, hull, clockwise=false, returnPoints=true), int32.
inline void ConvexHull(const IPt* data0, int total, std::vector<IPt>& out,
                       std::vector<const IPt*>& pointer, std::vector<int>& stack,
                       std::vector<int>& hullbuf) {
  out.clear();
  if (total <= 0) return;
  pointer.resize(total);
  stack.assign(std::size_t(total) + 2, 0);
  hullbuf.assign(std::size_t(total), 0);
  for (int i = 0; i < total; ++i) pointer[i] = &data0[i];
  std::sort(pointer.begin(), pointer.end(), [](const IPt* p1, const IPt* p2) {
    if (p1->x != p2->x) return p1->x < p2->x;
    if (p1->y != p2->y) return p1->y < p2->y;
    return p1 < p2;
  });
  int miny_ind = 0, maxy_ind = 0, nout = 0;
  for (int i = 1; i < total; ++i) {
    const int y = pointer[i]->y;
    if (pointer[miny_ind]->y > y) miny_ind = i;
    if (pointer[maxy_ind]->y < y) maxy_ind = i;
  }
  const IPt* const* P = pointer.data();
  int* st = stack.data();
  int* hb = hullbuf.data();
  if (P[0]->x == P[total - 1]->x && P[0]->y == P[total - 1]->y) {
    hb[nout++] = 0;
  } else {
    int* tl_stack = st;
    int tl_count = Sklansky(P, 0, maxy_ind, tl_stack, -1, 1);
    int* tr_stack = st + tl_count;
    int tr_count = Sklansky(P, total - 1, maxy_ind, tr_stack, -1, -1);
    std::swap(tl_stack, tr_stack);  // !clockwise
    std::swap(tl_count, tr_count);
    for (int i = 0; i < tl_count - 1; i++) hb[nout++] = tl_stack[i];
    for (int i = tr_count - 1; i > 0; i--) hb[nout++] = tr_stack[i];
    const int stop_idx = tr_count > 2 ? tr_stack[1] : tl_count > 2 ? tl_stack[tl_count - 2] : -1;
    int* bl_stack = st;
    int bl_count = Sklansky(P, 0, miny_ind, bl_stack, 1, -1);
    int* br_stack = st + bl_count;
    int br_count = Sklansky(P, total - 1, miny_ind, br_stack, 1, 1);
    if (stop_idx >= 0) {
      const int check_idx = bl_count > 2 ? bl_stack[1]
                            : bl_count + br_count > 2 ? br_stack[2 - bl_count] : -1;
      if (check_idx == stop_idx ||
          (check_idx >= 0 && P[check_idx]->x == P[stop_idx]->x &&
           P[check_idx]->y == P[stop_idx]->y)) {
        bl_count = std::min(bl_count, 2);
        br_count = std::min(br_count, 2);
      }
    }
    for (int i = 0; i < bl_count - 1; i++) hb[nout++] = bl_stack[i];
    for (int i = br_count - 1; i > 0; i--) hb[nout++] = br_stack[i];
  }
  for (int i = 0; i < nout; ++i) hb[i] = int(P[hb[i]] - data0);
  if (nout >= 3) {
    int min_idx = 0, max_idx = 0, lt = 0;
    for (int i = 1; i < nout; i++) {
      const int idx = hb[i];
      lt += hb[i - 1] < idx;
      if (lt > 1 && lt <= i - 2) break;
      if (idx < hb[min_idx]) min_idx = i;
      if (idx > hb[max_idx]) max_idx = i;
    }
    const int mmdist = std::abs(max_idx - min_idx);
    if ((mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt >= nout - 2)) {
      const int ascending = (max_idx + 1) % nout == min_idx;
      const int i0 = ascending ? min_idx : max_idx;
      int j = i0;
      if (i0 > 0) {
        int i;
        for (i = 0; i < nout; i++) {
          const int curr_idx = st[i] = hb[j];
          const int next_j = j + 1 < nout ? j + 1 : 0;
          const int next_idx = hb[next_j];
          if (i < nout - 1 && (ascending != (curr_idx < next_idx))) break;
          j = next_j;
        }
        if (i == nout) std::copy(st, st + nout, hb);
      }
    }
  }
  out.resize(std::size_t(nout));
  for (int i = 0; i < nout; ++i) out[i] = data0[hb[i]];
}

inline bool FirstVecIsRight(const FPt& v1, const FPt& v2) {
  const FPt tmp{v1.y, -v1.x};
  return tmp.x * v2.x + tmp.y * v2.y < 0;
}

inline void RotatingCalipersMinArea(const FPt* points, int n, float* out,
                                    std::vector<float>& abuf) {
  float minarea = FLT_MAX;
  abuf.resize(std::size_t(n) * 3);
  float* inv_vect_length = abuf.data();
  FPt* vect = reinterpret_cast<FPt*>(inv_vect_length + n);
  int left = 0, bottom = 0, right = 0, top = 0;
  int seq[4] = {-1, -1, -1, -1};
  FPt rot_vect[4];
  float base_a, base_b = 0;
  FPt pt0 = points[0];
  float left_x = pt0.x, right_x = pt0.x, top_y = pt0.y, bottom_y = pt0.y;
  for (int i = 0; i < n; i++) {
    if (pt0.x < left_x) left_x = pt0.x, left = i;
    if (pt0.x > right_x) right_x = pt0.x, right = i;
    if (pt0.y > top_y) top_y = pt0.y, top = i;
    if (pt0.y < bottom_y) bottom_y = pt0.y, bottom = i;
    const FPt pt = points[(i + 1) & (i + 1 < n ? -1 : 0)];
    const double dx = double(pt.x) - double(pt0.x), dy = double(pt.y) - double(pt0.y);
    vect[i].x = float(dx);
    vect[i].y = float(dy);
    inv_vect_length[i] = float(1. / std::sqrt(dx * dx + dy * dy));
    pt0 = pt;
  }
  float orientation = 1.f;
  base_a = orientation;
  seq[0] = bottom; seq[1] = right; seq[2] = top; seq[3] = left;
  int best_left = 0, best_bottom = 0;
  float best_a = 0, best_w = 0, best_b = 0, best_h = 0;
  for (int k = 0; k < n; k++) {
    int main_element = 0;
    rot_vect[0] = vect[seq[0]];
    rot_vect[1] = FPt{vect[seq[1]].y, -vect[seq[1]].x};
    rot_vect[2] = FPt{-vect[seq[2]].x, -vect[seq[2]].y};
    rot_vect[3] = FPt{-vect[seq[3]].y, vect[seq[3]].x};
    for (int i = 1; i < 4; i++)
      if (FirstVecIsRight(rot_vect[i], rot_vect[main_element])) main_element = i;
    {
      const int pindex = seq[main_element];
      const float lead_x = vect[pindex].x * inv_vect_length[pindex];
      const float lead_y = vect[pindex].y * inv_vect_length[pindex];
      switch (main_element) {
        case 0: base_a = lead_x; base_b = lead_y; break;
        case 1: base_a = lead_y; base_b = -lead_x; break;
        case 2: base_a = -lead_x; base_b = -lead_y; break;
        default: base_a = -lead_y; base_b = lead_x; break;
      }
    }
    seq[main_element] += 1;
    seq[main_element] = (seq[main_element] == n) ? 0 : seq[main_element];
    float dx = points[seq[1]].x - points[seq[3]].x;
    float dy = points[seq[1]].y - points[seq[3]].y;
    const float width = dx * base_a + dy * base_b;
    dx = points[seq[2]].x - points[seq[0]].x;
    dy = points[seq[2]].y - points[seq[0]].y;
    const float height = -dx * base_b + dy * base_a;
    const float area = width * height;
    if (area <= minarea) {
      minarea = area;
      best_left = seq[3]; best_a = base_a; best_w = width; best_b = base_b;
      best_h = height; best_bottom = seq[0];
    }
  }
  const float A1 = best_a, B1 = best_b, A2 = -best_b, B2 = best_a;
  const float C1 = A1 * points[best_left].x + points[best_left].y * B1;
  const float C2 = A2 * points[best_bottom].x + points[best_bottom].y * B2;
  const float idet = 1.f / (A1 * B2 - A2 * B1);
  out[0] = (C1 * B2 - C2 * B1) * idet;
  out[1] = (A1 * C2 - A2 * C1) * idet;
  out[2] = A1 * best_w;
  out[3] = B1 * best_w;
  out[4] = A2 * best_h;
  out[5] = B2 * best_h;
}


// cv::fillPoly(mask, [pts], 1) with lineType 8, shift 0 (int vertices):
// the 8-connected outline of every edge plus the scan-line interior.
inline bool ClipLine(long long w, long long h, long long& x1, long long& y1, long long& x2,
                     long long& y2) {
  const long long right = w - 1, bottom = h - 1;
  if (w <= 0 || h <= 0) return false;
  int c1 = (x1 < 0) + (x1 > right) * 2 + (y1 < 0) * 4 + (y1 > bottom) * 8;
  int c2 = (x2 < 0) + (x2 > right) * 2 + (y2 < 0) * 4 + (y2 > bottom) * 8;
  if ((c1 & c2) == 0 && (c1 | c2) != 0) {
    long long a;
    if (c1 & 12) {
      a = c1 < 8 ? 0 : bottom;
      x1 += (long long)((double)(a - y1) * (x2 - x1) / (y2 - y1));
      y1 = a;
      c1 = (x1 < 0) + (x1 > right) * 2;
    }
    if (c2 & 12) {
      a = c2 < 8 ? 0 : bottom;
      x2 += (long long)((double)(a - y2) * (x2 - x1) / (y2 - y1));
      y2 = a;
      c2 = (x2 < 0) + (x2 > right) * 2;
    }
    if ((c1 & c2) == 0 && (c1 | c2) != 0) {
      if (c1) {
        a = c1 == 1 ? 0 : right;
        y1 += (long long)((double)(a - x1) * (y2 - y1) / (x2 - x1));
        x1 = a;
        c1 = 0;
      }
      if (c2) {
        a = c2 == 1 ? 0 : right;
        y2 += (long long)((double)(a - x2) * (y2 - y1) / (x2 - x1));
        x2 = a;
        c2 = 0;
      }
    }
  }
  return (c1 | c2) == 0;
}

inline void DrawLine8(std::uint8_t* img, int w, int h, int x1, int y1, int x2, int y2) {
  long long a1 = x1, b1 = y1, a2 = x2, b2 = y2;
  if ((unsigned)x1 >= (unsigned)w || (unsigned)x2 >= (unsigned)w || (unsigned)y1 >= (unsigned)h ||
      (unsigned)y2 >= (unsigned)h) {
    if (!ClipLine(w, h, a1, b1, a2, b2)) return;
  }
  int px = int(a1), py = int(b1);
  int dx = int(a2 - a1), dy = int(b2 - b1);
  int delta_x = 1, delta_y = 1;
  if (dx < 0) { dx = -dx; dy = -dy; px = int(a2); py = int(b2); }
  if (dy < 0) { dy = -dy; delta_y = -1; }
  const bool vert = dy > dx;
  if (vert) { std::swap(dx, dy); std::swap(delta_x, delta_y); }
  int err = dx - (dy + dy);
  const int plusDelta = dx + dx, minusDelta = -(dy + dy);
  int minusShift = delta_x, plusShift = 0, minusStep = 0, plusStep = delta_y;
  if (vert) { std::swap(plusStep, plusShift); std::swap(minusStep, minusShift); }
  for (int i = 0; i < dx + 1; ++i) {
    img[std::size_t(py) * w + px] = 1;
    const int mask = err < 0 ? -1 : 0;
    err += minusDelta + (plusDelta & mask);
    px += minusShift + (plusShift & mask);
    py += minusStep + (plusStep & mask);
  }
}

struct PolyEdge { int y0, y1; long long x, dx; PolyEdge* next; };

inline void FillPoly(std::uint8_t* img, int w, int h, const IPt* v, int count,
                     std::vector<PolyEdge>& edges) {
  constexpr int XY_SHIFT = 16;
  constexpr long long XY_ONE = 1LL << XY_SHIFT;
  edges.clear();
  long long p0x = (long long)v[count - 1].x << XY_SHIFT, p0y = v[count - 1].y;
  for (int i = 0; i < count; ++i) {
    long long p1x = (long long)v[i].x << XY_SHIFT, p1y = v[i].y;
    long long p0cx = p0x, p0cy = p0y, p1cx = p1x, p1cy = p1y;
    long long t0x = (p0x + (XY_ONE >> 1)) >> XY_SHIFT, t0y = p0y;
    long long t1x = (p1x + (XY_ONE >> 1)) >> XY_SHIFT, t1y = p1y;
    DrawLine8(img, w, h, int(t0x), int(t0y), int(t1x), int(t1y));
    if ((unsigned long long)t0x >= (unsigned long long)w ||
        (unsigned long long)t1x >= (unsigned long long)w ||
        (unsigned long long)t0y >= (unsigned long long)h ||
        (unsigned long long)t1y >= (unsigned long long)h) {
      ClipLine(w, h, t0x, t0y, t1x, t1y);
      if (t0y != t1y) { p0cy = t0y; p1cy = t1y; }
    }
    p0cx = t0x << XY_SHIFT;
    p1cx = t1x << XY_SHIFT;
    if (p0y != p1y) {
      PolyEdge e{};
      e.dx = (p1cx - p0cx) / (p1cy - p0cy);
      if (p0y < p1y) { e.y0 = int(p0y); e.y1 = int(p1y); e.x = p0cx + (p0y - p0cy) * e.dx; }
      else { e.y0 = int(p1y); e.y1 = int(p0y); e.x = p1cx + (p1y - p1cy) * e.dx; }
      edges.push_back(e);
    }
    p0x = p1x; p0y = p1y;
  }
  const int total = int(edges.size());
  if (total < 2) return;
  int y_max = INT32_MIN, y_min = INT32_MAX;
  long long x_max = -1, x_min = 0x7FFFFFFFFFFFFFFFLL;
  for (auto& e1 : edges) {
    const long long x1 = e1.x + (e1.y1 - e1.y0) * e1.dx;
    y_min = std::min(y_min, e1.y0); y_max = std::max(y_max, e1.y1);
    x_min = std::min({x_min, e1.x, x1}); x_max = std::max({x_max, e1.x, x1});
  }
  if (y_max < 0 || y_min >= h || x_max < 0 || x_min >= ((long long)w << XY_SHIFT)) return;
  std::sort(edges.begin(), edges.end(), [](const PolyEdge& e1, const PolyEdge& e2) {
    return e1.y0 - e2.y0 ? e1.y0 < e2.y0 : e1.x - e2.x ? e1.x < e2.x : e1.dx < e2.dx;
  });
  PolyEdge tmp{}; tmp.y0 = INT32_MAX;
  edges.push_back(tmp);
  int i = 0;
  tmp.next = nullptr;
  PolyEdge* e = &edges[0];
  y_max = std::min(y_max, h);
  const long long delta = XY_ONE - 1;
  for (int y = e->y0; y < y_max; y++) {
    PolyEdge *last, *prelast, *keep_prelast;
    int draw = 0;
    const bool clipline = y < 0;
    prelast = &tmp;
    last = tmp.next;
    while (last || e->y0 == y) {
      if (last && last->y1 == y) { prelast->next = last->next; last = last->next; continue; }
      keep_prelast = prelast;
      if (last && (e->y0 > y || last->x < e->x)) { prelast = last; last = last->next; }
      else if (i < total) { prelast->next = e; e->next = last; prelast = e; e = &edges[++i]; }
      else break;
      if (draw) {
        if (!clipline) {
          int x1, x2;
          if (keep_prelast->x > prelast->x) {
            x1 = int((prelast->x + delta) >> XY_SHIFT); x2 = int(keep_prelast->x >> XY_SHIFT);
          } else {
            x1 = int((keep_prelast->x + delta) >> XY_SHIFT); x2 = int(prelast->x >> XY_SHIFT);
          }
          if (x1 < w && x2 >= 0) {
            if (x1 < 0) x1 = 0;
            if (x2 >= w) x2 = w - 1;
            for (int x = x1; x <= x2; ++x) img[std::size_t(y) * w + x] = 1;
          }
        }
        keep_prelast->x += keep_prelast->dx;
        prelast->x += prelast->dx;
      }
      draw ^= 1;
    }
    keep_prelast = nullptr;
    do {
      prelast = &tmp;
      last = tmp.next;
      PolyEdge* last_exchange = nullptr;
      while (last != keep_prelast && last->next != nullptr) {
        PolyEdge* te = last->next;
        if (last->x > te->x) {
          prelast->next = te; last->next = te->next; te->next = last; prelast = te;
          last_exchange = prelast;
        } else { prelast = last; last = te; }
      }
      if (last_exchange == nullptr) break;
      keep_prelast = last_exchange;
    } while (keep_prelast != tmp.next && keep_prelast != &tmp);
  }
}

struct Scratch {
  std::vector<IPt> hull;
  std::vector<FPt> fhull;
  std::vector<const IPt*> pointer;
  std::vector<int> stack, hullbuf;
  std::vector<float> abuf;
};

// cv::minAreaRect on int32 points.
inline RRect MinAreaRect(const IPt* pts, int total, Scratch& s) {
  RRect box{{0.f, 0.f}, 0.f, 0.f, 0.f};
  double angle = -kPi / 2;
  ConvexHull(pts, total, s.hull, s.pointer, s.stack, s.hullbuf);
  const int n = int(s.hull.size());
  s.fhull.resize(s.hull.size());
  for (int i = 0; i < n; ++i) s.fhull[i] = FPt{float(s.hull[i].x), float(s.hull[i].y)};
  const FPt* hp = s.fhull.data();
  if (n > 2) {
    float o[6];
    RotatingCalipersMinArea(hp, n, o, s.abuf);
    box.center.x = o[0] + (o[2] + o[4]) * 0.5f;
    box.center.y = o[1] + (o[3] + o[5]) * 0.5f;
    box.w = float(std::sqrt(double(o[4]) * o[4] + double(o[5]) * o[5]));
    box.h = float(std::sqrt(double(o[2]) * o[2] + double(o[3]) * o[3]));
    if (o[2] == 0.f && o[3] > 0.f) std::swap(box.w, box.h);
    else angle = -std::atan2(double(o[2]), double(o[3]));
  } else if (n == 2) {
    box.center.x = (hp[0].x + hp[1].x) * 0.5f;
    box.center.y = (hp[0].y + hp[1].y) * 0.5f;
    const double dx = double(hp[0].x) - hp[1].x, dy = double(hp[0].y) - hp[1].y;
    box.w = 0;
    box.h = float(std::sqrt(dx * dx + dy * dy));
    if (dx == 0) std::swap(box.w, box.h);
    else if (dy < 0) { angle = std::atan2(dy, dx); std::swap(box.w, box.h); }
    else if (dy > 0) angle = -std::atan2(dx, dy);
  } else if (n == 1) {
    box.center = hp[0];
  }
  box.angle = float(angle * 180 / kPi);
  return box;
}

inline std::array<FPt, 4> BoxPoints(const RRect& r) {
  const double a_ = double(r.angle) * kPi / 180.;
  const float b = float(std::cos(a_)) * 0.5f;
  const float a = float(std::sin(a_)) * 0.5f;
  const float ah = a * r.h, aw = a * r.w, bh = b * r.h, bw = b * r.w;
  std::array<FPt, 4> pt{};
  pt[0] = {r.center.x - ah - bw, r.center.y + bh - aw};
  pt[1] = {r.center.x + ah - bw, r.center.y - bh - aw};
  pt[2] = {r.center.x + ah + bw, r.center.y - bh + aw};
  pt[3] = {r.center.x - ah + bw, r.center.y + bh + aw};
  return pt;
}

}}}  // namespace ppocr::detail::cvgeom

namespace ppocr { namespace detail {

namespace {

using namespace cvgeom;

// get_mini_boxes: sorted(boxPoints, key=x) (stable), then pick by y.
std::array<FPt, 4> MiniBoxOrder(std::array<FPt, 4> p) {
  std::stable_sort(p.begin(), p.end(), [](const FPt& a, const FPt& b) { return a.x < b.x; });
  FPt i1, i2, i3, i4;
  if (p[1].y > p[0].y) { i1 = p[0]; i4 = p[1]; } else { i1 = p[1]; i4 = p[0]; }
  if (p[3].y > p[2].y) { i2 = p[2]; i3 = p[3]; } else { i2 = p[3]; i3 = p[2]; }
  return {i1, i2, i3, i4};
}

struct DbScratch {
  Scratch geom;
  std::vector<IPt> points;
  std::vector<std::uint8_t> mask;
  std::vector<PolyEdge> edges;
  ClipperLib::Path path;
  ClipperLib::Paths solution;
};

// box_score_fast: cv2.fillPoly of the int32-truncated box, then cv2.mean.
double BoxScoreFast(const float* prob, int h, int w, const std::array<FPt, 4>& box, DbScratch& s) {
  float fxmin = box[0].x, fxmax = box[0].x, fymin = box[0].y, fymax = box[0].y;
  for (const auto& p : box) {
    fxmin = std::min(fxmin, p.x); fxmax = std::max(fxmax, p.x);
    fymin = std::min(fymin, p.y); fymax = std::max(fymax, p.y);
  }
  const int xmin = std::clamp(int(std::floor(fxmin)), 0, w - 1);
  const int xmax = std::clamp(int(std::ceil(fxmax)), 0, w - 1);
  const int ymin = std::clamp(int(std::floor(fymin)), 0, h - 1);
  const int ymax = std::clamp(int(std::ceil(fymax)), 0, h - 1);
  const int mw = xmax - xmin + 1, mh = ymax - ymin + 1;
  s.mask.assign(std::size_t(mw) * mh, 0);
  IPt q[4];
  for (int i = 0; i < 4; ++i) q[i] = {int(box[i].x - float(xmin)), int(box[i].y - float(ymin))};
  FillPoly(s.mask.data(), mw, mh, q, 4, s.edges);
  double sum = 0.0;
  std::size_t count = 0;
  for (int y = 0; y < mh; ++y) {
    const std::uint8_t* m = s.mask.data() + std::size_t(y) * mw;
    const float* row = prob + std::size_t(y + ymin) * w + xmin;
    for (int x = 0; x < mw; ++x) {
      if (m[x]) { sum += row[x]; ++count; }
    }
  }
  return count == 0 ? 0.0 : sum / double(count);
}

}  // namespace

double ShortestDouble(float value) {
  // Options store float; rapidocr's YAML values are Python floats. Recover
  // the decimal the float was parsed from (0.5f -> 0.5, 1.6f -> 1.6).
  char buf[32];
  for (int digits = 1; digits <= 9; ++digits) {
    std::snprintf(buf, sizeof(buf), "%.*g", digits, double(value));
    const double d = std::strtod(buf, nullptr);
    if (float(d) == value) return d;
  }
  return double(value);
}

bool Rapidocr3DbBox(const int* contour_xy, int count, const float* prob, int h, int w,
                    int dst_w, int dst_h, double box_thresh, double unclip_ratio,
                    float out_xy[8], float* score) {
  thread_local DbScratch s;
  constexpr float kMinSize = 3.F;
  // get_mini_boxes(contour)
  const RRect rect = MinAreaRect(reinterpret_cast<const IPt*>(contour_xy), count, s.geom);
  if (std::min(rect.w, rect.h) < kMinSize) return false;
  const auto mini = MiniBoxOrder(BoxPoints(rect));
  const double box_score = BoxScoreFast(prob, h, w, mini, s);
  if (box_thresh > box_score) return false;
  // unclip: shapely Polygon area/length, pyclipper JT_ROUND offset of the
  // int-truncated box (miter limit 2, arc tolerance 0.25).
  double area = 0.0, length = 0.0;
  for (int i = 0; i < 4; ++i) {
    const FPt& a = mini[i];
    const FPt& b = mini[(i + 1) % 4];
    area += double(a.x) * double(b.y) - double(b.x) * double(a.y);
    const double ex = double(b.x) - double(a.x), ey = double(b.y) - double(a.y);
    length += std::sqrt(ex * ex + ey * ey);
  }
  area = std::fabs(area) * 0.5;
  const double distance = area * unclip_ratio / length;
  s.path.clear();
  for (const auto& p : mini) {
    s.path.push_back(ClipperLib::IntPoint(ClipperLib::cInt(p.x), ClipperLib::cInt(p.y)));
  }
  ClipperLib::ClipperOffset offset(2.0, 0.25);
  offset.AddPath(s.path, ClipperLib::jtRound, ClipperLib::etClosedPolygon);
  s.solution.clear();
  offset.Execute(s.solution, distance);
  s.points.clear();
  for (const auto& poly : s.solution) {
    for (const auto& p : poly) s.points.push_back(IPt{int(p.X), int(p.Y)});
  }
  if (s.points.empty()) return false;
  const RRect grown = MinAreaRect(s.points.data(), int(s.points.size()), s.geom);
  if (std::min(grown.w, grown.h) < kMinSize + 2.F) return false;
  auto box = MiniBoxOrder(BoxPoints(grown));
  // np.clip(np.round(x / width * dest_width), 0, dest_width) in float32,
  // astype(int32), order_points_clockwise, clip_det_res.
  std::array<FPt, 4> q{};
  for (int i = 0; i < 4; ++i) {
    float x = std::nearbyint(box[i].x / float(w) * float(dst_w));
    float y = std::nearbyint(box[i].y / float(h) * float(dst_h));
    x = std::min(std::max(x, 0.F), float(dst_w));
    y = std::min(std::max(y, 0.F), float(dst_h));
    q[i] = FPt{float(int(x)), float(int(y))};
  }
  std::stable_sort(q.begin(), q.end(), [](const FPt& a, const FPt& b) { return a.x < b.x; });
  FPt l0 = q[0], l1 = q[1], r0 = q[2], r1 = q[3];
  if (l1.y < l0.y) std::swap(l0, l1);
  if (r1.y < r0.y) std::swap(r0, r1);
  std::array<FPt, 4> o{l0, r0, r1, l1};
  for (auto& p : o) {
    p.x = std::min(std::max(p.x, 0.F), float(dst_w - 1));
    p.y = std::min(std::max(p.y, 0.F), float(dst_h - 1));
  }
  const auto norm = [](const FPt& a, const FPt& b) {
    const float dx = a.x - b.x, dy = a.y - b.y;
    return int(std::sqrt(dx * dx + dy * dy));
  };
  if (norm(o[0], o[1]) <= 3 || norm(o[0], o[3]) <= 3) return false;
  for (int i = 0; i < 4; ++i) { out_xy[2 * i] = o[i].x; out_xy[2 * i + 1] = o[i].y; }
  *score = float(box_score);
  return true;
}

}}  // namespace ppocr::detail
