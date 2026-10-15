#pragma once

namespace ppocr { namespace detail {

// One rapidocr 3.x DB candidate. contour_xy holds `count` int (x, y) pairs
// in cv2.findContours order (only hull vertices are needed). Returns false
// when 3.x drops the candidate (size, score, unclip or final size filter);
// otherwise out_xy gets the clockwise box (tl, tr, br, bl) in destination
// pixels and *score the box_score_fast value.
bool Rapidocr3DbBox(const int* contour_xy, int count, const float* prob, int h, int w,
                    int dst_w, int dst_h, double box_thresh, double unclip_ratio,
                    float out_xy[8], float* score);

// Shortest decimal (as double) that parses back to `value`.
double ShortestDouble(float value);

}}  // namespace ppocr::detail
