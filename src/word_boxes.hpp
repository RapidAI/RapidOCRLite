#pragma once
#include <vector>

#include "rapidocrlite/rapidocrlite.hpp"

namespace rpocr::detail {
// WordBoxes with the line box given as float corners (the C ABI's
// rpocr_text_box.box, 0.1 px for the lite model) instead of Result::box.
std::vector<Word> WordBoxesAt(const Result& line, const float box[4][2], bool single_char);
}  // namespace rpocr::detail
