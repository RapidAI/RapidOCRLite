"""Word boxes on lines the classifier turned by 180 degrees (rapidocr main
#747, issue #328). Model-free: drives the CalRecBoxes port directly with the
same synthetic lines as main's tests/test_word_box_cls_rotation.py. The
native (C / C++ / Java) path is compared with this port on real pages by the
word-box-parity workflow."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pytest

from rapidocr.cal_rec_boxes import CalRecBoxes, WordInfo, WordType

LINE_TXT = "我是人"

# A 200x20 horizontal line and a 20x200 vertical line in a 400x400 image.
HORIZONTAL_BOX = [[100, 50], [300, 50], [300, 70], [100, 70]]
VERTICAL_BOX = [[100, 100], [120, 100], [120, 300], [100, 300]]


def make_word_info() -> WordInfo:
    return WordInfo(
        words=[list(LINE_TXT)],
        word_cols=[[2, 15, 28]],
        word_types=[WordType.CN],
        line_txt_len=30,
        confs=[0.9, 0.9, 0.9],
    )


def run_word_boxes(boxes, rotated: Optional[List[bool]]):
    # The crop handed to recognition is always laid out horizontally.
    crop_sizes = [(20, 200) for _ in boxes]
    return CalRecBoxes()(
        crop_sizes,
        np.array(boxes, dtype=np.float32),
        tuple(LINE_TXT for _ in boxes),
        [make_word_info() for _ in boxes],
        False,
        rotated,
    )


def word_centers(word_line) -> List[Tuple[float, float]]:
    return [tuple(np.array(points).mean(axis=0)) for _, _, points in word_line]


@pytest.mark.parametrize(
    "box,rotated,axis,reading_direction",
    [
        (HORIZONTAL_BOX, None, 0, 1),
        (HORIZONTAL_BOX, [False], 0, 1),
        # issue #328: upside-down text is read right to left in the image
        (HORIZONTAL_BOX, [True], 0, -1),
        (VERTICAL_BOX, [False], 1, 1),
        (VERTICAL_BOX, [True], 1, -1),
    ],
)
def test_word_boxes_follow_cls_rotation(box, rotated, axis, reading_direction):
    word_line = run_word_boxes([box], rotated)[0]
    assert "".join(txt for txt, _, _ in word_line) == LINE_TXT
    centers = [center[axis] for center in word_centers(word_line)]
    assert np.all(reading_direction * np.diff(centers) > 0)


def test_rotated_word_boxes_are_mirrored_within_line():
    upright = run_word_boxes([HORIZONTAL_BOX], [False])[0]
    flipped = run_word_boxes([HORIZONTAL_BOX], [True])[0]
    upright_xs = [sorted({p[0] for p in points}) for _, _, points in upright]
    flipped_xs = [sorted({p[0] for p in points}) for _, _, points in flipped]
    # Each character's box is mirrored about the centre of the line.
    for (x0, x1), (fx0, fx1) in zip(upright_xs, flipped_xs):
        assert (fx0, fx1) == (400 - x1, 400 - x0)
    # main's PR #747 numbers: 我 240-300, 是 154-240, 人 100-154.
    assert flipped_xs == [[240, 300], [154, 240], [100, 154]]


def test_flags_follow_their_lines():
    lines = run_word_boxes([HORIZONTAL_BOX, VERTICAL_BOX], [False, True])
    xs = [c[0] for c in word_centers(lines[0])]
    ys = [c[1] for c in word_centers(lines[1])]
    assert np.all(np.diff(xs) > 0)
    assert np.all(np.diff(ys) < 0)
