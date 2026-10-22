"""return_word_box / return_single_char_box (rapidocr main word boxes).

Runs against the native library (PPOCR_LIBRARY) with the tiny CI models.
Bit-exact parity with rapidocr main on real pages is checked by the
word-box-parity workflow (tools/word_parity); here the output format and
the invariants of main's CalRecBoxes are checked, plus the pure-Python port
on fixed inputs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from rapidocr import RapidOCR
from rapidocr.cal_rec_boxes import (
    CalRecBoxes,
    WordType,
    build_word_info,
    get_perspective_transform,
    invert3x3,
)

DATA = Path(__file__).resolve().parent / "data"


def _engine(**extra):
    det = os.environ.get("PPOCR_TEST_DET")
    rec = os.environ.get("PPOCR_TEST_REC")
    dictionary = os.environ.get("PPOCR_TEST_DICT")
    if not (os.environ.get("PPOCR_LIBRARY") and det and rec and dictionary):
        pytest.skip("set PPOCR_LIBRARY and PPOCR_TEST_DET/REC/DICT")
    if not all(Path(p).is_file() for p in (det, rec, dictionary)):
        pytest.skip("CI models are missing")
    params = {
        "Det.model_path": det,
        "Rec.model_path": rec,
        "Rec.rec_keys_path": dictionary,
        "Det.model_type": "tiny",
        "Rec.model_type": "tiny",
        "Global.use_cls": False,
        "Global.log_level": "error",
    }
    params.update(extra)
    return RapidOCR(params=params)


def test_word_info_groups_like_main():
    # "ab cd" at steps 2,3,(space 5),8,9 of 40; CN chars split per char.
    info = build_word_info("ab cd", [2, 3, 5, 8, 9], [0.9, 0.8, 0.99, 0.7, 0.6], 40, 10.0, 10.0)
    assert info.words == [["a", "b"], ["c", "d"]]
    assert info.word_cols == [[2, 3], [8, 9]]
    assert info.word_types == [WordType.EN_NUM, WordType.EN_NUM]
    assert info.line_txt_len == 40.0
    assert info.confs == [0.9, 0.8, 0.99, 0.7, 0.6]
    cn = build_word_info("中文a", [3, 6, 20], [0.5, 0.5, 0.5], 40, 5.0, 10.0)
    assert cn.words == [["中", "文"], ["a"]]
    assert cn.word_types == [WordType.CN, WordType.EN_NUM]
    assert cn.line_txt_len == 20.0
    empty = build_word_info("", [], [], 40, 1.0, 1.0)
    assert empty.words == [] and empty.confs == [0]


def test_cal_rec_boxes_horizontal_line():
    info = build_word_info("ab cd", [2, 3, 5, 8, 9], [0.9, 0.8, 0.99, 0.7, 0.6], 40, 10.0, 10.0)
    box = np.array([[100, 50], [500, 50], [500, 90], [100, 90]], dtype=np.float32)
    (line,) = CalRecBoxes()([(40, 400)], box[None], ("ab cd",), [info], False)
    assert [w for w, _, _ in line] == ["ab", "cd"]
    for _, _, quad in line:
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        assert min(xs) >= 100 and max(xs) <= 500
        assert min(ys) == 50 and max(ys) == 90
    (chars,) = CalRecBoxes()([(40, 400)], box[None], ("ab cd",), [info], True)
    assert [w for w, _, _ in chars] == ["a", "b", "c", "d"]


def test_perspective_port_identity():
    src = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], dtype=np.float32)
    m = get_perspective_transform(src, src)
    assert np.allclose(m, np.eye(3))
    assert np.allclose(invert3x3(m), np.eye(3))


def _check_words(result):
    assert len(result.word_results) == len(result.txts)
    for txt, line in zip(result.txts, result.word_results):
        assert isinstance(line, tuple) and line
        joined = "".join(w for w, _, _ in line)
        assert joined == "".join(txt.split()) or len(joined) <= len(txt)
        for word, score, box in line:
            assert isinstance(word, str) and word
            assert isinstance(score, float)
            assert len(box) == 4
            assert all(isinstance(v, int) for p in box for v in p)


@pytest.mark.parametrize("name", ["hello.png", "page_mixed.png", "page_zh.png"])
def test_return_word_box(name):
    engine = _engine()
    image = str(DATA / name)
    plain = engine(image)
    words = engine(image, return_word_box=True)
    assert words.txts == plain.txts
    assert np.array_equal(words.boxes, plain.boxes)
    assert words.scores == plain.scores
    _check_words(words)
    chars = engine(image, return_word_box=True, return_single_char_box=True)
    assert chars.txts == plain.txts
    _check_words(chars)
    for line in chars.word_results:
        assert all(len(w) == 1 for w, _, _ in line)
    # The switch is per call, as on main.
    again = engine(image, return_word_box=False)
    assert again.word_results == ()


def test_return_word_box_from_config():
    engine = _engine(**{"Global.return_word_box": True})
    result = engine(str(DATA / "hello.png"))
    _check_words(result)


def test_rec_only_returns_word_info():
    engine = _engine()
    result = engine(str(DATA / "hello.png"), use_det=False, return_word_box=True)
    assert len(result.word_results) == len(result.txts) == 1
    info = result.word_results[0]
    assert hasattr(info, "words") and hasattr(info, "line_txt_len")
    if result.txts[0]:
        assert info.line_txt_len > 0
        assert sum(len(w) for w in info.words) == len("".join(result.txts[0].split()))
