#!/usr/bin/env python3
"""Run under rapidocr main (with OpenCV): RapidOCRLite's cal_rec_boxes port
must equal main's CTCLabelDecode.get_word_info / CalRecBoxes and OpenCV's
getPerspectiveTransform / invert bit for bit on random inputs."""
import importlib.util
import sys

import cv2
import numpy as np
from rapidocr.cal_rec_boxes import CalRecBoxes as MainCal
from rapidocr.ch_ppocr_rec.typings import WordInfo as MainWordInfo
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode

spec = importlib.util.spec_from_file_location("lite_crb", sys.argv[1])
lite = importlib.util.module_from_spec(spec)
sys.modules["lite_crb"] = lite
spec.loader.exec_module(lite)
rng = np.random.default_rng(1234)
bad = 0
n = 0
for t in range(20000):
    if t % 2:
        x, y = rng.integers(0, 2000, 2)
        w, h = rng.integers(5, 900, 2)
        src = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
        if t % 4 == 1:
            src = src + rng.integers(-3, 4, (4, 2))
        src = src.astype(np.float32)
    else:
        src = np.round(rng.random((4, 2)) * 1500).astype(np.float32)
    W = int(np.linalg.norm(src[0] - src[1]))
    H = int(np.linalg.norm(src[0] - src[3]))
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    try:
        M = cv2.getPerspectiveTransform(src, dst)
        M2 = lite.get_perspective_transform(src, dst)
    except Exception:
        continue
    n += 1
    _, IM = cv2.invert(M)
    if not (np.array_equal(M, M2) and np.array_equal(IM, lite.invert3x3(M))):
        bad += 1
print(f"perspective/invert: {n - bad}/{n} identical")

chars = list("abcXYZ019 中文字，。") + ["é"]
mism = 0
for t in range(5000):
    steps = int(rng.integers(10, 160))
    k = int(rng.integers(0, min(steps, 30)))
    cols = sorted(rng.choice(steps, size=k, replace=False).tolist())
    text = "".join(rng.choice(chars) for _ in range(k))
    probs = rng.random(k).astype(np.float32).tolist()
    wh, max_wh = float(rng.uniform(0.5, 30)), float(rng.uniform(6.67, 40))
    sel = np.zeros(steps, bool)
    sel[cols] = True
    ref = CTCLabelDecode.get_word_info(text, sel)
    conf = [round(c, 5) for c in probs] or [0]
    ref.line_txt_len = steps * wh / max_wh
    ref.confs = conf
    # main #750's CTCLabelDecode.decode: each word's own character confs.
    char_confs = iter(c for ch, c in zip(text, conf) if not ch.isspace())
    ref.char_confs = [[next(char_confs) for _ in word] for word in ref.words]
    got = lite.build_word_info(text, cols, probs, steps, wh, max_wh)
    same = (ref.words == got.words and ref.word_cols == got.word_cols
            and [x.value for x in ref.word_types] == [x.value for x in got.word_types]
            and ref.line_txt_len == got.line_txt_len and ref.confs == got.confs
            and ref.char_confs == got.char_confs)
    h, w = int(rng.integers(8, 120)), int(rng.integers(8, 1500))
    box = (rng.random((4, 2)) * 1000).astype(np.float32)
    if t % 3 == 0:  # vertical crop
        box = np.array([[10, 10], [40, 10], [40, 400], [10, 400]], np.float32)
    rotated = [bool(t % 5 == 2)]  # main #747: crop turned 180 by the classifier
    for single in (False, True):
        img = np.zeros((h, w, 3), np.uint8)
        try:
            r = MainCal()([img], box[None], type("R", (), {"txts": (text,), "word_results": (ref,)})(), single,
                          rotated).word_results
        except Exception as e:  # noqa: BLE001
            r = repr(e)
        try:
            g = lite.CalRecBoxes()([(h, w)], box[None], (text,), [got], single, rotated)
        except Exception as e:  # noqa: BLE001
            g = repr(e)
        if (r if isinstance(r, str) else [list(x) for x in r]) != (g if isinstance(g, str) else [list(x) for x in g]):
            same = False
    mism += not same
print(f"word info + CalRecBoxes: {5000 - mism}/5000 identical")
sys.exit(1 if (bad or mism) else 0)
