#!/usr/bin/env python3
"""Java binding word boxes (RapidOcrDemo --words-json) against the Python
binding's cal_rec_boxes on the same engine output: the script runs the same
models through the Python C-API wrapper on the same decoded RGB pixels
(use PNG inputs: JPEG decoders differ) and requires identical lines,
per-character data, word boxes and character boxes.

  java_words_check.py JAVA.json DET REC DICT
"""
import json
import sys

import numpy as np
from PIL import Image
from rapidocr import _native
from rapidocr.cal_rec_boxes import CalRecBoxes, build_word_info

java = json.load(open(sys.argv[1], encoding="utf-8"))
det, rec, dic = sys.argv[2:5]
eng = _native.NativeOCR(det, rec, dic, _native.default_options())


def ref_words(txt, box, ch, single):
    steps, cw, chh, max_wh, cols, probs = ch
    if not txt.strip() or steps <= 0:
        return []
    info = build_word_info(txt, cols, probs, steps, cw * 1.0 / chh, max_wh)
    b = np.asarray([[[x, y] for x, y in box]], dtype=np.float32)
    out = CalRecBoxes()([(chh, cw)], b, [txt], [info], single)
    return [[t, float(c), [list(map(int, p)) for p in bb]] for t, c, bb in (out[0] if out else [])]


lines = words = bad = 0
for spec, jlines in java.items():
    rgb = np.asarray(Image.open(spec).convert("RGB"))
    chars = []
    boxes, txts, _, _ = eng.run_ex(rgb, _native.RPOCR_TASK_DET | _native.RPOCR_TASK_REC, chars)
    if [l["text"] for l in jlines] != list(txts):
        print(f"{spec}: line texts differ between Java and Python")
        bad += 1
        continue
    for i, jl in enumerate(jlines):
        lines += 1
        ch = chars[i]
        if jl["rec_steps"] != ch[0] or jl["crop"] != [ch[1], ch[2]] or jl["cols"] != ch[4] or float(jl["max_wh"]) != ch[3]:
            print(f"{spec} line {i}: per-character data differs")
            bad += 1
        for key, single in (("words", False), ("chars", True)):
            want = ref_words(txts[i], boxes[i], ch, single)
            got = [[w[0], float(w[1]), w[2]] for w in jl[key]]
            words += len(want)
            if got != want:
                bad += 1
                print(f"{spec} line {i} {key} {txts[i]!r}: java {got[:3]} python {want[:3]}")
print(f"Java word boxes vs Python cal_rec_boxes: {lines} lines, {words} words: "
      f"{'identical' if not bad else f'{bad} differences'}")
sys.exit(1 if bad or not lines else 0)
