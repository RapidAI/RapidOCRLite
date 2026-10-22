#!/usr/bin/env python3
"""Run whichever `rapidocr` is importable (RapidOCRLite or rapidocr main)
with return_word_box on and dump the word / character boxes to JSON.

  eval_words.py OUT.json [--limit-det N] [--limit-rec N] [--extra DIR ...]

Pages: SWHL/text_det_test_dataset (README accuracy set), crops:
SWHL/text_rec_test_dataset (use_det=False, main returns WordInfo there),
plus every image in --extra directories (rapidocr main's
python/tests/test_files word-box pages). For each page and each mode
(word: return_word_box=True; char: + return_single_char_box=True) the dump
holds txts, boxes, scores and word_results as [text, conf, box].
"""
import argparse
import json
from pathlib import Path

import numpy as np
from rapidocr import RapidOCR

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--limit-det", type=int, default=0)
ap.add_argument("--limit-rec", type=int, default=0)
ap.add_argument("--extra", nargs="*", default=[])
ap.add_argument("--model-root", default=None)
ap.add_argument("--no-datasets", action="store_true")
ap.add_argument("--no-rec", action="store_true", help="skip the use_det=False crops")
ap.add_argument("--param", nargs="*", default=[], help="extra RapidOCR params KEY=VALUE")
a = ap.parse_args()
params = {"Global.log_level": "error"}
if a.model_root:
    params["Global.model_root_dir"] = a.model_root
for kv in a.param:
    k, v = kv.split("=", 1)
    params[k] = v
eng = RapidOCR(params=params)


def plain(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def words_of(r):
    wr = getattr(r, "word_results", None)
    if wr is None:
        return None
    if len(r) == 0:  # empty result: keep the default verbatim
        return plain(list(wr))
    return [[[w[0], float(w[1]), plain(w[2])] for w in line] for line in wr]


def dump_full(img):
    out = {}
    for mode, single in (("word", False), ("char", True)):
        r = eng(img, return_word_box=True, return_single_char_box=single)
        out[mode] = {
            "txts": list(r.txts or ()),
            "boxes": plain(r.boxes) if getattr(r, "boxes", None) is not None else [],
            "scores": [float(x) for x in (r.scores or ())],
            "words": words_of(r),
        }
    return out


def dump_rec(img):
    r = eng(img, use_det=False, return_word_box=True)
    infos = []
    for wi in r.word_results or ():
        if hasattr(wi, "words"):
            infos.append({
                "words": wi.words, "word_cols": [list(map(int, c)) for c in wi.word_cols],
                "word_types": [t.value for t in wi.word_types],
                "line_txt_len": float(wi.line_txt_len), "confs": [float(c) for c in wi.confs],
            })
        else:
            infos.append(None)
    return {"txts": list(r.txts or ()), "infos": infos}


bgr = lambda im: np.array(im.convert("RGB"))[:, :, ::-1].copy()
out = {"pages": {}, "rec": {}}
for d in a.extra:
    for p in sorted(Path(d).iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            try:
                out["pages"][f"extra/{p.name}"] = dump_full(str(p))
            except Exception as e:  # noqa: BLE001 - recorded for the compare step
                out["pages"][f"extra/{p.name}"] = {"error": repr(e)}
if not a.no_datasets:
    from datasets import load_dataset

    det = list(load_dataset("SWHL/text_det_test_dataset")["test"])
    rec = list(load_dataset("SWHL/text_rec_test_dataset")["test"])
    if a.limit_det:
        det = det[: a.limit_det]
    if a.limit_rec:
        rec = rec[: a.limit_rec]
    for i, s in enumerate(det):
        out["pages"][f"det/{i}"] = dump_full(bgr(s["image"]))
    for i, s in enumerate([] if a.no_rec else rec):
        out["rec"][f"rec/{i}"] = dump_rec(bgr(s["image"]))
json.dump(out, open(a.out, "w"), ensure_ascii=False)
print("wrote", a.out, "pages", len(out["pages"]), "rec", len(out["rec"]))
