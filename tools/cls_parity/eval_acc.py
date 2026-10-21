#!/usr/bin/env python3
"""Run whichever `rapidocr` is importable (RapidOCRLite or rapidocr main) on
the accuracy sets used in README (SWHL/text_det_test_dataset, 212 pages;
SWHL/text_rec_test_dataset, 310 crops) and dump everything to JSON.

  eval_acc.py OUT.json [--use-cls 0|1] [--limit-det N] [--limit-rec N]

Keys: det (det-only boxes), full ([txts, boxes, scores] of det+cls?+rec),
rec ([txt, score] of cls?+rec on each crop), det_gt, rec_gt, times_full,
times_rec (seconds per call), cls_rec (the classifier's label/score on each
rec crop when --use-cls 1, from a use_det=False/use_rec=False call).
"""
import argparse
import json
import time

import numpy as np
from datasets import load_dataset
from rapidocr import RapidOCR

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--use-cls", type=int, default=1)
ap.add_argument("--limit-det", type=int, default=0)
ap.add_argument("--limit-rec", type=int, default=0)
ap.add_argument("--model-root", default=None)
a = ap.parse_args()
params = {"Global.log_level": "error"}
if a.model_root:
    params["Global.model_root_dir"] = a.model_root
eng = RapidOCR(params=params)
cls = bool(a.use_cls)
det = list(load_dataset("SWHL/text_det_test_dataset")["test"])
rec = list(load_dataset("SWHL/text_rec_test_dataset")["test"])
if a.limit_det:
    det = det[: a.limit_det]
if a.limit_rec:
    rec = rec[: a.limit_rec]
out = {
    "use_cls": cls,
    "det": [], "full": [], "rec": [], "cls_rec": [], "times_full": [], "times_rec": [],
    "det_gt": [[x["points"] for x in s["shapes"]] for s in det],
    "rec_gt": [s["label"] for s in rec],
}
bgr = lambda im: np.array(im.convert("RGB"))[:, :, ::-1].copy()
# warm-up
eng(bgr(det[0]["image"]), use_det=True, use_cls=cls, use_rec=True)
for s in det:
    img = bgr(s["image"])
    r = eng(img, use_det=True, use_cls=False, use_rec=False)
    out["det"].append(np.asarray(r.boxes).tolist() if r.boxes is not None else [])
    t = time.perf_counter()
    r = eng(img, use_det=True, use_cls=cls, use_rec=True)
    out["times_full"].append(time.perf_counter() - t)
    out["full"].append([list(r.txts or ()), np.asarray(r.boxes).tolist() if r.boxes is not None else [],
                        [float(x) for x in (r.scores or ())]])
for s in rec:
    img = bgr(s["image"])
    t = time.perf_counter()
    r = eng(img, use_det=False, use_cls=cls, use_rec=True)
    out["times_rec"].append(time.perf_counter() - t)
    out["rec"].append([r.txts[0] if r.txts else "", float(r.scores[0]) if r.scores else 0.0])
    if cls:
        c = eng(img, use_det=False, use_cls=True, use_rec=False)
        out["cls_rec"].append([str(c.cls_res[0][0]), float(c.cls_res[0][1])])
json.dump(out, open(a.out, "w"), ensure_ascii=False)
print("wrote", a.out, "det", len(det), "rec", len(rec))
