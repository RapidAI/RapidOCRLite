#!/usr/bin/env python3
"""Time the orientation classifier's overhead (whichever rapidocr is importable).

  bench.py LABEL OUT.json [--pages N] [--crops N] [--rounds R] [--model-root DIR]

Two engines (use_cls on / off) run interleaved on the same SWHL det pages
and rec crops; per item the median of R rounds is kept. Prints and stores
median per-page full-pipeline ms for each, median rec-crop ms, and the
cls-only call on the crops (use_det=False, use_rec=False).
"""
import argparse
import json
import statistics
import time

import numpy as np
from datasets import load_dataset
from rapidocr import RapidOCR

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("out")
ap.add_argument("--pages", type=int, default=212)
ap.add_argument("--crops", type=int, default=310)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--model-root", default=None)
a = ap.parse_args()
params = {"Global.log_level": "error"}
if a.model_root:
    params["Global.model_root_dir"] = a.model_root
on = RapidOCR(params={**params, "Global.use_cls": True})
off = RapidOCR(params={**params, "Global.use_cls": False})
bgr = lambda im: np.array(im.convert("RGB"))[:, :, ::-1].copy()
pages = [bgr(s["image"]) for s in list(load_dataset("SWHL/text_det_test_dataset")["test"])[: a.pages]]
crops = [bgr(s["image"]) for s in list(load_dataset("SWHL/text_rec_test_dataset")["test"])[: a.crops]]


def timed(fn):
    t = time.perf_counter()
    fn()
    return (time.perf_counter() - t) * 1000


for e in (on, off):  # warm-up
    e(pages[0]); e(crops[0], use_det=False)
on(crops[0], use_det=False, use_rec=False, use_cls=True)
res = {"label": a.label, "full_on": [], "full_off": [], "rec_on": [], "rec_off": [], "cls_only": []}
for img in pages:
    x, y = [], []
    for _ in range(a.rounds):
        x.append(timed(lambda: on(img)))
        y.append(timed(lambda: off(img)))
    res["full_on"].append(statistics.median(x)); res["full_off"].append(statistics.median(y))
for img in crops:
    x, y, z = [], [], []
    for _ in range(a.rounds):
        x.append(timed(lambda: on(img, use_det=False)))
        y.append(timed(lambda: off(img, use_det=False)))
        z.append(timed(lambda: on(img, use_det=False, use_rec=False, use_cls=True)))
    res["rec_on"].append(statistics.median(x)); res["rec_off"].append(statistics.median(y))
    res["cls_only"].append(statistics.median(z))
m = {k: statistics.median(v) for k, v in res.items() if isinstance(v, list)}
res["median"] = m
print(f"{a.label}: full page off {m['full_off']:.1f} ms on {m['full_on']:.1f} ms "
      f"(+{m['full_on'] - m['full_off']:.1f} ms, +{100 * (m['full_on'] / m['full_off'] - 1):.1f}%) | "
      f"rec crop off {m['rec_off']:.2f} on {m['rec_on']:.2f} (+{m['rec_on'] - m['rec_off']:.2f}) | "
      f"cls-only crop {m['cls_only']:.2f} ms")
json.dump(res, open(a.out, "w"))
