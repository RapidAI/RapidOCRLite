#!/usr/bin/env python3
"""Geometry checks for a word-box dump of a model with no rapidocr main
counterpart (model_type=lite): every kept line has word boxes, the
single-character boxes spell the line text (spaces removed), every box lies
within its line box (2 px slack) and boxes run left to right.

  check_words_sane.py DUMP.json [--min-ok 0.98]
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--min-ok", type=float, default=0.98)
a = ap.parse_args()
d = json.load(open(a.dump, encoding="utf-8"))
lines = ok = spelled = inside = ordered = with_words = 0
for key, page in d["pages"].items():
    if "error" in page:
        print(key, page["error"])
        continue
    for mode in ("word", "char"):
        m = page[mode]
        words = m["words"] or []
        for i, txt in enumerate(m["txts"]):
            if not txt.strip():
                continue
            lines += 1
            wl = words[i] if i < len(words) else []
            if wl:
                with_words += 1
            xs = [p[0] for p in m["boxes"][i]]
            ys = [p[1] for p in m["boxes"][i]]
            a_in = all(min(xs) - 2 <= x <= max(xs) + 2 and min(ys) - 2 <= y <= max(ys) + 2
                       for w in wl for x, y in w[2])
            lefts = [min(p[0] for p in w[2]) for w in wl]
            a_ord = lefts == sorted(lefts)
            a_sp = mode == "word" or "".join(w[0] for w in wl) == "".join(txt.split())
            inside += a_in
            ordered += a_ord
            spelled += a_sp
            ok += bool(wl) and a_in and a_ord and a_sp
rec = d.get("rec", {})
rec_infos = sum(1 for r in rec.values() for x in r["infos"] if x and x["words"])
frac = ok / lines if lines else 0.0
print(f"lite model word boxes: {lines} line-modes, with boxes {with_words}, inside line {inside}, "
      f"left-to-right {ordered}, chars spell text {spelled}; all checks {ok} ({frac:.2%}); "
      f"use_det=False WordInfo with words: {rec_infos}/{len(rec)}")
raise SystemExit(0 if lines and frac >= a.min_ok else 1)
