#!/usr/bin/env python3
"""Geometry checks for a word-box dump of a model with no rapidocr main
counterpart (model_type=lite). main's build_final_output drops lines whose
word list comes out empty from word_results, so word lists are matched to
lines by geometry, not index: every word list must lie inside some line box
of its page (2 px slack) and run left to right, and in single-character
mode spell that line's text without spaces.

  check_words_sane.py DUMP.json [--min-ok 0.98]
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--min-ok", type=float, default=0.98)
a = ap.parse_args()
d = json.load(open(a.dump, encoding="utf-8"))


def inside(wl, box):
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return all(min(xs) - 2 <= x <= max(xs) + 2 and min(ys) - 2 <= y <= max(ys) + 2 for w in wl for x, y in w[2])


lists = ok = lines = 0
shown = 0
for key, page in d["pages"].items():
    if "error" in page:
        print(key, page["error"])
        continue
    for mode in ("word", "char"):
        m = page[mode]
        lines += sum(1 for t in m["txts"] if t.strip())
        for wl in m["words"] or []:
            # an empty page keeps main's default (("", 1.0, None),)
            wl = [w for w in wl if isinstance(w, list) and len(w) == 3 and w[2]] if isinstance(wl, list) else []
            if not wl:
                continue
            lists += 1
            lefts = [min(p[0] for p in w[2]) for w in wl]
            spelled = "".join(w[0] for w in wl)
            good = lefts == sorted(lefts) and any(
                inside(wl, bx) and (mode == "word" or spelled == "".join(t.split()))
                for t, bx in zip(m["txts"], m["boxes"]))
            ok += good
            if not good and shown < 6:
                shown += 1
                print(f"  {key} {mode} words={wl[:4]}")
rec = d.get("rec", {})
rec_infos = sum(1 for r in rec.values() for x in r["infos"] if x and x["words"])
frac = ok / lists if lists else 0.0
print(f"lite model word boxes: {lines} kept line-modes, {lists} word lists; inside a line, left to right "
      f"and (char mode) spelling its text: {ok} ({frac:.2%}); use_det=False WordInfo with words: {rec_infos}/{len(rec)}")
raise SystemExit(0 if lists and frac >= a.min_ok else 1)
