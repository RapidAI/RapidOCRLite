#!/usr/bin/env python3
"""Word boxes of another backend (strict Vulkan GPU-only graph) against the
CPU dump of the same models (eval_words.py JSON). Lines are paired by
identical text and line box; for those, the word lists must have the same
texts and every word box corner must lie within --tol px.

  compare_backends.py CPU.json OTHER.json [--tol 3] [--min-same 0.98]
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("cpu")
ap.add_argument("other")
ap.add_argument("--tol", type=float, default=3.0)
ap.add_argument("--min-same", type=float, default=0.98)
a = ap.parse_args()
A = json.load(open(a.cpu))["pages"]
B = json.load(open(a.other))["pages"]
paired = same_txt = within = exact = words = 0
max_dev = 0.0
for key, pa in A.items():
    pb = B.get(key)
    if not pb or "error" in pa or "error" in pb:
        continue
    for mode in ("word", "char"):
        ma, mb = pa[mode], pb[mode]
        if not ma["words"] or not mb["words"]:
            continue
        idx_b = {}
        for j, (t, bx) in enumerate(zip(mb["txts"], mb["boxes"])):
            idx_b.setdefault((t, json.dumps(bx)), j)
        for i, (t, bx) in enumerate(zip(ma["txts"], ma["boxes"])):
            j = idx_b.get((t, json.dumps(bx)))
            if j is None or i >= len(ma["words"]) or j >= len(mb["words"]):
                continue
            paired += 1
            wa, wb = ma["words"][i], mb["words"][j]
            if [w[0] for w in wa] != [w[0] for w in wb]:
                continue
            same_txt += 1
            dev = 0.0
            for x, y in zip(wa, wb):
                for p, q in zip(x[2], y[2]):
                    dev = max(dev, abs(p[0] - q[0]), abs(p[1] - q[1]))
            words += len(wa)
            max_dev = max(max_dev, dev)
            within += dev <= a.tol
            exact += dev == 0
frac = same_txt / paired if paired else 0.0
print(f"lines with identical text and box: {paired}; same word texts: {same_txt} ({frac:.2%}); "
      f"boxes within {a.tol:g}px: {within}, identical: {exact}; words {words}; max corner deviation {max_dev:g}px")
ok = paired > 0 and frac >= a.min_same and within == same_txt
raise SystemExit(0 if ok else 1)
