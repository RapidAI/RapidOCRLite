#!/usr/bin/env python3
"""Compare two eval_words.py dumps (reference first).

Word text, word boxes, line text and line boxes must be identical; word
confidences may differ by --conf-tol (main rounds each to 5 decimals, so one
ulp of float32 softmax can move the last digit). With --strict a mismatch
exits 1. --subset compares only the
keys the second dump has.
"""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("other")
ap.add_argument("--conf-tol", type=float, default=2e-5)
ap.add_argument("--strict", action="store_true")
ap.add_argument("--subset", action="store_true", help="only keys present in the second dump")
a = ap.parse_args()
ref, oth = json.load(open(a.ref)), json.load(open(a.other))

stats = {"pages": 0, "page_equal": 0, "words": 0, "word_equal": 0, "conf_max_diff": 0.0}
bad = []
for key, rp in ref["pages"].items():
    if a.subset and key not in oth["pages"]:
        continue
    op = oth["pages"].get(key)
    for mode in ("word", "char"):
        stats["pages"] += 1
        if op is None or "error" in rp or "error" in op:
            if rp != op:
                bad.append((key, mode, "error/missing", rp.get("error"), op and op.get("error")))
            else:
                stats["page_equal"] += 1
            continue
        r, o = rp[mode], op[mode]
        if not r["txts"] or not o["txts"]:
            if r == o:
                stats["page_equal"] += 1
            else:
                bad.append((key, mode, "empty page differs", r, o))
            continue
        ok = r["txts"] == o["txts"] and r["boxes"] == o["boxes"] and len(r["words"]) == len(o["words"])
        for lr, lo in zip(r["words"], o["words"]):
            if len(lr) != len(lo):
                ok = False
            for wr, wo in zip(lr, lo):
                stats["words"] += 1
                d = abs(wr[1] - wo[1])
                stats["conf_max_diff"] = max(stats["conf_max_diff"], d)
                if wr[0] == wo[0] and wr[2] == wo[2] and d <= a.conf_tol:
                    stats["word_equal"] += 1
                else:
                    ok = False
        if ok:
            stats["page_equal"] += 1
        else:
            first = None
            for li, (lr, lo) in enumerate(zip(r["words"], o["words"])):
                if lr != lo:
                    first = (li, lr[:3], lo[:3])
                    break
            bad.append((key, mode, "txt_eq=%s box_eq=%s nlines=%d/%d" % (
                r["txts"] == o["txts"], r["boxes"] == o["boxes"], len(r["words"]), len(o["words"])), first))
rec_total = rec_eq = 0
for key, rr in ref.get("rec", {}).items():
    if a.subset and key not in oth.get("rec", {}):
        continue
    ro = oth.get("rec", {}).get(key)
    rec_total += 1
    same = ro is not None and rr["txts"] == ro["txts"] and len(rr["infos"]) == len(ro["infos"])
    if same:
        for ir, io in zip(rr["infos"], ro["infos"]):
            if (ir is None) != (io is None):
                same = False
                break
            if ir is None:
                continue
            if any(ir[k] != io[k] for k in ("words", "word_cols", "word_types")):
                same = False
            if abs(ir["line_txt_len"] - io["line_txt_len"]) > 1e-9:
                same = False
            if len(ir["confs"]) != len(io["confs"]) or any(abs(x - y) > a.conf_tol for x, y in zip(ir["confs"], io["confs"])):
                same = False
    rec_eq += same
    if not same:
        bad.append((key, "rec", "WordInfo differs", rr, ro))
print(f"pages x modes equal: {stats['page_equal']}/{stats['pages']}")
print(f"words equal (text+box, conf<= {a.conf_tol}): {stats['word_equal']}/{stats['words']}  max conf diff {stats['conf_max_diff']:.2e}")
print(f"rec-only WordInfo equal: {rec_eq}/{rec_total}")
for b in bad[:20]:
    print("DIFF", json.dumps(b, ensure_ascii=False)[:600])
if a.strict and bad:
    sys.exit(1)
