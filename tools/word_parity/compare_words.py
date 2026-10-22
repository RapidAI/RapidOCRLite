#!/usr/bin/env python3
"""Compare two eval_words.py dumps (reference first).

Word text, word boxes, line text and line boxes must be identical; word
confidences may differ by --conf-tol (main rounds each to 5 decimals, so one
ulp of float32 softmax can move the last digit). With --strict a mismatch
exits 1. --subset compares only the
keys the second dump has. --det-1px-pages N (with --strict) tolerates up to N
pages whose text is identical and whose line boxes differ by at most 1px per
coordinate (a det probability pixel flipping at the binary threshold, which
moves rec batching and so word boxes on those pages); text differences and
any larger box change still fail. Dumps may be .gz.
"""
import argparse
import gzip
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("other")
ap.add_argument("--conf-tol", type=float, default=1e-2)
ap.add_argument("--max-word-diffs", type=int, default=0,
                help="words whose text/box may differ (CTC argmax near-ties) before --strict fails")
ap.add_argument("--strict", action="store_true")
ap.add_argument("--det-1px-pages", type=int, default=0)
ap.add_argument("--subset", action="store_true", help="only keys present in the second dump")
a = ap.parse_args()
def load(p):
    return json.load(gzip.open(p, "rt") if p.endswith(".gz") else open(p))


ref, oth = load(a.ref), load(a.other)

stats = {"pages": 0, "page_equal": 0, "words": 0, "word_equal": 0, "conf_max_diff": 0.0}
bad = []
bad_words = []
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
                    if wr[0] != wo[0] or wr[2] != wo[2]:
                        stats["word_box_diff"] = stats.get("word_box_diff", 0) + 1
                        bad_words.append((key, mode, wr, wo))
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
print(f"word text/box differences: {len(bad_words)} (allowed {a.max_word_diffs})")
for w in bad_words[:10]:
    print("WORD", json.dumps(w, ensure_ascii=False))
tol_pages = set()
for key, rp in ref["pages"].items():
    op = oth["pages"].get(key)
    if not op or "error" in rp or "error" in op or (a.subset and key not in oth["pages"]):
        continue
    def near(m):
        r, o = rp[m], op[m]
        if r == o:
            return None
        if r["txts"] != o["txts"] or len(r["boxes"]) != len(o["boxes"]):
            return False
        if [[w[0] for w in l] for l in r["words"]] != [[w[0] for w in l] for l in o["words"]]:
            return False
        return all(abs(x - y) <= 1 for br, bo in zip(r["boxes"], o["boxes"])
                   for pr, po in zip(br, bo) for x, y in zip(pr, po))
    st = [near("word"), near("char")]
    if False not in st and True in st:
        tol_pages.add(key)
if a.det_1px_pages:
    print(f"det 1px threshold pages: {len(tol_pages)} (allowed {a.det_1px_pages}) {sorted(tol_pages)}")
    if len(tol_pages) <= a.det_1px_pages:
        bad = [b for b in bad if b[0] not in tol_pages]
        bad_words = [w for w in bad_words if w[0] not in tol_pages]
line_level = [b for b in bad if b[2] not in ("WordInfo differs",) and not str(b[2]).startswith("txt_eq=True box_eq=True")]
rec_bad = [b for b in bad if b[2] == "WordInfo differs"]
if a.strict and (line_level or rec_bad or len(bad_words) > a.max_word_diffs or any(
        str(b[2]).startswith("txt_eq=True box_eq=True") and "nlines" in str(b[2]) and
        b[2].split("nlines=")[1].split("/")[0] != b[2].split("/")[-1] for b in bad)):
    sys.exit(1)
