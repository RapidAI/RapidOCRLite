"""Compare ppocr_lite_c_test output (LITE_SCORES=1) with tools/omnidoc50/ref.json.

Per page: text lines must match exactly, boxes are compared as float32, scores as float32.
Prints identical pages, pages that differ only in scores (with the max score diff), and the
details of every text / box difference. usage: compare.py run.json label [report.md]
"""
import json
import sys
from pathlib import Path

import numpy as np

REF = Path(__file__).parent / "ref.json"


def main():
    run = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    label = sys.argv[2]
    ref = {p["id"]: p for p in json.loads(REF.read_text(encoding="utf-8"))["pages"]}
    got = {p["id"]: p for p in run["pages"]}
    ident, score_only, other, smax = 0, [], [], 0.0
    lines = []
    for pid, r in ref.items():
        g = got.get(pid)
        if g is None:
            other.append(pid); lines.append(f"- page {pid}: missing"); continue
        tx = g["txts"] == r["txts"]
        bx = len(g["boxes"]) == len(r["boxes"]) and (np.array(g["boxes"], np.float32) == np.array(r["boxes"], np.float32)).all()
        if tx and bx:
            sg, sr = np.array(g["scores"], np.float32), np.array(r["scores"], np.float32)
            if (sg == sr).all():
                ident += 1
            else:
                d = float(np.abs(sg.astype(np.float64) - sr).max()); smax = max(smax, d)
                score_only.append((pid, int((sg != sr).sum()), d))
            continue
        other.append(pid)
        n = max(len(g["txts"]), len(r["txts"]))
        for i in range(n):
            gt = g["txts"][i] if i < len(g["txts"]) else None
            rt = r["txts"][i] if i < len(r["txts"]) else None
            gb = g["boxes"][i] if i < len(g["boxes"]) else None
            rb = r["boxes"][i] if i < len(r["boxes"]) else None
            if gt != rt or (gb is None) != (rb is None) or (gb is not None and not (np.array(gb, np.float32) == np.array(rb, np.float32)).all()):
                lines.append(f"- page {pid} line {i + 1}: got {gt!r} {gb} | ref {rt!r} {rb}")
    head = (f"{label}: identical_pages={ident}/{len(ref)} score_only_pages={len(score_only)} "
            f"max_score_diff={smax:.3g} text_or_box_diff_pages={len(other)} mean_ms={run.get('mean_ms', 0):.1f} "
            f"median_ms={run.get('median_ms', 0):.1f}")
    print(head)
    body = [f"## {label}", "", head, ""]
    if score_only:
        body += ["score-only pages (id, lines, max diff): " + ", ".join(f"{p}:{n}:{d:.3g}" for p, n, d in score_only), ""]
    body += lines[:200] + [""]
    if len(sys.argv) > 3:
        with open(sys.argv[3], "a", encoding="utf-8") as f:
            f.write("\n".join(body) + "\n")
    for l in lines[:50]:
        print(l)
    sys.exit(1 if other else 0)  # score-only differences are reported, text / box differences fail


if __name__ == "__main__":
    main()
