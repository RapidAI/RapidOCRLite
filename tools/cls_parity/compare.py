#!/usr/bin/env python3
"""Compare eval_acc.py dumps image by image and print det/rec metrics.

  compare.py REF.json OTHER.json [MORE.json ...]  [--strict]

REF is normally rapidocr main. --strict exits 1 when any OTHER differs from
REF in full-pipeline text (per page), rec text, a cls label on the rec crops,
or box coordinates (> 0.5 px). --box-baseline=A.json,B.json excuses box
diffs on pages where A and B (the same pair with use_cls off) already
differ; --no-box skips the box check (cross-ISA runs: AVX-512 and AVX2
already differ on one det page before this change).
"""
import json
import os
import statistics
import sys
import tempfile

import numpy as np


def det_metric(boxes_list, gts):
    from text_det_metric import TextDetMetric
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for b, g in zip(boxes_list, gts):
        f.write(f"{b}\t{g}\t0\n")
    f.close()
    r = TextDetMetric()(f.name)
    os.unlink(f.name)
    return r


def rec_metric(preds, gts):
    from text_rec_metric import TextRecMetric
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    for p, g in zip(preds, gts):
        f.write(f"{p.replace(chr(9), ' ')}\t{g}\t0\n")
    f.close()
    r = TextRecMetric()(f.name)
    os.unlink(f.name)
    return r


def summary(name, d):
    line = [name, f"use_cls={d.get('use_cls')}"]
    try:
        dm = det_metric(d["det"], d["det_gt"])
        rm = rec_metric([x[0] for x in d["rec"]], d["rec_gt"])
        line.append("det " + str({k: round(v, 4) if isinstance(v, float) else v for k, v in dm.items()}))
        line.append("rec " + str({k: round(v, 4) if isinstance(v, float) else v for k, v in rm.items()}))
    except Exception as exc:  # metrics packages optional
        line.append(f"(metrics unavailable: {exc})")
    exact = sum(p[0] == g for p, g in zip(d["rec"], d["rec_gt"]))
    line.append(f"rec exact {exact}/{len(d['rec'])}")
    if d.get("times_full"):
        line.append(f"full median {statistics.median(d['times_full'])*1000:.1f} ms")
        line.append(f"full total {sum(d['times_full']):.2f} s")
    if d.get("times_rec"):
        line.append(f"rec median {statistics.median(d['times_rec'])*1000:.2f} ms")
    print(" | ".join(line))


def boxdiff(x, y):
    if len(x) != len(y):
        return True
    return bool(x) and float(np.abs(np.array(x) - np.array(y)).max()) > 0.5


def compare(na, a, nb, b, excused=frozenset(), check_box=True):
    n_det = min(len(a["det"]), len(b["det"]))
    n_rec = min(len(a["rec"]), len(b["rec"]))
    dd = [i for i in range(n_det) if boxdiff(a["det"][i], b["det"][i])]
    ft = [i for i in range(n_det) if a["full"][i][0] != b["full"][i][0]]
    fb = [i for i in range(n_det) if boxdiff(a["full"][i][1], b["full"][i][1])]
    rd = [i for i in range(n_rec) if a["rec"][i][0] != b["rec"][i][0]]
    cd = []
    if a.get("cls_rec") and b.get("cls_rec"):
        cd = [i for i in range(min(len(a["cls_rec"]), len(b["cls_rec"])))
              if a["cls_rec"][i][0] != b["cls_rec"][i][0]]
    print(f"== {na} vs {nb}: det box diffs {len(dd)}/{n_det}, full text diffs {len(ft)}/{n_det}, "
          f"full box diffs {len(fb)}/{n_det}, rec diffs {len(rd)}/{n_rec}, cls label diffs {len(cd)}")
    for i in ft[:5]:
        x, y = a["full"][i][0], b["full"][i][0]
        print(f"  full img {i}: only-{na} {sorted(set(x)-set(y))[:3]} only-{nb} {sorted(set(y)-set(x))[:3]}")
    for i in rd[:5]:
        print(f"  rec #{i}: GT {a['rec_gt'][i]!r} {na} {a['rec'][i][0]!r} {nb} {b['rec'][i][0]!r}")
    for i in cd[:5]:
        print(f"  cls #{i}: {na} {a['cls_rec'][i]} {nb} {b['cls_rec'][i]}")
    if a.get("cls_rec") and b.get("cls_rec"):
        sd = [abs(x[1] - y[1]) for x, y in zip(a["cls_rec"], b["cls_rec"])]
        print(f"  cls score |diff| max {max(sd):.2e} median {statistics.median(sd):.2e}")
    hard = [i for i in set(dd) | set(fb) if i not in excused] if check_box else []
    if check_box and (dd or fb):
        print(f"  box diff pages {sorted(set(dd) | set(fb))}, excused by baseline {sorted(excused)}")
    return len(hard) + len(ft) + len(rd) + len(cd)


if __name__ == "__main__":
    strict = "--strict" in sys.argv
    files = [f for f in sys.argv[1:] if not f.startswith("--")]
    check_box = "--no-box" not in sys.argv
    excused = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--box-baseline="):
            x, y = (json.load(open(f)) for f in arg.split("=", 1)[1].split(","))
            excused = {i for i in range(min(len(x["det"]), len(y["det"])))
                       if boxdiff(x["det"][i], y["det"][i]) or boxdiff(x["full"][i][1], y["full"][i][1])}
    ds = {f: json.load(open(f)) for f in files}
    for f, d in ds.items():
        summary(os.path.basename(f), d)
    bad = 0
    for f in files[1:]:
        bad += compare(os.path.basename(files[0]), ds[files[0]], os.path.basename(f), ds[f],
                       frozenset(excused), check_box)
    if strict and bad:
        sys.exit(1)
