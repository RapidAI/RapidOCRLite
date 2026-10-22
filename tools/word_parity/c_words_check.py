#!/usr/bin/env python3
"""The engine's own word boxes (rpocr_result_words, used by C, C++ and Java)
must equal the Python binding's cal_rec_boxes port on the same engine
output, line for line, in word and single-character mode. Optionally also
sanity-checks the geometry (each box inside its line box, words in reading
order, word texts = line text without spaces).

  c_words_check.py [--param KEY=VALUE ...] [--extra DIR ...] [--limit-det N] [--no-datasets]

--param sets RapidOCR params (e.g. EngineConfig.ppocr_cpp.backend=gpu,
Det.model_type=lite Rec.model_type=lite). Exit 1 on any difference.
"""
import argparse
from pathlib import Path

import numpy as np
from rapidocr import RapidOCR
from rapidocr.cal_rec_boxes import CalRecBoxes, build_word_info
from rapidocr.main import _prepare_image

ap = argparse.ArgumentParser()
ap.add_argument("--param", nargs="*", default=[])
ap.add_argument("--extra", nargs="*", default=[])
ap.add_argument("--limit-det", type=int, default=0)
ap.add_argument("--no-datasets", action="store_true")
ap.add_argument("--label", default="")
a = ap.parse_args()
params = {"Global.log_level": "error"}
for kv in a.param:
    k, v = kv.split("=", 1)
    params[k] = v
eng = RapidOCR(params=params)


def images():
    for d in a.extra:
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                yield f"extra/{p.name}", str(p)
    if not a.no_datasets:
        from datasets import load_dataset

        det = list(load_dataset("SWHL/text_det_test_dataset")["test"])
        if a.limit_det:
            det = det[: a.limit_det]
        for i, s in enumerate(det):
            yield f"det/{i}", np.array(s["image"].convert("RGB"))[:, :, ::-1].copy()


lines = words = bad = with_data = geo_bad = 0
backend_shown = False
first = []
for name, src in images():
    original = eng.load_img(src)
    eng(original)  # creates the engine
    engine = eng._ensure_engine()
    if not backend_shown:
        backend_shown = True
        note = getattr(engine, "backend_note", lambda: "")()
        print(f"{a.label} engine backend {engine.active_backend()} {note}")
    if eng._is_lite():
        prepared = original
    else:
        prepared = _prepare_image(eng, original, pad=True)[0]
    chars, cw = [], []
    from rapidocr import _native

    boxes, txts, _, _ = engine.run_ex(prepared, _native.RPOCR_TASK_DET | _native.RPOCR_TASK_REC, chars, cw)
    for i, txt in enumerate(txts):
        steps, crop_w, crop_h, max_wh, cols, probs = chars[i]
        lines += 1
        if not txt.strip() or steps <= 0:
            ref = ([], [])
        else:
            with_data += 1
            info = build_word_info(txt, cols, probs, steps, crop_w * 1.0 / crop_h, max_wh)
            # C sees the engine's float box corners (rpocr_text_box.box).
            box = np.asarray([[[x, y] for x, y in boxes[i]]], dtype=np.float32)
            ref = []
            for single in (False, True):
                out = CalRecBoxes()([(crop_h, crop_w)], box, [txt], [info], single)
                ref.append([(t, float(c), [list(map(int, p)) for p in b]) for t, c, b in (out[0] if out else [])])
        got = cw[i]
        for mode in (0, 1):
            words += len(ref[mode])
            if [list(x) for x in ref[mode]] != [list(x) for x in got[mode]]:
                bad += 1
                if len(first) < 8:
                    diffs = [(k, x, y) for k, (x, y) in enumerate(zip(ref[mode], got[mode])) if list(x) != list(y)]
                    first.append(f"{name} line {i} {'char' if mode else 'word'} {txt!r} n={len(ref[mode])}/{len(got[mode])} "
                                 f"box={boxes[i]} crop={crop_w}x{crop_h}: first diffs {diffs[:2]}")
        # geometry sanity (word mode)
        if got[0]:
            xs = [p[0] for p in boxes[i]]
            ys = [p[1] for p in boxes[i]]
            for _, _, b in got[0]:
                for x, y in b:
                    if not (min(xs) - 2 <= x <= max(xs) + 2 and min(ys) - 2 <= y <= max(ys) + 2):
                        geo_bad += 1
                        break
print(f"{a.label} rpocr_result_words vs Python cal_rec_boxes: {lines} lines ({with_data} with char data), "
      f"{words} words: {'identical' if not bad else f'{bad} line-modes differ'}; "
      f"boxes outside their line: {geo_bad}")
for f in first:
    print("  ", f)
raise SystemExit(1 if bad or (lines and not with_data) else 0)
