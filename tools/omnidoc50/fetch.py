"""Fetch and prepare the 50 OmniDocBench pages of the lite regression (research use only).

The pages are NOT stored in this repository. This script downloads them at run time from the
original source (Hugging Face dataset opendatalab/OmniDocBench, pinned revision), applies the
same preparation as the benchmark (non-text regions painted white, as in build_zh_print.py) and
writes binary PPMs plus list.txt for ppocr_lite_c_test. Every source image and every prepared
PPM is checked against the SHA256 in pages.json; any mismatch aborts.

usage: python fetch.py OUT_DIR       (needs huggingface_hub, numpy, opencv-python-headless, pillow)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

REPO, REV = "opendatalab/OmniDocBench", "aa1ee96d106dbe53d0ae59474d75c6e6d9b53fec"
NONTEXT = {"table", "equation_isolated", "figure", "table_mask", "text_mask", "chart_mask", "unknown_mask",
           "organic_chemical_formula_mask", "equation_semantic", "code_txt", "equation_caption"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def poly(c):
    return np.array(c["poly"], dtype=np.float32).reshape(-1, 2)


def main():
    out = Path(sys.argv[1]).resolve()
    cache = out / "hf"
    out.mkdir(parents=True, exist_ok=True)
    pages = json.loads((Path(__file__).parent / "pages.json").read_text(encoding="utf-8"))
    ann = json.loads(Path(hf_hub_download(REPO, "OmniDocBench.json", repo_type="dataset", revision=REV,
                                          local_dir=str(cache))).read_text(encoding="utf-8"))
    by_img = {x["page_info"]["image_path"]: x for x in ann}
    lines, bad = [], []
    for p in pages:
        src = hf_hub_download(REPO, "images/" + p["src"], repo_type="dataset", revision=REV, local_dir=str(cache))
        if sha(src) != p["src_sha256"]:
            bad.append(f"{p['id']} source sha256 mismatch")
            continue
        x = by_img[p["src"]]
        img = cv2.imread(src)
        for c in sorted(x["layout_dets"], key=lambda c: (c.get("order") is None, c.get("order") or 0)):
            t = c.get("text", "") or ""
            if c["category_type"] in NONTEXT or "$" in t or c.get("ignore"):
                cv2.fillPoly(img, [poly(c).astype(np.int32)], (255, 255, 255))
        ppm = out / f"{p['id']}.ppm"
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(ppm)
        if sha(ppm) != p["ppm_sha256"]:
            bad.append(f"{p['id']} prepared page sha256 mismatch")
        lines.append(f"{p['id']} {ppm.as_posix()}")
    (out / "list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"pages={len(lines)} verified={len(lines) - len(bad)}")
    if bad:
        print("\n".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
