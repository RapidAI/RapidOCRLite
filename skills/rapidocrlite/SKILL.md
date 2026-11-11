---
name: rapidocrlite
description: Extract text (with line boxes and confidence scores) from images, screenshots, photos, scanned documents and PDF pages using RapidOCRLite, a fast local PP-OCRv6 OCR engine (Chinese, English, Japanese and 50+ Latin, Cyrillic, Greek and Devanagari languages; CPU, optional Vulkan/Metal GPU). Use when the user asks to OCR, read, transcribe or extract text from an image or scanned PDF, or when text inside a picture is needed. Runs offline after the first model download; no API key.
license: Apache-2.0
metadata:
  homepage: https://github.com/RapidAI/RapidOCRLite
  package: rapidocrlite
---

# RapidOCRLite OCR

Local OCR with RapidOCRLite (PyPI `rapidocrlite`, import name `rapidocr`).
`scripts/ocr.py` in this skill folder wraps the public API and prints JSON.

## Setup (once)

```bash
python -c "import rapidocr" 2>/dev/null || pip install --pre rapidocrlite
```

- `rapidocrlite` and the official `rapidocr` package install the same `rapidocr`
  module. If `pip show rapidocr` lists the official package, `pip uninstall rapidocr` first.
- Python 3.9+, wheels for Linux/Windows/macOS (x86_64, arm64). The first run
  downloads the PP-OCRv6 small models (about 30 MB, SHA256-checked) into the package's
  `models/` folder; later runs are offline.

## Run

Use the script next to this file (replace `<skill-dir>` with this skill's folder):

```bash
python <skill-dir>/scripts/ocr.py image.png                 # JSON
python <skill-dir>/scripts/ocr.py a.png b.jpg scans/ --text # plain text, one block per input
python <skill-dir>/scripts/ocr.py https://example.com/x.png
python <skill-dir>/scripts/ocr.py doc.pdf --pages 1-3       # PDF: needs `pip install pymupdf`
python <skill-dir>/scripts/ocr.py shot.png --word-box       # add word boxes
```

JSON is a list with one object per image (or PDF page):

```json
[{"source": "image.png", "text": "line 1\nline 2",
  "lines": [{"text": "line 1", "score": 0.9876, "box": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]}],
  "elapse": 0.21}]
```

- `box` is the 4-point text-line polygon in pixels (clockwise from top-left).
- `page` (1-based) is added for PDF pages; `words` (per line: text/score/box) with `--word-box`.
- A failed input gives `{"source": ..., "error": "..."}`; exit code 1 if any input failed.
- Logs and download progress go to stderr; stdout holds only the result.

Useful options: `--format text|markdown|json`, `--model-type tiny|small|medium|lite`
(default `small`; `tiny` is fastest, `medium` most accurate), `--no-cls` (skip the
0/180 line-orientation classifier for upright text), `--text-score 0.5`,
`--backend vulkan|metal` (GPU, falls back to CPU), `-r` (recurse into folders).

Without the script, the package CLI prints a result summary plus `text<TAB>score` lines:
`rapidocr image.png` (or `python -m rapidocr image.png`).

## Python API

```python
from rapidocr import RapidOCR

engine = RapidOCR()                      # reuse one engine for many images
result = engine("image.png")             # path, URL, bytes, numpy array or PIL image
print(result.txts, result.scores)        # tuples, top-to-bottom reading order
print(result.to_json())                  # [{"box": ..., "txt": ..., "score": ...}]
print(result.to_markdown())              # layout-aware text
result = engine("image.png", return_word_box=True)   # result.word_results
```

GPU: `RapidOCR(params={"EngineConfig.ppocr_cpp.use_vulkan": True})` (Windows wheels
and Vulkan builds) or `{"EngineConfig.ppocr_cpp.use_metal": True}` (macOS wheels);
both fall back to CPU. Default results are the same as RapidOCR on PP-OCRv6.

## Tips

- Low-resolution or tiny text: upscale the image 2x before OCR. Very large pages are
  resized to `Global.max_side_len` (2000) by default.
- Rotated pages (90/270): rotate the image first; `use_cls` only fixes 0/180 per line.
- Empty `lines` means no text was found above `--text-score`.
- Tables/layout: use `--format markdown` for rough reading order, or the boxes to rebuild columns.
