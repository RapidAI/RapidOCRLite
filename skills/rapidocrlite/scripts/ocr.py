#!/usr/bin/env python3
"""OCR images (files, folders, URLs, PDF pages) with RapidOCRLite and print JSON.

Part of the rapidocrlite Agent Skill. Uses only the public ``rapidocr`` API
(``pip install rapidocrlite``), so it works with any rapidocrlite 5.x wheel.

    python ocr.py page.png                    # JSON on stdout
    python ocr.py scans/ --format text        # plain text, one block per image
    python ocr.py doc.pdf --pages 1-3         # PDF pages (needs: pip install pymupdf)
    python ocr.py shot.png --word-box         # add word boxes

stdout carries only the result; logs and download progress go to stderr.
Exit code: 0 ok, 1 some input failed, 2 usage / rapidocrlite not installed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".jfif", ".ppm", ".pgm"}
MODEL_TYPES = ("tiny", "small", "medium", "lite")
BACKENDS = ("cpu", "hybrid", "vulkan", "metal")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ocr.py",
        description="OCR images / folders / URLs / PDF pages with RapidOCRLite (PP-OCRv6) and print JSON or text.",
    )
    p.add_argument("inputs", nargs="+", help="image file, directory, http(s) URL, or .pdf")
    p.add_argument("--format", "-f", choices=("json", "text", "markdown"), default="json",
                   help="json (default): lines with text/score/box; text: plain text; markdown: layout-aware text")
    p.add_argument("--text", action="store_const", dest="format", const="text", help="same as --format text")
    p.add_argument("--model-type", choices=MODEL_TYPES, default=None,
                   help="PP-OCRv6 tiny / small (default) / medium, or lite (bundled Indra OCR)")
    p.add_argument("--lang", default=None, help="Det/Rec lang_type (default ch: the multilingual PP-OCRv6 model)")
    p.add_argument("--text-score", type=float, default=None, help="drop lines below this score (default 0.5)")
    p.add_argument("--no-cls", action="store_true", help="skip the 0/180 text-line orientation classifier")
    p.add_argument("--word-box", action="store_true", help="add per-word boxes (return_word_box)")
    p.add_argument("--char-box", action="store_true", help="with --word-box: one box per character")
    p.add_argument("--backend", choices=BACKENDS, default=None,
                   help="native backend; vulkan/metal fall back to CPU when no GPU is usable")
    p.add_argument("--device-index", type=int, default=None, help="Vulkan physical device index")
    p.add_argument("--recursive", "-r", action="store_true", help="walk directories recursively")
    p.add_argument("--pages", default=None, help="PDF pages, 1-based, e.g. 1,3-5 (default: all)")
    p.add_argument("--dpi", type=int, default=200, help="PDF render resolution (default 200)")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="extra RapidOCR config override, e.g. Global.max_side_len=4000 (repeatable)")
    p.add_argument("--compact", action="store_true", help="single-line JSON")
    return p.parse_intermixed_args(argv)


def parse_pages(spec: Optional[str], count: int) -> List[int]:
    if not spec:
        return list(range(count))
    pages: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            start = int(lo) if lo else 1
            end = int(hi) if hi else count
            pages.extend(range(start - 1, min(end, count)))
        else:
            pages.append(int(part) - 1)
    return [p for p in pages if 0 <= p < count]


def iter_inputs(items: List[str], recursive: bool) -> Iterator[str]:
    for item in items:
        if item.startswith(("http://", "https://")):
            yield item
            continue
        path = Path(item).expanduser()
        if path.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in sorted(path.glob(pattern)):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTS | {".pdf"}:
                    yield str(child)
        else:
            yield str(path)


def iter_units(source: str, args: argparse.Namespace) -> Iterator[Tuple[Dict[str, Any], Any]]:
    """Yield (meta, image) pairs; a PDF yields one pair per page as PNG bytes."""

    if source.lower().endswith(".pdf") and not source.startswith(("http://", "https://")):
        try:
            import pymupdf as fitz  # PyMuPDF >= 1.24
        except ImportError:
            try:
                import fitz  # older PyMuPDF
            except ImportError as exc:
                raise RuntimeError("PDF input needs PyMuPDF: pip install pymupdf "
                                   "(or render pages to PNG first, e.g. pdftoppm -r 200 -png doc.pdf page)") from exc
        with fitz.open(source) as doc:
            for index in parse_pages(args.pages, doc.page_count):
                pix = doc[index].get_pixmap(dpi=args.dpi)
                yield {"source": source, "page": index + 1}, pix.tobytes("png")
        return
    yield {"source": source}, source


def build_params(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {"Global.log_level": "warning"}
    if args.model_type:
        params["Det.model_type"] = args.model_type
        params["Rec.model_type"] = args.model_type
    if args.lang:
        params["Det.lang_type"] = args.lang
        params["Rec.lang_type"] = args.lang
    if args.text_score is not None:
        params["Global.text_score"] = args.text_score
    if args.no_cls:
        params["Global.use_cls"] = False
    if args.backend:
        params["EngineConfig.ppocr_cpp.backend"] = args.backend
        if args.backend == "vulkan":
            params["EngineConfig.ppocr_cpp.use_vulkan"] = True
        if args.backend == "metal":
            params["EngineConfig.ppocr_cpp.use_metal"] = True
    if args.device_index is not None:
        params["EngineConfig.ppocr_cpp.device_index"] = args.device_index
    for item in args.param:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--param expects KEY=VALUE, got {item!r}")
        params[key.strip()] = _scalar(value.strip())
    return params


def _scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _box(box: Any) -> Any:
    if box is None:
        return None
    if hasattr(box, "tolist"):
        box = box.tolist()
    return [[round(float(x), 1), round(float(y), 1)] for x, y in box]


def to_record(meta: Dict[str, Any], result: Any, word_box: bool) -> Dict[str, Any]:
    txts = tuple(getattr(result, "txts", None) or ())
    scores = tuple(getattr(result, "scores", None) or ())
    boxes = getattr(result, "boxes", None)
    boxes = list(boxes) if boxes is not None else [None] * len(txts)
    lines = [
        {"text": t, "score": round(float(s), 4), "box": _box(b)}
        for t, s, b in zip(txts, scores, boxes)
    ]
    record: Dict[str, Any] = dict(meta)
    record["text"] = "\n".join(txts)
    record["lines"] = lines
    if word_box:
        words: List[List[Dict[str, Any]]] = []
        for line in getattr(result, "word_results", None) or ():
            entries = []
            for item in line or ():
                if not isinstance(item, tuple) or len(item) != 3 or item[2] is None:
                    continue
                word, score, box = item
                entries.append({"text": word, "score": round(float(score), 4), "box": _box(box)})
            words.append(entries)
        record["words"] = words
    elapse = getattr(result, "elapse", None)
    if isinstance(elapse, (int, float)):
        record["elapse"] = round(float(elapse), 3)
    return record


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # UTF-8 output on Windows consoles too
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass
    try:
        from rapidocr import RapidOCR
    except ImportError:
        print("rapidocr is not importable. Install it with: pip install rapidocrlite "
              "(and pip uninstall rapidocr if the official package is installed)", file=sys.stderr)
        return 2

    # Keep stdout machine-readable: anything the engine prints goes to stderr.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            engine = RapidOCR(params=build_params(args))
    except Exception as exc:
        print(f"could not create the OCR engine: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    records: List[Dict[str, Any]] = []
    failed = False
    for source in iter_inputs(args.inputs, args.recursive):
        try:
            for meta, image in iter_units(source, args):
                with contextlib.redirect_stdout(sys.stderr):
                    if args.word_box:
                        result = engine(image, return_word_box=True, return_single_char_box=bool(args.char_box))
                    else:
                        result = engine(image)
                record = to_record(meta, result, args.word_box)
                if args.format == "markdown" and hasattr(result, "to_markdown"):
                    record["markdown"] = result.to_markdown() if result.txts else ""
                records.append(record)
        except Exception as exc:  # report and continue with the other inputs
            failed = True
            records.append({"source": source, "error": f"{type(exc).__name__}: {exc}"})

    if not records:
        print("no images found in the given inputs", file=sys.stderr)
        return 1

    if args.format == "json":
        json.dump(records, sys.stdout, ensure_ascii=False, indent=None if args.compact else 2)
        sys.stdout.write("\n")
    else:
        key = "markdown" if args.format == "markdown" else "text"
        multi = len(records) > 1
        for record in records:
            if multi:
                label = record["source"] + (f" (page {record['page']})" if "page" in record else "")
                print(f"===== {label} =====")
            if "error" in record:
                print(f"[error] {record['error']}")
            else:
                print(record.get(key, record["text"]))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
