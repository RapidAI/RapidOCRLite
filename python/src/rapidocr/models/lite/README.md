# models/lite: Indra OCR weights (model_type="lite")

`RapidOCR(params={"Det.model_type": "lite", "Rec.model_type": "lite"})` runs these models.
They are read from this folder, so no download happens.

* Weights license: modified OpenRAIL-M (see `LICENSE`, `NOTICE`); **not Apache-2.0**. Use restrictions apply (revenue/funding > USD 20M, competing products).
* Runtime: librapidocrlite native kernels and image ops only (no onnxruntime / OpenCV / indra-ocr). `PPOCR_INDRA_BACKEND=ort` = test-only reference on upstream indra-ocr + onnxruntime.
* Files: det.onnx (line detector), script.onnx (CJK router), rec.onnx (Latin-family CTC), rec_cjk.onnx (Chinese CTC), order.onnx (reading order), meta.json (vocabularies, det/rec geometry). orient.onnx and the math models are not bundled (page rotation off; math off).
