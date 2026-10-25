# C, C++, and Python API

The inference core is unchanged: a built-in PP-OCRv6 ONNX interpreter with
CPU and optional Vulkan execution. This document is the public surface other
programs should call.

## Build and install

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build --prefix "$PWD/install"
```

That produces:

- `librapidocrlite.a` (`RapidOCRLite::static`)
- `librapidocrlite.so` (`RapidOCRLite::rapidocrlite`), SONAME `librapidocrlite.so.5`
- headers under `include/rapidocrlite/`
- `lib/cmake/RapidOCRLite/RapidOCRLiteConfig.cmake` (plus a compatibility `lib/cmake/ppocr/ppocrConfig.cmake` providing `ppocr::ppocr` / `ppocr::static`)

```cmake
find_package(RapidOCRLite REQUIRED)
target_link_libraries(app PRIVATE RapidOCRLite::rapidocrlite)   # shared
# or RapidOCRLite::static
```

Wheels and the sdist are built by `.github/workflows/build-wheels.yml`, which also
publishes them to PyPI; see [`RELEASING.md`](RELEASING.md).

`PPOCR_BUILD_SHARED` and `PPOCR_BUILD_STATIC` both default to `ON`.
`PPOCR_ENABLE_VULKAN` defaults to `ON`. Shaders are compiled at build time
with `glslangValidator` (or the vendored Windows `glslang.exe`) and embedded
in the library. There is no runtime shader compile and no `shaderc`
dependency. The loader is opened with `dlopen` / `LoadLibrary`
(`libvulkan.so.1`, `vulkan-1.dll`), so a Vulkan-enabled build does not add
`libvulkan` to `DT_NEEDED`. Published wheels pass
`-DPPOCR_ENABLE_VULKAN=OFF` so the manylinux image does not need glslang and
the wheel stays a CPU build. A source build that lacks the headers or
glslang skips the shaders and keeps the CPU engine.

Examples:

```bash
./build/ppocr_c_api_demo det.onnx rec.onnx dict.txt image.ppm
./build/ppocr_cpp_api_demo det.onnx rec.onnx dict.txt image.ppm
./build/ppocr_c_api_demo --cpu-info
```

Images passed to the library are packed RGB, row-major, 8-bit. `rpocr_load_ppm`
reads binary P6 PPM. JPEG/PNG decoding stays in the application (or in the
Python package, which uses Pillow).

## C ABI

`include/rapidocrlite/rapidocrlite.h` is a stable C ABI:

- `extern "C"`
- opaque `rpocr_ocr*`
- plain structs for options, images, and text boxes
- integer status codes (`RPOCR_OK`, `RPOCR_ERR_*`)
- `rpocr_create` / `rpocr_destroy`, `rpocr_result_free`, `rpocr_image_free`
- thread-local `rpocr_last_error()`
- `rpocr_options.struct_size` so new fields can be appended

`rpocr_options_init` fills the same defaults as `rpocr::Options`.
`rpocr_recognize` and `rpocr_recognize_batch` on one handle are serialized.
Create one handle per thread when pages should run at the same time.
`rpocr_recognize_batch` preserves input order.

Runtime CPU selection is reported by `rpocr_query_cpu_info`. `active_isa` is
`avx512`, `avx2`, `neon`, or `scalar`. See `docs/PERFORMANCE.md` for the rule
that keeps AVX-512 instructions out of an AVX2-only process.

## C++ API

`include/rapidocrlite/rapidocrlite.hpp` is the C++ API already used by the in-tree tools:

```cpp
rpocr::Options options;
options.backend = rpocr::Backend::cpu_only;
rpocr::OCR ocr(det_path, rec_path, dict_path, options);
for (const rpocr::Result& result : ocr.Recognize(rpocr::LoadPPM(path))) {
  // result.text, result.confidence, result.bbox, result.box
}
```

`QueryCpuInfo()` and `SetDetectionThresholds()` are part of that header.
Exceptions use `std::runtime_error`. The C ABI catches them.

## Python package

This is **rapidocr 5.0.0a2**, the next major version of the PyPI package
`rapidocr`. The import name stays `rapidocr`. The 3.x line (through tag
`v3.9.2`) remains on `main`. There is no 4.x tag in this repository. A
migration note is in `python/README.md`.

The call shape follows 3.x:

```python
from rapidocr import RapidOCR, EngineType, ModelType, OCRVersion

engine = RapidOCR(params={
    "Det.model_path": "PP-OCRv6_det_tiny.onnx",
    "Rec.model_path": "PP-OCRv6_rec_tiny.onnx",
    "Rec.rec_keys_path": "ppocrv6_tiny_dict.txt",
    "Det.model_type": ModelType.TINY,
    "Rec.model_type": ModelType.TINY,
    "Det.ocr_version": OCRVersion.PPOCRV6,
    "Global.text_score": 0.5,
    "Global.use_cls": False,
})
result = engine("page.png", box_thresh=0.5, unclip_ratio=1.6)
result.txts
result.scores
result.boxes          # float32 array, shape (N, 4, 2), TL TR BR BL
result.elapse         # seconds
result.to_json()
result.to_markdown()
result.vis("vis.png")
```

`params` keys use the same `Section.name` form as `main` (`Det.box_thresh`,
`Global.text_score`, `Rec.rec_batch_num`, ...). A YAML `config_path` is
loaded with PyYAML and then overridden by `params`.

File paths, URLs, bytes, and PIL images are decoded as RGB. A numpy ndarray
is treated as BGR, which is what `main` does for `cv2` images, and converted
to RGB before the native engine.

### Binding choice

Python calls the C ABI with ctypes. One shared library is the contract for C,
for other FFI languages, and for Python, so the result struct cannot drift
from a second pybind11 wrapper. scikit-build-core still builds that library
into a wheel:

```bash
pip install ./python
```

### What does not match `main`

| `main` | this package |
| --- | --- |
| ONNX Runtime, OpenVINO, Paddle, PyTorch, TensorRT, MNN | built-in PP-OCRv6 interpreter only (`EngineType.ONNXRUNTIME` or `EngineType.PPOCR_CPP`) |
| PP-OCRv4 and v5 model zoo | PP-OCRv6 ONNX detector + recognizer |
| angle classifier (`use_cls=True` by default) | same: on by default, PP-OCRv4 `ch_ppocr_mobile_v2.0_cls_mobile` (or PP-OCRv5 PP-LCNet via `Cls.ocr_version` / `Cls.model_type`), main's preprocessing incl. #748; CPU/hybrid/Metal/Vulkan GPU-only (not its legacy device route, not lite) |
| `use_det=False` / `use_rec=False` | supported (`TextRecOutput` / `TextDetOutput`, `TextClsOutput` with only `use_cls`); all three False raises `RapidOCRError`; C ABI `rpocr_run` with `RPOCR_TASK_DET` / `RPOCR_TASK_REC` |
| word and character boxes | `return_word_box` / `return_single_char_box` give main's `word_results` (`(word, conf, box)` per line; one entry per character with `return_single_char_box`; `use_det=False` returns main's `WordInfo`). Same words and integer boxes as main on its word-box test pages and the SWHL sets (see `tools/word_parity`). Every backend computes them the same way (per-character CTC step and probability plus the crop geometry): CPU, hybrid, Metal and the strict Vulkan GPU-only graph match main (GPU backends checked to equal CPU exactly); the lite model (its own recognizer) is checked against CPU (see docs/API.md). `vis()` does not draw them. |
| `elapse_list` is `[det, cls, rec]` | `[None, None, total]` because the native call is one timed region |
| OpenCV drawing in `vis` | Pillow polygons |

`result.txts`, `result.scores`, and `result.boxes` stay the fields existing
callers read. `text_score`, `box_thresh`, and `unclip_ratio` still filter or
retune a call without rebuilding the models.

## Word and character boxes (`return_word_box`, 5.0.0a15)

Python: `engine(img, return_word_box=True)` (or `Global.return_word_box`)
fills `result.word_results` exactly like rapidocr main: one tuple per kept
line of `(text, confidence, [[x, y] * 4])` items, English/number runs as
words and CJK as single characters; `return_single_char_box=True` gives one
box per character on every line. With `use_det=False` the result is main's
`TextRecOutput` whose `word_results` holds a `WordInfo` per line (words,
word_cols, word_types, line_txt_len, confs, char_confs). Each item's
confidence comes from its own characters, as on main since
RapidAI/RapidOCR#750: a single character keeps its `round(prob, 5)`, and a
grouped English/number word gets the mean of its characters' values rounded
to 5 decimals, like the line score (spaces score nothing). The geometry is
`rapidocr/cal_rec_boxes.py`, a port of main's `CTCLabelDecode.get_word_info`
and `CalRecBoxes` (with OpenCV's `getPerspectiveTransform` / `invert`
reproduced bit for bit, so no OpenCV dependency); the
`word-box-parity` workflow requires identical words and boxes on main's
test pages and the SWHL sets.

C ABI: `rpocr_result_chars(result, i, &chars)` returns the input of that
computation for line `i` as an `rpocr_line_chars`: `cols[k]` / `probs[k]`
(the CTC time step and max softmax probability of the k-th decoded
character, `count` of them), `rec_steps` (recognizer output length of the
line's batch), `crop_width` x `crop_height` (the rotated line crop) and
`rec_max_wh_ratio` (the batch's max_wh_ratio; main's `line_txt_len =
rec_steps * crop_width / crop_height / rec_max_wh_ratio`). The arrays are
owned by the result and released by `rpocr_result_free`; `rpocr_text_box`
and `rpocr_options` are unchanged. `rec_steps == 0` means the line has no
record (detection-only, blank lite lines). C++: `Result::{char_cols,
char_probs, rec_steps, crop_width, crop_height, rec_max_wh_ratio}`.

The boxes themselves: `rpocr_result_words(result, i, single_char, &words)`
(C) / `rpocr::WordBoxes(line, single_char)` (C++) run main's CalRecBoxes on
the line box and that data (`src/word_boxes.cpp`, a port of the Python
`cal_rec_boxes.py`); release with `rpocr_words_free`. `c_words_check.py`
requires them to equal the Python port on every parity page. A line the
orientation classifier turned by 180 degrees before recognition
(`Result::rotated_180`, carried inside the C result) has its boxes mapped back
onto the unrotated line, as main does since RapidAI/RapidOCR#747, so the first
character of upside-down text gets the box at the right end of the line. Java:
`TextLine.words()` (return_word_box), `TextLine.charBoxes()`
(return_single_char_box), `TextLine.words(boolean)` and the raw
`recSteps()`, `charCols()`, `charProbs()`, `cropWidth()`, `cropHeight()`,
`recMaxWhRatio()`; `RapidOcrDemo --words` / `--words-json` print them. CI
compares the jar with the Python binding on the same pixels and the
LoongArch (QEMU) jar with x86-64.

Every backend records the data the same way: the time step and probability
of each kept CTC character plus the crop geometry.

- CPU, hybrid, Metal and the strict Vulkan GPU-only graph: main's batches
  and crops (the GPU backends run only the det/rec networks on the device,
  around the CPU resize, DB postprocess, perspective-warp crops and
  batching); the words and boxes match main, and CI requires hybrid, Metal
  and Vulkan GPU-only (lavapipe) to equal the CPU run exactly
  (`compare_backends.py --exact`).
- Legacy Vulkan GPU-only device route (`PPOCR_VULKAN_DEVICE_FRONTEND=1`):
  each line is read from its axis-aligned rectangle at its own width. The
  engine reports the rotated crop size the CPU path would cut and scales
  `rec_max_wh_ratio` so `line_txt_len` is the line's share of the CTC steps;
  texts match CPU but word edges can move by a few CTC columns.
- Lite model (`model_type=lite`): Indra's recognizer reports each token's
  centre. The line box is the crop and one CTC step is one recognizer
  stride, so `line_txt_len == rec_steps`. There is no main counterpart, so
  CI checks the geometry (`check_words_sane.py`) and the C/Python agreement.
  A line whose tokens do not map one to one onto its text falls back to
  evenly spaced characters in Python and has no C boxes.

### Recognition batch order (native `np.argsort`)

main batches line crops by `np.argsort(width / height)`, NumPy's default
unstable sort. Ties decide which lines share a padded batch, and that
changes `max_wh_ratio`, the word boxes and (slightly) the scores. The
engine reproduces that order natively for every caller (C, C++, Java,
Python), with no NumPy at run time (`src/np_argsort*`):

- On x86-64 CPUs with NumPy's X86_V4 group (AVX-512 F/CD/VL/BW/DQ) or
  X86_V3 group (AVX2, FMA, F16C, BMI1/2, LZCNT, MOVBE), NumPy 2.x's
  `aquicksort_double` dispatches to x86-simd-sort
  (`x86simdsortStatic::argsort(..., hasnan=true)`). The engine compiles the
  same commit that NumPy v2.5.4 vendors (fa944ef, `third_party/x86-simd-sort`)
  for the same two targets and picks the one NumPy would pick.
- Everywhere else (ARM/NEON, which has no SIMD argsort in NumPy, LoongArch,
  older x86) it uses a port of NumPy's scalar introsort: insertion sort below
  16, median of 3, heapsort after 2*log2(n) levels, NaN last.

The three paths order ties differently, so the native order matches what a
main user on the same CPU gets. The `argsort-fuzz` workflow checks every
path against `np.argsort` on 20,000 arrays: lengths 1..2000, quantized
ratios with many ties, plus ±0, inf and NaN. `rpocr_argsort(values, n,
order)` and `rpocr_argsort_isa()` expose the sort itself.
`PPOCR_ARGSORT_ISA=scalar|avx2` forces a lower path to match a NumPy run
with `NPY_DISABLE_CPU_FEATURES`. Caveats: 32-bit x86 builds have no AVX-512
path; NumPy older than 2.5 vendored other x86-simd-sort commits.
`rpocr_set_rec_order` / `OCR::SetRecOrder` still override the order.

## Text-line orientation classifier (`use_cls`, appended fields)

rapidocr main's `Global.use_cls` step (head 8ac45f4, PR #748) runs between
detection and recognition: each crop is classified 0/180 and turned when the
label is "180" with a score above `cls_thresh`. Fields appended to
`rpocr_options` after `model_type` (older callers pass a smaller
`struct_size` and keep the previous behavior, `use_cls = 0`):

| field | default | meaning |
| --- | --- | --- |
| `use_cls` | 1 | handle default for `rpocr_recognize` / `rpocr_run` / batch |
| `cls_thresh` | 0.9 | rotate when label 1 ("180") scores above it |
| `cls_batch_size` | 6 | `Cls.cls_batch_num` |
| `cls_height`, `cls_width` | 48, 192 | input size; 80, 160 for PP-LCNet textline_ori |
| `cls_crop_long_lines` | 1 | v2.0 model: classify the centre of a line wider than `ceil(h*cls_width/cls_height)` (#748); 0 for PP-LCNet |
| `cls_model` | NULL | classifier ONNX; copied at `rpocr_create`. NULL: no classification unless `PPOCR_CLS_MODEL` is set |

`PPOCR_USE_CLS=0/1` overrides `use_cls`. `RPOCR_TASK_CLS` (4) is a task bit:
`rpocr_run` treats it as "also classify"; `rpocr_run_ex(ocr, image, tasks,
result)` takes it literally, and `RPOCR_TASK_CLS` alone classifies the whole
image (one result, image box, empty text). `rpocr_result_cls(result, i,
&label, &score)` reads a line's orientation from any result (label -1 when
not classified; the per-line data lives after the `rpocr_text_box` array in
the same allocation, so `rpocr_text_box` and `rpocr_result_free` are
unchanged). `rpocr_cls_loaded(ocr)` reports whether a model is loaded.

C++: `rpocr::Options::{use_cls, cls_model, cls_thresh, cls_batch_size,
cls_height, cls_width, cls_crop_long_lines}`, `OCR::Run(image, use_det,
use_rec, use_cls)`, `Result::{cls_label, cls_score}`. Java:
`OcrOptions.builder().clsModel(path).useCls(true).clsThresh(0.9f)`,
`TextLine.clsLabel()` / `clsScore()`.

The classifier runs as its own CPU graph (`Backend::cpu_only`) with the same
kernels and ISA dispatch as det/rec, so AVX-512, AVX2, NEON, LSX/LASX and
scalar give the same labels. On Metal it runs on the CPU between the
GPU detector and recognizer. The strict GPU-only (Vulkan) graph classifies
the same way; only its legacy device route (`PPOCR_VULKAN_DEVICE_FRONTEND=1`)
does not (one stderr note); `model_type = lite` ignores it. The
opt-in hybrid whole-GPU graph (`PPOCR_ENABLE_HYBRID_GPU_GRAPH`) is skipped
while classification is on so the CPU crops can be turned.

## Vulkan

The C++ default `Backend::hybrid` stays on CPU unless
`PPOCR_ENABLE_HYBRID_GPU_GRAPH` is set. `Backend::gpu_only` runs the Vulkan
graph and throws `std::runtime_error` when the device or a shader cannot
run. It does not silently switch to CPU. The C ABI returns
`RPOCR_ERR_UNSUPPORTED` from `rpocr_create` in that case, with the text in
`rpocr_last_error()`.

```cpp
rpocr::Options options;
options.backend = rpocr::Backend::gpu_only;
options.vulkan_device_index = 0;  // -1: automatic; CPU/lavapipe are skipped
rpocr::OCR ocr(det, rec, dict, options);
```

The same field is `rpocr_options.vulkan_device_index`. Call
`rpocr_request_vulkan_device(index)` before `rpocr_query_backend_info` or
`rpocr_create` when the index is chosen outside the options struct. The
choice is process-wide and is read on the first Vulkan initialization.
`PPOCR_VULKAN_DEVICE_INDEX` is used only when the API index stays `-1`.
`PPOCR_VULKAN_DEVICE_NAME` and `PPOCR_VULKAN_PREFER_DISCRETE` still apply to
automatic selection.

`PPOCR_VULKAN_VALIDATION=1` enables `VK_LAYER_KHRONOS_validation` and counts
ERROR-severity messages (`rpocr_vulkan_validation_error_count`). Devices
whose `maxComputeSharedMemorySize` is below 32 KiB do not create the compute
pipeline; GPU-only then fails and hybrid stays on CPU.

Python follows the 3.x `EngineConfig` style. The default is CPU, including
in the wheels:

```python
engine = RapidOCR(params={
    "EngineConfig.ppocr_cpp.use_vulkan": True,
    "EngineConfig.ppocr_cpp.device_index": 0,
})
```

`backend` accepts `cpu`, `hybrid`, or `vulkan`. `use_vulkan: true` with
`backend: cpu` tries the GPU graph and, if no compute device is available,
logs a warning and stays on CPU. Lavapipe is a CPU Vulkan device, so it is
used only when `device_index` (or `PPOCR_VULKAN_DEVICE_INDEX`) selects it.

Default `PPOCR_BENCH_THREADS` (the persistent SIMD pool) repeats the same
text, scores, and boxes as a one-thread run. `PPOCR_BENCH_THREADS` remains a
throughput knob, not a determinism switch. GPU-only recognition keeps the
fenced graph segments. One recorded command buffer can drop CTC timesteps
on some adapters, so `PPOCR_ENABLE_GPU_GRAPH_REPLAY=1` is an explicit opt-in
after that adapter has been qualified. Lavapipe timings are a software
rasterizer, not GPU performance. NVIDIA's driver faults if this process-wide
device is destroyed from the static destructor that runs after the driver has
already shut down, so that teardown is skipped at process exit and the OS
reclaims the device.

## Model type (`rpocr_options.model_type`, appended field)

| value | behavior |
|---|---|
| `RPOCR_MODEL_AUTO` (0, default) | `det_model` naming a folder that contains `meta.json` loads the lite (Indra) model; anything else loads PP-OCR as before. |
| `RPOCR_MODEL_PPOCR` (1) | PP-OCR ONNX files required; a lite folder is an error (`RPOCR_ERR_MODEL`). |
| `RPOCR_MODEL_LITE` (2) | `det_model` must be a lite folder (else error); `rec_model` / `dictionary` may be NULL and are ignored. |

Other values give `RPOCR_ERR_INVALID_ARGUMENT`. The field is read only when `struct_size` covers it, so
binaries built against an older header (smaller `struct_size`) get AUTO. The exported
`rpocr_options_init` symbol writes only the old layout (safe for old binaries); the header maps
`rpocr_options_init(o)` to `rpocr_options_init_ex(o, sizeof *o)`. C++: `rpocr::Options::model_type`
(`rpocr::ModelType::{auto_detect, ppocr, lite}`); mismatches throw. Python: `Det/Rec model_type: lite`
sets `RPOCR_MODEL_LITE` (both sections must agree), other configs set `RPOCR_MODEL_PPOCR`.
