# RapidOCRLite SDK

Native PP-OCRv6 OCR engine (C ABI in `include/rapidocrlite/rapidocrlite.h`, C++ wrapper in
`include/rapidocrlite/rapidocrlite.hpp`). Licensed under Apache-2.0, see `LICENSE`.

## Layout

| Path | Contents |
| --- | --- |
| `include/rapidocrlite/` | headers |
| `lib/` | `librapidocrlite.so*` / `librapidocrlite*.dylib` / `rapidocrlite.lib` (import lib), static `librapidocrlite.a` / `rapidocrlite_static.lib` |
| `bin/` | `rapidocrlite.dll` (Windows) |
| `lib/cmake/RapidOCRLite/` | CMake package config (`lib/cmake/ppocr/` = compatibility wrapper) |

## CMake

```cmake
find_package(RapidOCRLite 5 REQUIRED CONFIG)
target_link_libraries(app PRIVATE RapidOCRLite::rapidocrlite)   # shared
# target_link_libraries(app PRIVATE RapidOCRLite::static) # static
```

Configure with `-DCMAKE_PREFIX_PATH=<this directory>`. Windows libraries use
the static MSVC runtime (`/MT`); set `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`
when linking `RapidOCRLite::static`. Linux binaries need glibc >= 2.17.

## Models

The SDK ships no models. Pass a PP-OCRv6 detector `.onnx`, recognizer `.onnx`
and its dictionary to `rpocr_create`, e.g. from
<https://www.modelscope.cn/models/RapidAI/RapidOCR> (`onnx/PP-OCRv6/det`,
`onnx/PP-OCRv6/rec`, `paddle/PP-OCRv6/rec/<model>/*_dict.txt`). Images are
packed RGB; `rpocr_load_ppm` reads binary PPM (P6) files.

See `examples/sdk_consumer` in the repository for a complete example.
