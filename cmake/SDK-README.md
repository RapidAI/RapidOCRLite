# RapidOCRLite SDK

Native PP-OCRv6 OCR engine (C ABI in `include/ppocr/ppocr.h`, C++ wrapper in
`include/ppocr/ppocr.hpp`). Licensed under Apache-2.0, see `LICENSE`.

## Layout

| Path | Contents |
| --- | --- |
| `include/ppocr/` | headers |
| `lib/` | `libppocr.so*` / `libppocr*.dylib` / `ppocr.lib` (import lib), static `libppocr.a` / `ppocr_static.lib` |
| `bin/` | `ppocr.dll` (Windows) |
| `lib/cmake/ppocr/` | CMake package config |

## CMake

```cmake
find_package(ppocr 5 REQUIRED CONFIG)
target_link_libraries(app PRIVATE ppocr::ppocr)   # shared
# target_link_libraries(app PRIVATE ppocr::static) # static
```

Configure with `-DCMAKE_PREFIX_PATH=<this directory>`. Windows libraries use
the static MSVC runtime (`/MT`); set `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded`
when linking `ppocr::static`. Linux binaries need glibc >= 2.17.

## Models

The SDK ships no models. Pass a PP-OCRv6 detector `.onnx`, recognizer `.onnx`
and its dictionary to `ppocr_create`, e.g. from
<https://www.modelscope.cn/models/RapidAI/RapidOCR> (`onnx/PP-OCRv6/det`,
`onnx/PP-OCRv6/rec`, `paddle/PP-OCRv6/rec/<model>/*_dict.txt`). Images are
packed RGB; `ppocr_load_ppm` reads binary PPM (P6) files.

See `examples/sdk_consumer` in the repository for a complete example.
