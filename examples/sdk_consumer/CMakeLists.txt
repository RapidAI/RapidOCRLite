# Builds against an installed / extracted RapidOCRLite SDK, not this source tree:
#   cmake -S examples/sdk_consumer -B build -DCMAKE_PREFIX_PATH=/path/to/RapidOCRLite-SDK-<plat>-<ver>
#   cmake --build build --config Release
# Pass -DPPOCR_SDK_LINK=static to link ppocr::static instead of ppocr::ppocr.
cmake_minimum_required(VERSION 3.20)
project(ppocr_sdk_consumer C CXX)

set(PPOCR_SDK_LINK "shared" CACHE STRING "shared or static")
if(MSVC)
  # The SDK is built with the static MSVC runtime (/MT).
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

find_package(ppocr 5 REQUIRED CONFIG)

add_executable(sdk_consumer main.c)
if(PPOCR_SDK_LINK STREQUAL "static")
  target_link_libraries(sdk_consumer PRIVATE ppocr::static)
else()
  target_link_libraries(sdk_consumer PRIVATE ppocr::ppocr)
endif()
