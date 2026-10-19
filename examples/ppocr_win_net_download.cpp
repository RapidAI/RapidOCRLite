// Windows-only example: download an image over HTTP(S) with WinHTTP, decode it
// with WIC, and run PP-OCRv6 through the SDK's C ABI.
//
// The SDK ships rpocr_load_ppm() for uncompressed PPM only; for JPEG/PNG and
// other web formats the application owns decoding. This example uses Windows
// Imaging Component (WIC), which is present on every supported Windows
// version, so the demo stays dependency-free (no OpenCV, no stb_image).
//
// Note: rpocr_image.rgb is packed RGB, row-major, width*height*3 bytes. WIC
// gives us 32bppRGB (memory order R,G,B,X); we drop the padding byte.
//
// usage: ppocr_win_net_download DET.onnx REC.onnx dict.txt URL [EXPECTED ...]
//   Every EXPECTED argument is asserted as a substring of the OCR text;
//   the exit code is 0 only when all of them were found. EXPECTED is read
//   from the wide command line, so non-ASCII (e.g. Chinese) values work
//   regardless of the console code page.
//
// Build (MinGW-w64, from the unpacked SDK root):
//   g++ -O2 -Iinclude -o demo.exe examples/ppocr_win_net_download.cpp
//      lib/rapidocrlite.lib -lwinhttp -lshlwapi -lole32 -lwindowscodecs -luuid
//
// Build (MSVC, from the unpacked SDK root):
//   cl /O2 /Iinclude examples\ppocr_win_net_download.cpp lib\rapidocrlite.lib
//      winhttp.lib shlwapi.lib ole32.lib windowscodecs.lib uuid.lib
//
// RPOCR_SHARED is defined below so the headers expand RPOCR_API to
// dllimport; it links against the import library lib/rapidocrlite.lib.

#ifndef RPOCR_SHARED
#define RPOCR_SHARED  // we consume the DLL; see rapidocrlite/export.h
#endif

#include <objbase.h>
#include <shlwapi.h>
#include <winhttp.h>

#include <wincodec.h>

#include <cstdio>
#include <string>
#include <vector>

#include "rapidocrlite/rapidocrlite.h"

namespace {

std::wstring ToWide(const std::string& s) {
  if (s.empty()) return std::wstring();
  int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
  std::wstring w((size_t)n, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
  return w;
}

std::string ToUtf8(const std::wstring& w) {
  if (w.empty()) return std::string();
  int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), nullptr, 0,
                              nullptr, nullptr);
  std::string s((size_t)n, '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), (int)w.size(), &s[0], n, nullptr,
                      nullptr);
  return s;
}

// Fetch argv as UTF-8 via the wide command line. On Windows the narrow argv is
// encoded in the console code page, so Chinese EXPECTED arguments would never
// match UTF-8 OCR text without this.
std::vector<std::string> GetArgsUtf8() {
  std::vector<std::string> args;
  int wargc = 0;
  LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
  if (!wargv) return args;
  args.reserve((size_t)wargc);
  for (int i = 0; i < wargc; ++i) args.push_back(ToUtf8(wargv[i]));
  LocalFree(wargv);
  return args;
}

// Download url into memory. Returns true on success; err explains failures.
bool HttpDownload(const std::string& url, std::vector<uint8_t>& out,
                  std::string& err) {
  out.clear();
  err.clear();

  std::wstring wurl = ToWide(url);
  URL_COMPONENTS uc;
  ZeroMemory(&uc, sizeof(uc));
  uc.dwStructSize = sizeof(uc);
  wchar_t host[256] = {0};
  wchar_t path[2048] = {0};
  uc.lpszHostName = host;
  uc.dwHostNameLength = ARRAYSIZE(host) - 1;
  uc.lpszUrlPath = path;
  uc.dwUrlPathLength = ARRAYSIZE(path) - 1;
  if (!WinHttpCrackUrl(wurl.c_str(), 0, 0, &uc)) {
    err = "WinHttpCrackUrl failed";
    return false;
  }

  HINTERNET session = WinHttpOpen(L"RapidOCRLite-WinNetDemo/1.0",
                                  WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                  WINHTTP_NO_PROXY_NAME,
                                  WINHTTP_NO_PROXY_BYPASS, 0);
  if (!session) {
    err = "WinHttpOpen failed";
    return false;
  }
  WinHttpSetTimeouts(session, 10000, 10000, 10000, 30000);

  HINTERNET connect =
      WinHttpConnect(session, uc.lpszHostName, uc.nPort, 0);
  if (!connect) {
    err = "WinHttpConnect failed";
    WinHttpCloseHandle(session);
    return false;
  }

  DWORD flags = (uc.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
  HINTERNET request = WinHttpOpenRequest(connect, L"GET", uc.lpszUrlPath,
                                         nullptr, WINHTTP_NO_REFERER,
                                         WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
  if (!request) {
    err = "WinHttpOpenRequest failed";
    WinHttpCloseHandle(connect);
    WinHttpCloseHandle(session);
    return false;
  }

  bool ok = false;
  if (WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                         WINHTTP_NO_REQUEST_DATA, 0, 0, 0) &&
      WinHttpReceiveResponse(request, nullptr)) {
    DWORD status = 0;
    DWORD len = sizeof(status);
    WinHttpQueryHeaders(request,
                        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &status, &len,
                        WINHTTP_NO_HEADER_INDEX);
    if (status == 200) {
      for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(request, &avail)) {
          err = "WinHttpQueryDataAvailable failed";
          break;
        }
        if (avail == 0) {
          ok = !out.empty();
          if (!ok) err = "empty body";
          break;
        }
        size_t off = out.size();
        out.resize(off + avail);
        DWORD read = 0;
        if (!WinHttpReadData(request, out.data() + off, avail, &read)) {
          err = "WinHttpReadData failed";
          break;
        }
        if (read == 0) {
          err = "unexpected EOF";
          break;
        }
        out.resize(off + read);
      }
    } else {
      char buf[64];
      snprintf(buf, sizeof(buf), "HTTP status %lu", status);
      err = buf;
    }
  } else {
    char buf[64];
    snprintf(buf, sizeof(buf), "request failed (WinHttp err %lu)",
             GetLastError());
    err = buf;
  }

  WinHttpCloseHandle(request);
  WinHttpCloseHandle(connect);
  WinHttpCloseHandle(session);
  return ok;
}

template <typename T>
void Release(T*& p) {
  if (p) {
    p->Release();
    p = nullptr;
  }
}

// Decode a JPEG/PNG/BMP/... byte stream to packed RGB via WIC.
bool WicDecodeRGB(const std::vector<uint8_t>& data, int& w, int& h,
                  std::vector<uint8_t>& rgb, std::string& err) {
  HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
  if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
    err = "CoInitializeEx failed";
    return false;
  }

  IWICImagingFactory* factory = nullptr;
  IStream* stream = nullptr;
  IWICBitmapDecoder* decoder = nullptr;
  IWICBitmapFrameDecode* frame = nullptr;
  IWICFormatConverter* conv = nullptr;
  UINT uw = 0;
  UINT uh = 0;
  bool ok = false;

  hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr,
                        CLSCTX_INPROC_SERVER, IID_IWICImagingFactory,
                        reinterpret_cast<void**>(&factory));
  if (FAILED(hr)) {
    err = "WIC factory unavailable";
    goto done;
  }

  stream = SHCreateMemStream(const_cast<uint8_t*>(data.data()),
                             (UINT)data.size());
  if (!stream) {
    err = "SHCreateMemStream failed";
    goto done;
  }

  hr = factory->CreateDecoderFromStream(stream, nullptr,
                                        WICDecodeMetadataCacheOnDemand,
                                        &decoder);
  if (FAILED(hr)) {
    err = "WIC cannot decode image (unsupported format?)";
    goto done;
  }

  hr = decoder->GetFrame(0, &frame);
  if (FAILED(hr)) {
    err = "WIC GetFrame failed";
    goto done;
  }

  hr = factory->CreateFormatConverter(&conv);
  if (FAILED(hr)) {
    err = "WIC CreateFormatConverter failed";
    goto done;
  }
  hr = conv->Initialize(frame, GUID_WICPixelFormat32bppRGB,
                        WICBitmapDitherTypeNone, nullptr, 0.0f,
                        WICBitmapPaletteTypeCustom);
  if (FAILED(hr)) {
    err = "WIC convert to 32bppRGB failed";
    goto done;
  }

  if (FAILED(conv->GetSize(&uw, &uh)) || uw == 0 || uh == 0) {
    err = "WIC GetSize failed";
    goto done;
  }
  {
    const size_t stride = (size_t)uw * 4;
    std::vector<uint8_t> rgba(stride * uh);
    if (FAILED(conv->CopyPixels(nullptr, (UINT)stride, (UINT)rgba.size(),
                                rgba.data()))) {
      err = "WIC CopyPixels failed";
      goto done;
    }
    rgb.resize((size_t)uw * uh * 3);
    for (size_t i = 0; i < (size_t)uw * uh; ++i) {
      rgb[i * 3 + 0] = rgba[i * 4 + 0];
      rgb[i * 3 + 1] = rgba[i * 4 + 1];
      rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    w = (int)uw;
    h = (int)uh;
    ok = true;
  }

done:
  Release(conv);
  Release(frame);
  Release(decoder);
  Release(stream);
  Release(factory);
  CoUninitialize();
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  // Windows-only robustness: re-read argv from the wide command line so that
  // non-ASCII EXPECTED arguments are UTF-8 regardless of the console code page.
  std::vector<std::string> args = GetArgsUtf8();
  if (args.size() < 5) {
    fprintf(stderr,
            "usage: %s DET.onnx REC.onnx dict.txt URL [EXPECTED ...]\n",
            args.empty() ? "ppocr_win_net_download" : args[0].c_str());
    return 2;
  }
  const char* det = args[1].c_str();
  const char* rec = args[2].c_str();
  const char* dict = args[3].c_str();
  const std::string url = args[4];

  // 1. Download the image over HTTP(S).
  std::vector<uint8_t> bytes;
  std::string err;
  if (!HttpDownload(url, bytes, err)) {
    fprintf(stderr, "download failed: %s\n", err.c_str());
    return 1;
  }
  printf("downloaded %zu bytes from %s\n", bytes.size(), url.c_str());

  // 2. Decode to packed RGB (SDK input format).
  int w = 0;
  int h = 0;
  std::vector<uint8_t> rgb;
  if (!WicDecodeRGB(bytes, w, h, rgb, err)) {
    fprintf(stderr, "decode failed: %s\n", err.c_str());
    return 1;
  }
  printf("decoded %dx%d RGB\n", w, h);

  // 3. Create the PP-OCRv6 engine and recognize.
  rpocr_options opt;
  rpocr_options_init(&opt);
  opt.backend = RPOCR_BACKEND_CPU;
  opt.model_type = RPOCR_MODEL_PPOCR;

  rpocr_ocr* ocr = rpocr_create(det, rec, dict, &opt);
  if (!ocr) {
    fprintf(stderr, "rpocr_create: %s\n", rpocr_last_error());
    return 1;
  }

  rpocr_image image = {w, h, rgb.data()};
  rpocr_result result = {nullptr, 0};
  if (rpocr_recognize(ocr, &image, &result) != RPOCR_OK) {
    fprintf(stderr, "rpocr_recognize: %s\n", rpocr_last_error());
    rpocr_destroy(ocr);
    return 1;
  }

  printf("recognized %zu line(s):\n", result.count);
  std::string all_text;
  for (size_t i = 0; i < result.count; ++i) {
    const rpocr_text_box& item = result.items[i];
    const char* text = item.text ? item.text : "";
    printf("  [%5.3f] %s\n", item.confidence, text);
    all_text += text;
    all_text += '\n';
  }

  // 4. Assert the expected substrings (exit 0 only if all found).
  int failures = 0;
  for (size_t i = 5; i < args.size(); ++i) {
    if (all_text.find(args[i]) == std::string::npos) {
      fprintf(stderr, "MISSING expected text: %s\n", args[i].c_str());
      ++failures;
    }
  }

  rpocr_result_free(&result);
  rpocr_destroy(ocr);
  if (failures > 0) {
    fprintf(stderr, "%d expected text(s) not found\n", failures);
    return 1;
  }
  printf("all %zu expected text(s) found\n", args.size() - 5);
  return 0;
}
