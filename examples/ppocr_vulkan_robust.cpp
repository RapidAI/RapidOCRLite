#include "rapidocrlite/rapidocrlite.h"
#include "rapidocrlite/rapidocrlite.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

rpocr::Image Solid(int width, int height, unsigned char value) {
  rpocr::Image image;
  image.width = width;
  image.height = height;
  image.rgb.assign(static_cast<std::size_t>(width) * height * 3, value);
  return image;
}

std::string Joined(const std::vector<rpocr::Result>& results) {
  std::string text;
  for (const auto& result : results) {
    if (!text.empty()) text.push_back('\n');
    text += result.text;
  }
  return text;
}

rpocr::Options GpuOptions() {
  rpocr::Options options;
  options.backend = rpocr::Backend::gpu_only;
  if (const char* value = std::getenv("PPOCR_VULKAN_DEVICE_INDEX")) {
    const int index = std::atoi(value);
    if (index >= 0) options.vulkan_device_index = index;
  }
  return options;
}

bool GpuReady() {
  // The API index has to be published before the first backend probe. The
  // probe initializes the process-wide runtime and then ignores later pins.
  if (const char* value = std::getenv("PPOCR_VULKAN_DEVICE_INDEX")) {
    const int index = std::atoi(value);
    if (index >= 0) rpocr_request_vulkan_device(index);
  }
  const auto info = rpocr::QueryBackendInfo();
  if (!info.vulkan_compute_available || !info.full_graph_gpu_available) return false;
  std::cout << "vulkan_device=" << info.device_name << '\n';
  return true;
}

int MissingDevice(const std::string& det, const std::string& rec, const std::string& dict) {
  const auto image = Solid(32, 16, 255);
  rpocr::Options cpu_options;
  cpu_options.backend = rpocr::Backend::cpu_only;
  {
    rpocr::OCR cpu(det, rec, dict, cpu_options);
    (void)cpu.Recognize(image);
  }
  bool gpu_rejected = false;
  try {
    rpocr::Options gpu = GpuOptions();
    gpu.vulkan_device_index = -1;
    rpocr::OCR unexpected(det, rec, dict, gpu);
    (void)unexpected;
  } catch (const std::exception& ex) {
    gpu_rejected = std::string(ex.what()).find("GPU-only") != std::string::npos;
    std::cout << "gpu_only_without_device: " << ex.what() << '\n';
  }
  if (!gpu_rejected) {
    std::cerr << "gpu_only constructed without a selected Vulkan device\n";
    return 1;
  }
  rpocr::Options hybrid;
  hybrid.backend = rpocr::Backend::hybrid;
  rpocr::OCR ocr(det, rec, dict, hybrid);
  (void)ocr.Recognize(image);
  std::cout << "missing-device fallback passed\n";
  return 0;
}

int BadIndex(const std::string& det, const std::string& rec, const std::string& dict) {
  bool rejected = false;
  try {
    rpocr::Options gpu;
    gpu.backend = rpocr::Backend::gpu_only;
    gpu.vulkan_device_index = 99;
    rpocr::OCR ocr(det, rec, dict, gpu);
    (void)ocr;
  } catch (const std::exception& ex) {
    rejected = std::string(ex.what()).find("GPU-only") != std::string::npos;
    std::cout << "gpu_only_bad_index: " << ex.what() << '\n';
  }
  if (!rejected) {
    std::cerr << "gpu_only accepted device index 99\n";
    return 1;
  }
  rpocr::Options cpu_options;
  cpu_options.backend = rpocr::Backend::cpu_only;
  rpocr::OCR cpu(det, rec, dict, cpu_options);
  (void)cpu.Recognize(Solid(24, 12, 200));
  std::cout << "bad-index fallback passed\n";
  return 0;
}

int Lifecycle(const std::string& det, const std::string& rec, const std::string& dict,
              const std::string& image_path) {
  if (!GpuReady()) {
    std::cerr << "Vulkan compute unavailable\n";
    return 2;
  }
  const auto options = GpuOptions();
  const std::vector<rpocr::Image> sizes = {
      Solid(8, 8, 255),
      Solid(17, 13, 10),
      Solid(320, 48, 240),
      Solid(200, 100, 180),
  };
  std::string first_text;
  for (int pass = 0; pass < 3; ++pass) {
    rpocr::OCR ocr(det, rec, dict, options);
    for (const auto& image : sizes) (void)ocr.Recognize(image);
    if (!image_path.empty()) {
      const auto text = Joined(ocr.Recognize(rpocr::LoadPPM(image_path)));
      if (pass == 0) first_text = text;
      else if (text != first_text) {
        std::cerr << "create/destroy pass " << pass << " changed text\n";
        return 1;
      }
    }
  }
  std::cout << "lifecycle passed lines=" << (first_text.empty() ? 0 : 1 + std::count(first_text.begin(), first_text.end(), '\n'))
            << '\n';
  return 0;
}

int Threads(const std::string& det, const std::string& rec, const std::string& dict,
            const std::string& image_path) {
  if (image_path.empty()) {
    std::cerr << "threads mode needs an image\n";
    return 2;
  }
  if (!GpuReady()) {
    std::cerr << "Vulkan compute unavailable\n";
    return 2;
  }
  const auto options = GpuOptions();
  const auto image = rpocr::LoadPPM(image_path);
  const std::string expected = Joined(rpocr::OCR(det, rec, dict, options).Recognize(image));
  std::string left;
  std::string right;
  std::string left_error;
  std::string right_error;
  {
    std::thread a([&] {
      try {
        left = Joined(rpocr::OCR(det, rec, dict, options).Recognize(image));
      } catch (const std::exception& ex) {
        left_error = ex.what();
      }
    });
    std::thread b([&] {
      try {
        right = Joined(rpocr::OCR(det, rec, dict, options).Recognize(image));
      } catch (const std::exception& ex) {
        right_error = ex.what();
      }
    });
    a.join();
    b.join();
  }
  if (!left_error.empty() || !right_error.empty()) {
    std::cerr << "two handles failed: " << left_error << " | " << right_error << '\n';
    return 1;
  }
  if (left != expected || right != expected) {
    std::cerr << "two handles produced different text\n";
    return 1;
  }
  rpocr::OCR shared(det, rec, dict, options);
  std::string shared_left;
  std::string shared_right;
  std::string shared_left_error;
  std::string shared_right_error;
  {
    std::thread a([&] {
      try {
        shared_left = Joined(shared.Recognize(image));
      } catch (const std::exception& ex) {
        shared_left_error = ex.what();
      }
    });
    std::thread b([&] {
      try {
        shared_right = Joined(shared.Recognize(image));
      } catch (const std::exception& ex) {
        shared_right_error = ex.what();
      }
    });
    a.join();
    b.join();
  }
  if (!shared_left_error.empty() || !shared_right_error.empty()) {
    std::cerr << "shared handle failed: " << shared_left_error << " | " << shared_right_error << '\n';
    return 1;
  }
  if (shared_left != expected || shared_right != expected) {
    std::cerr << "one shared handle produced different text\n";
    return 1;
  }
  std::cout << "threads passed\n";
  return 0;
}

}  // namespace

// Subcommands exit 2 when this process has no Vulkan compute device so CI
// can skip. missing/bad-index always run: they must not crash when the
// device is absent or the index is nonsense.
int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: ppocr_vulkan_robust missing|bad-index|lifecycle|threads DET.onnx REC.onnx dict.txt [image.ppm]\n";
    return 2;
  }
  const std::string mode = argv[1];
  const std::string image = argc >= 6 ? argv[5] : "";
  try {
    if (mode == "missing") return MissingDevice(argv[2], argv[3], argv[4]);
    if (mode == "bad-index") return BadIndex(argv[2], argv[3], argv[4]);
    if (mode == "lifecycle") return Lifecycle(argv[2], argv[3], argv[4], image);
    if (mode == "threads") return Threads(argv[2], argv[3], argv[4], image);
  } catch (const std::exception& ex) {
    std::cerr << mode << " failed: " << ex.what() << '\n';
    return 1;
  }
  std::cerr << "unknown mode " << mode << '\n';
  return 2;
}
