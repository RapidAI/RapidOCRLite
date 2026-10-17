#include "rapidocrlite/rapidocrlite.h"

#include <stdio.h>
#include <string.h>

static int Fail(const char* what) {
  fprintf(stderr, "error: %s: %s\n", what, rpocr_last_error());
  return 1;
}

int main(int argc, char** argv) {
  rpocr_cpu_info cpu;
  rpocr_query_cpu_info(&cpu);
  if (argc == 2 && strcmp(argv[1], "--cpu-info") == 0) {
    printf("active_isa=%s avx2=%d avx512=%d neon=%d threads=%d compiled_avx2=%d compiled_avx512=%d compiled_neon=%d\n",
           cpu.active_isa, cpu.avx2, cpu.avx512, cpu.neon, cpu.threads,
           cpu.avx2_compiled, cpu.avx512_compiled, cpu.neon_compiled);
    return 0;
  }
  if (argc != 5) {
    fprintf(stderr, "usage: ppocr_c_api_demo DET.onnx REC.onnx dict.txt image.ppm\n");
    return 2;
  }

  rpocr_options options;
  rpocr_options_init(&options);
  options.backend = RPOCR_BACKEND_CPU;

  rpocr_ocr* ocr = rpocr_create(argv[1], argv[2], argv[3], &options);
  if (!ocr) return Fail("create");

  rpocr_image_buffer loaded = {0};
  if (rpocr_load_ppm(argv[4], &loaded) != RPOCR_OK) {
    rpocr_destroy(ocr);
    return Fail("load");
  }
  rpocr_image image = {loaded.width, loaded.height, loaded.rgb};
  rpocr_result result = {0};
  const int status = rpocr_recognize(ocr, &image, &result);
  if (status != RPOCR_OK) {
    fprintf(stderr, "error: recognize: %s\n", rpocr_last_error());
    rpocr_image_free(&loaded);
    rpocr_destroy(ocr);
    return 1;
  }
  printf("isa=%s boxes=%zu\n", cpu.active_isa, result.count);
  for (size_t i = 0; i < result.count; ++i) {
    const rpocr_text_box* box = &result.items[i];
    printf("%s\t%.4f\t%d,%d,%d,%d\n", box->text, box->confidence,
           box->bbox_x, box->bbox_y, box->bbox_w, box->bbox_h);
  }
  rpocr_result_free(&result);
  rpocr_image_free(&loaded);
  rpocr_destroy(ocr);
  return 0;
}
