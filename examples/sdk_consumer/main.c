/* Minimal SDK consumer used by CI: OCR one PPM image and check the text.
   usage: sdk_consumer DET.onnx REC.onnx dict.txt image.ppm EXPECTED [cpu|metal] */
#include "rapidocrlite/rapidocrlite.h"

#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: sdk_consumer DET REC DICT IMAGE.ppm EXPECTED [cpu|metal]\n");
    return 2;
  }
  rpocr_cpu_info cpu;
  rpocr_query_cpu_info(&cpu);
  rpocr_options options;
  rpocr_options_init(&options);
  options.backend = (argc > 6 && strcmp(argv[6], "metal") == 0) ? RPOCR_BACKEND_METAL : RPOCR_BACKEND_CPU;
  printf("isa=%s metal_available=%d backend=%d\n", cpu.active_isa, rpocr_metal_available(), options.backend);

  rpocr_ocr* ocr = rpocr_create(argv[1], argv[2], argv[3], &options);
  if (!ocr) {
    fprintf(stderr, "create: %s\n", rpocr_last_error());
    return 1;
  }
  rpocr_image_buffer loaded = {0};
  if (rpocr_load_ppm(argv[4], &loaded) != RPOCR_OK) {
    fprintf(stderr, "load: %s\n", rpocr_last_error());
    rpocr_destroy(ocr);
    return 1;
  }
  rpocr_image image = {loaded.width, loaded.height, loaded.rgb};
  rpocr_result result = {0};
  int found = 0;
  if (rpocr_recognize(ocr, &image, &result) != RPOCR_OK) {
    fprintf(stderr, "recognize: %s\n", rpocr_last_error());
  } else {
    for (size_t i = 0; i < result.count; ++i) {
      printf("%s\t%.4f\n", result.items[i].text, result.items[i].confidence);
      if (strstr(result.items[i].text, argv[5])) found = 1;
    }
    rpocr_result_free(&result);
  }
  rpocr_image_free(&loaded);
  rpocr_destroy(ocr);
  if (!found) {
    fprintf(stderr, "expected text '%s' not found\n", argv[5]);
    return 1;
  }
  printf("sdk_consumer_ok\n");
  return 0;
}
