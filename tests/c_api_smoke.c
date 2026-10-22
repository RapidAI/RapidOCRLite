#include "rapidocrlite/rapidocrlite.h"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int g_failures = 0;

static void Expect(int condition, const char* name) {
  if (!condition) {
    fprintf(stderr, "FAIL %s\n", name);
    ++g_failures;
  } else {
    printf("ok %s\n", name);
  }
}

int main(void) {
  rpocr_cpu_info cpu;
  memset(&cpu, 0, sizeof(cpu));
  rpocr_query_cpu_info(&cpu);
  Expect(cpu.active_isa[0] != '\0', "cpu info isa");
  Expect(cpu.threads > 0, "cpu info threads");
  Expect(!(cpu.avx512 && !cpu.avx2), "avx512 implies avx2");

  rpocr_options options;
  rpocr_options_init(&options);
  Expect(options.struct_size == (int)sizeof(rpocr_options), "options size");
  Expect(options.rec_batch_size > 0, "default batch");
  Expect(options.backend == RPOCR_BACKEND_HYBRID, "default backend");
  Expect(options.use_cls == 1, "default use_cls on");
  Expect(options.cls_thresh > 0.89f && options.cls_thresh < 0.91f, "default cls_thresh");
  Expect(options.cls_batch_size == 6 && options.cls_height == 48 && options.cls_width == 192,
         "default cls geometry");
  Expect(options.cls_crop_long_lines == 1 && options.cls_model == NULL, "default cls crop/model");
  {
    /* A caller built before use_cls existed: init_ex must not write past its size. */
    rpocr_options old;
    memset(&old, 0x5a, sizeof(old));
    rpocr_options_init_ex(&old, offsetof(rpocr_options, use_cls));
    Expect(old.struct_size == (int)offsetof(rpocr_options, use_cls), "old size kept");
    Expect(*(const unsigned char*)&old.use_cls == 0x5a, "old size untouched tail");
  }

  Expect(rpocr_create(NULL, "a", "b", NULL) == NULL, "null det");
  Expect(rpocr_last_error()[0] != '\0', "null det error");
  Expect(rpocr_create("missing-det.onnx", "missing-rec.onnx", "missing-dict.txt", &options) == NULL,
         "missing files");

  rpocr_result result = {0};
  Expect(rpocr_recognize(NULL, NULL, &result) == RPOCR_ERR_INVALID_ARGUMENT, "null recognize");
  Expect(rpocr_run(NULL, NULL, RPOCR_TASK_DET, &result) == RPOCR_ERR_INVALID_ARGUMENT, "null run");
  Expect(rpocr_run_ex(NULL, NULL, RPOCR_TASK_CLS, &result) == RPOCR_ERR_INVALID_ARGUMENT, "null run_ex");
  {
    int label = 7;
    float score = 1.f;
    Expect(rpocr_result_cls(&result, 0, &label, &score) == RPOCR_ERR_INVALID_ARGUMENT,
           "cls of empty result");
    rpocr_line_chars chars;
    Expect(rpocr_result_chars(&result, 0, &chars) == RPOCR_ERR_INVALID_ARGUMENT,
           "chars of empty result");
  }
  Expect(rpocr_cls_loaded(NULL) == 0, "cls_loaded null");
  Expect(rpocr_set_rec_order(NULL, NULL, NULL) == RPOCR_ERR_INVALID_ARGUMENT, "rec order null");
  rpocr_image_buffer image = {0};
  Expect(rpocr_load_ppm("missing.ppm", &image) != RPOCR_OK, "missing ppm");

  rpocr_backend_info backend;
  rpocr_query_backend_info(&backend);
  Expect(backend.vulkan_loader_available == 0 || backend.vulkan_loader_available == 1,
         "backend info");

  if (g_failures) {
    fprintf(stderr, "%d failure(s)\n", g_failures);
    return 1;
  }
  printf("c api smoke passed isa=%s\n", cpu.active_isa);
  return 0;
}
