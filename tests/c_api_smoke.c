#include "rapidocrlite/rapidocrlite.h"

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

  Expect(rpocr_create(NULL, "a", "b", NULL) == NULL, "null det");
  Expect(rpocr_last_error()[0] != '\0', "null det error");
  Expect(rpocr_create("missing-det.onnx", "missing-rec.onnx", "missing-dict.txt", &options) == NULL,
         "missing files");

  rpocr_result result = {0};
  Expect(rpocr_recognize(NULL, NULL, &result) == RPOCR_ERR_INVALID_ARGUMENT, "null recognize");
  Expect(rpocr_run(NULL, NULL, RPOCR_TASK_DET, &result) == RPOCR_ERR_INVALID_ARGUMENT, "null run");
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
