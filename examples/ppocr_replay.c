/* Replays inputs captured by tests/cross/capture_cases.py through the C API
   and prints "score<TAB>text" per line. Used to check cross builds (e.g.
   loongarch64 under qemu) against the host reference texts.
   usage: ppocr_replay DET REC DICT options.bin case1.case [case2.case ...] */
#include "rapidocrlite/rapidocrlite.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void* read_all(const char* path, long* size) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  *size = ftell(f);
  fseek(f, 0, SEEK_SET);
  void* data = malloc((size_t)*size);
  if (data && fread(data, 1, (size_t)*size, f) != (size_t)*size) {
    free(data);
    data = NULL;
  }
  fclose(f);
  return data;
}

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: ppocr_replay DET REC DICT options.bin CASE...\n");
    return 2;
  }
  rpocr_options options;
  rpocr_options_init(&options);
  long size = 0;
  unsigned char* raw = (unsigned char*)read_all(argv[4], &size);
  if (!raw) return fprintf(stderr, "cannot read %s\n", argv[4]), 1;
  /* Same LP64 little-endian layout on the capture host and the target. */
  memcpy(&options, raw, (size_t)size < sizeof(options) ? (size_t)size : sizeof(options));
  options.struct_size = (int)((size_t)size < sizeof(options) ? (size_t)size : sizeof(options));
  options.backend = RPOCR_BACKEND_CPU;
  options.cls_model = NULL; /* a host pointer in the capture; never dereference it */
  free(raw);
  {
    rpocr_cpu_info cpu;
    rpocr_query_cpu_info(&cpu);
    fprintf(stderr, "active_isa %s\n", cpu.active_isa);
  }
  rpocr_ocr* ocr = rpocr_create(argv[1], argv[2], argv[3], &options);
  if (!ocr) return fprintf(stderr, "create: %s\n", rpocr_last_error()), 1;
  for (int i = 5; i < argc; ++i) {
    unsigned char* data = (unsigned char*)read_all(argv[i], &size);
    if (!data || size < 24) return fprintf(stderr, "cannot read %s\n", argv[i]), 1;
    int32_t w, h, tasks;
    float thr[3];
    memcpy(&w, data, 4);
    memcpy(&h, data + 4, 4);
    memcpy(&tasks, data + 8, 4);
    memcpy(thr, data + 12, 12);
    rpocr_set_det_thresholds(ocr, thr[0], thr[1], thr[2]);
    rpocr_image image = {w, h, data + 24};
    rpocr_result result;
    int status = tasks == (RPOCR_TASK_DET | RPOCR_TASK_REC)
                     ? rpocr_recognize(ocr, &image, &result)
                     : rpocr_run(ocr, &image, tasks, &result);
    if (status != RPOCR_OK) return fprintf(stderr, "run: %s\n", rpocr_last_error()), 1;
    printf("== %s %d\n", argv[i], result.count);
    for (int k = 0; k < result.count; ++k)
      printf("%.6f\t%s\n", result.items[k].confidence, result.items[k].text ? result.items[k].text : "");
    rpocr_result_free(&result);
    free(data);
  }
  rpocr_destroy(ocr);
  return 0;
}
