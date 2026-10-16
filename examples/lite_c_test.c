/* Pure C test of the lite (Indra) model through the public C API only.
   usage: lite_c_test <lite_model_dir> <list.txt: "id path.ppm" per line> <threads> <out.json> [warmup=1]
   Writes {"threads":N,"n":..,"mean_ms":..,"median_ms":..,"pages":[{"id","txts","boxes"}]} (boxes: 4 points). */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 199309L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "ppocr/ppocr.h"

static double now_ms(void) {
  struct timespec t;
#ifdef _WIN32
  timespec_get(&t, TIME_UTC);
#else
  clock_gettime(CLOCK_MONOTONIC, &t);  /* bionic has timespec_get only from API 29 */
#endif
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static void put_str(FILE* f, const char* s) {
  fputc('"', f);
  for (; *s; s++) {
    unsigned char c = (unsigned char)*s;
    if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
    else if (c < 0x20) fprintf(f, "\\u%04x", c);
    else fputc(c, f);
  }
  fputc('"', f);
}
static int cmpd(const void* a, const void* b) { double x = *(const double*)a, y = *(const double*)b; return x < y ? -1 : x > y; }

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s model_dir list.txt threads out.json [warmup]\n", argv[0]); return 2; }
  ppocr_options opt; ppocr_options_init(&opt);
  opt.backend = PPOCR_BACKEND_CPU;
  opt.rec_parallelism = atoi(argv[3]);
  ppocr_ocr* ocr = ppocr_create(argv[1], NULL, NULL, &opt);
  if (!ocr) { fprintf(stderr, "create failed: %s\n", ppocr_last_error()); return 1; }
  fprintf(stderr, "note: %s\n", ppocr_backend_note(ocr));
  FILE* lf = fopen(argv[2], "r"); if (!lf) { perror("list"); return 1; }
  char ids[256][64], paths[256][1024]; int n = 0;
  while (n < 256 && fscanf(lf, "%63s %1023s", ids[n], paths[n]) == 2) n++;
  fclose(lf);
  int warm = argc > 5 ? atoi(argv[5]) : 1;
  ppocr_image_buffer* imgs = calloc((size_t)n, sizeof *imgs);
  for (int i = 0; i < n; i++) if (ppocr_load_ppm(paths[i], &imgs[i]) != PPOCR_OK) { fprintf(stderr, "load %s: %s\n", paths[i], ppocr_last_error()); return 1; }
  if (warm && n) {
    ppocr_image im = {imgs[0].width, imgs[0].height, imgs[0].rgb}; ppocr_result r;
    if (ppocr_recognize(ocr, &im, &r) != PPOCR_OK) { fprintf(stderr, "warmup: %s\n", ppocr_last_error()); return 1; }
    ppocr_result_free(&r);
  }
  FILE* f = fopen(argv[4], "w");
  double* ts = calloc((size_t)n, sizeof(double)); double tot = 0;
  fprintf(f, "{\"threads\": %s, \"pages\": [", argv[3]);
  /* LITE_BATCH=1: one ppocr_recognize_batch call over all pages (per-page ms = batch wall time / n). */
  ppocr_result* br = NULL;
  if (getenv("LITE_BATCH") && n) {
    ppocr_image* ims = calloc((size_t)n, sizeof *ims); br = calloc((size_t)n, sizeof *br);
    for (int i = 0; i < n; i++) { ims[i].width = imgs[i].width; ims[i].height = imgs[i].height; ims[i].rgb = imgs[i].rgb; }
    double t0 = now_ms();
    if (ppocr_recognize_batch(ocr, ims, (size_t)n, br) != PPOCR_OK) { fprintf(stderr, "batch: %s\n", ppocr_last_error()); return 1; }
    double per = (now_ms() - t0) / n; free(ims);
    for (int i = 0; i < n; i++) ts[i] = getenv("LITE_NO_TIMING") ? 0 : per;
  }
  for (int i = 0; i < n; i++) {
    ppocr_image im = {imgs[i].width, imgs[i].height, imgs[i].rgb}; ppocr_result r;
    int st = PPOCR_OK;
    if (br) r = br[i];
    else {
      double t0 = now_ms();
      st = ppocr_recognize(ocr, &im, &r);
      ts[i] = getenv("LITE_NO_TIMING") ? 0 : now_ms() - t0;
    }
    tot += ts[i];
    if (st != PPOCR_OK) { fprintf(stderr, "page %s: %s\n", ids[i], ppocr_last_error()); return 1; }
    fprintf(f, "%s{\"id\": ", i ? ", " : ""); put_str(f, ids[i]); fprintf(f, ", \"txts\": [");
    for (size_t k = 0; k < r.count; k++) { if (k) fputs(", ", f); put_str(f, r.items[k].text); }
    fprintf(f, "], \"boxes\": [");
    for (size_t k = 0; k < r.count; k++) {
      fprintf(f, "%s[", k ? ", " : "");
      for (int q = 0; q < 4; q++) fprintf(f, "%s[%.9g, %.9g]", q ? ", " : "", r.items[k].box[q].x, r.items[k].box[q].y);
      fputc(']', f);
    }
    fprintf(f, "], \"ms\": %.3f}", ts[i]);
    ppocr_result_free(&r);
  }
  qsort(ts, (size_t)n, sizeof(double), cmpd);
  double med = n ? (n % 2 ? ts[n / 2] : (ts[n / 2 - 1] + ts[n / 2]) / 2) : 0;
  fprintf(f, "], \"n\": %d, \"mean_ms\": %.3f, \"median_ms\": %.3f}\n", n, n ? tot / n : 0, med);
  fclose(f);
  printf("lite_c_test threads=%s n=%d mean_ms=%.1f median_ms=%.1f\n", argv[3], n, n ? tot / n : 0, med);
  for (int i = 0; i < n; i++) ppocr_image_free(&imgs[i]);
  ppocr_destroy(ocr);
  return 0;
}
