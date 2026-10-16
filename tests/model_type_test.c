/* model_type option: old struct_size, unset, AUTO, explicit LITE / PPOCR, mismatches, bad value.
   usage: ppocr_model_type_test <lite_dir> <ppocr_det.onnx> <ppocr_rec.onnx> <dict.txt> <image> */
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "ppocr/ppocr.h"

static int failures = 0;
#define CHECK(c, msg) do { if (c) printf("ok   %s\n", msg); else { printf("FAIL %s (%s)\n", msg, ppocr_last_error()); failures++; } } while (0)

static ppocr_image_buffer g_img;
static int lines(ppocr_ocr* o, const char* unused) {
  (void)unused;
  ppocr_image im = {g_img.width, g_img.height, g_img.rgb};
  ppocr_result r; memset(&r, 0, sizeof r);
  if (ppocr_recognize(o, &im, &r) != PPOCR_OK) return -1;
  int n = (int)r.count; ppocr_result_free(&r); return n;
}

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: %s lite_dir det rec dict image\n", argv[0]); return 2; }
  const char *lite = argv[1], *det = argv[2], *rec = argv[3], *dict = argv[4], *img = argv[5];
  if (ppocr_load_ppm(img, &g_img) != PPOCR_OK) { fprintf(stderr, "load: %s\n", ppocr_last_error()); return 2; }
  const size_t old_size = offsetof(ppocr_options, model_type);
  ppocr_options o; ppocr_ocr* h; int ref_lite, ref_pp;

  /* legacy symbol (what old binaries call) writes only the old layout */
  unsigned char buf[sizeof(ppocr_options) + 8]; memset(buf, 0xAB, sizeof buf);
#undef ppocr_options_init
  ppocr_options_init((ppocr_options*)buf);
  CHECK(((ppocr_options*)buf)->struct_size == (int)old_size, "legacy init sets old struct_size");
  CHECK(buf[old_size] == 0xAB && buf[sizeof(ppocr_options)] == 0xAB, "legacy init writes nothing past old layout");
  ppocr_options_init_ex(&o, sizeof o);
  CHECK(o.struct_size == (int)sizeof o && o.model_type == PPOCR_MODEL_AUTO, "init_ex: full size, AUTO default");

  /* NULL options / unset (AUTO) */
  h = ppocr_create(lite, NULL, NULL, NULL); ref_lite = h ? lines(h, img) : -1; ppocr_destroy(h);
  CHECK(ref_lite > 0, "NULL options + lite folder -> lite");
  ppocr_options_init_ex(&o, sizeof o);
  h = ppocr_create(det, rec, dict, &o); ref_pp = h ? lines(h, img) : -1; ppocr_destroy(h);
  CHECK(ref_pp > 0, "unset model_type + onnx files -> PP-OCR");

  /* old struct_size, garbage in the model_type slot must be ignored */
  memcpy(buf, &o, old_size); ((ppocr_options*)buf)->struct_size = (int)old_size;
  ((ppocr_options*)buf)->model_type = 12345;
  h = ppocr_create(det, rec, dict, (ppocr_options*)buf);
  CHECK(h && lines(h, img) == ref_pp, "old struct_size + PP-OCR files: as before"); ppocr_destroy(h);
  h = ppocr_create(lite, NULL, NULL, (ppocr_options*)buf);
  CHECK(h && lines(h, img) == ref_lite, "old struct_size + lite folder: AUTO"); ppocr_destroy(h);

  /* explicit */
  o.model_type = PPOCR_MODEL_AUTO; h = ppocr_create(lite, NULL, NULL, &o);
  CHECK(h && lines(h, img) == ref_lite, "AUTO + lite folder"); ppocr_destroy(h);
  o.model_type = PPOCR_MODEL_LITE; h = ppocr_create(lite, NULL, NULL, &o);
  CHECK(h && lines(h, img) == ref_lite, "LITE + lite folder"); ppocr_destroy(h);
  o.model_type = PPOCR_MODEL_LITE; h = ppocr_create(lite, rec, dict, &o);
  CHECK(h && lines(h, img) == ref_lite, "LITE ignores rec_model/dictionary"); ppocr_destroy(h);
  o.model_type = PPOCR_MODEL_PPOCR; h = ppocr_create(det, rec, dict, &o);
  CHECK(h && lines(h, img) == ref_pp, "PPOCR + onnx files"); ppocr_destroy(h);

  /* mismatches / invalid */
  o.model_type = PPOCR_MODEL_LITE; h = ppocr_create(det, rec, dict, &o);
  CHECK(!h && strstr(ppocr_last_error(), "meta.json"), "LITE + PP-OCR det.onnx -> error"); ppocr_destroy(h);
  o.model_type = PPOCR_MODEL_PPOCR; h = ppocr_create(lite, rec, dict, &o);
  CHECK(!h && strstr(ppocr_last_error(), "lite model folder"), "PPOCR + lite folder -> error"); ppocr_destroy(h);
  o.model_type = PPOCR_MODEL_LITE; h = ppocr_create(NULL, NULL, NULL, &o);
  CHECK(!h, "LITE + NULL det_model -> error"); ppocr_destroy(h);
  o.model_type = 7; h = ppocr_create(lite, NULL, NULL, &o);
  CHECK(!h && strstr(ppocr_last_error(), "model_type"), "invalid model_type -> error"); ppocr_destroy(h);

  printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
  return failures ? 1 : 0;
}
