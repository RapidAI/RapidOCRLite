/* model_type option: old struct_size, unset, AUTO, explicit LITE / PPOCR, mismatches, bad value.
   usage: ppocr_model_type_test <lite_dir> <ppocr_det.onnx> <ppocr_rec.onnx> <dict.txt> <image> */
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "rapidocrlite/rapidocrlite.h"

static int failures = 0;
#define CHECK(c, msg) do { if (c) printf("ok   %s\n", msg); else { printf("FAIL %s (%s)\n", msg, rpocr_last_error()); failures++; } } while (0)

static rpocr_image_buffer g_img;
static int lines(rpocr_ocr* o, const char* unused) {
  (void)unused;
  rpocr_image im = {g_img.width, g_img.height, g_img.rgb};
  rpocr_result r; memset(&r, 0, sizeof r);
  if (rpocr_recognize(o, &im, &r) != RPOCR_OK) return -1;
  int n = (int)r.count; rpocr_result_free(&r); return n;
}

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: %s lite_dir det rec dict image\n", argv[0]); return 2; }
  const char *lite = argv[1], *det = argv[2], *rec = argv[3], *dict = argv[4], *img = argv[5];
  if (rpocr_load_ppm(img, &g_img) != RPOCR_OK) { fprintf(stderr, "load: %s\n", rpocr_last_error()); return 2; }
  const size_t old_size = offsetof(rpocr_options, model_type);
  rpocr_options o; rpocr_ocr* h; int ref_lite, ref_pp;

  /* legacy symbol (what old binaries call) writes only the old layout */
  unsigned char buf[sizeof(rpocr_options) + 8]; memset(buf, 0xAB, sizeof buf);
#undef rpocr_options_init
  rpocr_options_init((rpocr_options*)buf);
  CHECK(((rpocr_options*)buf)->struct_size == (int)old_size, "legacy init sets old struct_size");
  CHECK(buf[old_size] == 0xAB && buf[sizeof(rpocr_options)] == 0xAB, "legacy init writes nothing past old layout");
  rpocr_options_init_ex(&o, sizeof o);
  CHECK(o.struct_size == (int)sizeof o && o.model_type == RPOCR_MODEL_AUTO, "init_ex: full size, AUTO default");

  /* NULL options / unset (AUTO) */
  h = rpocr_create(lite, NULL, NULL, NULL); ref_lite = h ? lines(h, img) : -1; rpocr_destroy(h);
  CHECK(ref_lite > 0, "NULL options + lite folder -> lite");
  rpocr_options_init_ex(&o, sizeof o);
  h = rpocr_create(det, rec, dict, &o); ref_pp = h ? lines(h, img) : -1; rpocr_destroy(h);
  CHECK(ref_pp > 0, "unset model_type + onnx files -> PP-OCR");

  /* old struct_size, garbage in the model_type slot must be ignored */
  memcpy(buf, &o, old_size); ((rpocr_options*)buf)->struct_size = (int)old_size;
  ((rpocr_options*)buf)->model_type = 12345;
  h = rpocr_create(det, rec, dict, (rpocr_options*)buf);
  CHECK(h && lines(h, img) == ref_pp, "old struct_size + PP-OCR files: as before"); rpocr_destroy(h);
  h = rpocr_create(lite, NULL, NULL, (rpocr_options*)buf);
  CHECK(h && lines(h, img) == ref_lite, "old struct_size + lite folder: AUTO"); rpocr_destroy(h);

  /* explicit */
  o.model_type = RPOCR_MODEL_AUTO; h = rpocr_create(lite, NULL, NULL, &o);
  CHECK(h && lines(h, img) == ref_lite, "AUTO + lite folder"); rpocr_destroy(h);
  o.model_type = RPOCR_MODEL_LITE; h = rpocr_create(lite, NULL, NULL, &o);
  CHECK(h && lines(h, img) == ref_lite, "LITE + lite folder"); rpocr_destroy(h);
  o.model_type = RPOCR_MODEL_LITE; h = rpocr_create(lite, rec, dict, &o);
  CHECK(h && lines(h, img) == ref_lite, "LITE ignores rec_model/dictionary"); rpocr_destroy(h);
  o.model_type = RPOCR_MODEL_PPOCR; h = rpocr_create(det, rec, dict, &o);
  CHECK(h && lines(h, img) == ref_pp, "PPOCR + onnx files"); rpocr_destroy(h);

  /* mismatches / invalid */
  o.model_type = RPOCR_MODEL_LITE; h = rpocr_create(det, rec, dict, &o);
  CHECK(!h && strstr(rpocr_last_error(), "meta.json"), "LITE + PP-OCR det.onnx -> error"); rpocr_destroy(h);
  o.model_type = RPOCR_MODEL_PPOCR; h = rpocr_create(lite, rec, dict, &o);
  CHECK(!h && strstr(rpocr_last_error(), "lite model folder"), "PPOCR + lite folder -> error"); rpocr_destroy(h);
  o.model_type = RPOCR_MODEL_LITE; h = rpocr_create(NULL, NULL, NULL, &o);
  CHECK(!h, "LITE + NULL det_model -> error"); rpocr_destroy(h);
  o.model_type = 7; h = rpocr_create(lite, NULL, NULL, &o);
  CHECK(!h && strstr(rpocr_last_error(), "model_type"), "invalid model_type -> error"); rpocr_destroy(h);

  printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED", failures);
  return failures ? 1 : 0;
}
