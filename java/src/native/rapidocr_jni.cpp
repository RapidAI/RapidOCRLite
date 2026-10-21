// JNI bridge for the RapidOCRLite C ABI (librapidocrlite).
//
// Compiled into rapidocrlite_jni.<dll|so|dylib>, which links against the
// platform's SDK engine library and needs it at runtime. The file is
// portable C++ (no platform-specific APIs); per-platform build commands
// live in build.sh. RPOCR_SHARED makes export.h expand RPOCR_API to
// dllimport, matching a normal DLL consumer on Windows.

#include <jni.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "rapidocrlite/rapidocrlite.h"

namespace {

jclass g_textLineCls = nullptr;
jmethodID g_textLineCtor = nullptr;
jclass g_floatArrayCls = nullptr;
jclass g_textLineArrayCls = nullptr;
jclass g_cpuInfoCls = nullptr;
jmethodID g_cpuInfoCtor = nullptr;

void ThrowIllegal(JNIEnv* env, const char* msg) {
  jclass cls = env->FindClass("java/lang/IllegalStateException");
  if (cls) {
    env->ThrowNew(cls, msg);
  }
}

// Reads a java String into a C string; caller must ReleaseStringUTFChars.
// Returns nullptr only when the java string is null (release is a no-op then).
const char* GetUtf(JNIEnv* env, jstring s) {
  return s == nullptr ? nullptr : env->GetStringUTFChars(s, nullptr);
}

void ReleaseUtf(JNIEnv* env, jstring s, const char* p) {
  if (s && p) {
    env->ReleaseStringUTFChars(s, p);
  }
}

jfloatArray NewFloatArray2(JNIEnv* env, float x, float y) {
  jfloatArray a = env->NewFloatArray(2);
  if (!a) return nullptr;
  jfloat buf[2] = {x, y};
  env->SetFloatArrayRegion(a, 0, 2, buf);
  return a;
}

// 4x2 corner array (TL, TR, BR, BL) as a Java float[][].
jobjectArray NewBox2D(JNIEnv* env, const rpocr_point box[4]) {
  jobjectArray outer = env->NewObjectArray(4, g_floatArrayCls, nullptr);
  if (!outer) return nullptr;
  for (int i = 0; i < 4; ++i) {
    jfloatArray row = NewFloatArray2(env, box[i].x, box[i].y);
    if (!row) return nullptr;
    env->SetObjectArrayElement(outer, i, row);
    env->DeleteLocalRef(row);
  }
  return outer;
}

jobject NewTextLine(JNIEnv* env, const rpocr_text_box& item, int cls_label, float cls_score) {
  jstring text = env->NewStringUTF(item.text ? item.text : "");
  if (!text) return nullptr;
  jobjectArray box = NewBox2D(env, item.box);
  if (!box) {
    env->DeleteLocalRef(text);
    return nullptr;
  }
  jobject obj = env->NewObject(g_textLineCls, g_textLineCtor, text,
                               (jfloat)item.confidence, box, (jint)item.bbox_x,
                               (jint)item.bbox_y, (jint)item.bbox_w,
                               (jint)item.bbox_h, (jint)cls_label, (jfloat)cls_score);
  env->DeleteLocalRef(box);
  env->DeleteLocalRef(text);
  return obj;
}

jobjectArray ResultToArray(JNIEnv* env, const rpocr_result& result) {
  jobjectArray arr = env->NewObjectArray((jsize)result.count, g_textLineCls, nullptr);
  if (!arr) return nullptr;
  for (size_t i = 0; i < result.count; ++i) {
    int cls_label = -1;
    float cls_score = 0.f;
    rpocr_result_cls(&result, i, &cls_label, &cls_score);
    jobject line = NewTextLine(env, result.items[i], cls_label, cls_score);
    if (!line) return nullptr;
    env->SetObjectArrayElement(arr, (jsize)i, line);
    env->DeleteLocalRef(line);
  }
  return arr;
}

// options layout must match OcrOptions.nativeArgs():
// backend, detLimitSideLen, recParallelism, detThreshold, detBoxThreshold, detUnclipRatio,
// useCls (1/0), clsThresh (NaN = default)
void ApplyOptions(rpocr_options& opt, const double* a, jsize n) {
  if (!a || n < 6) return;
  opt.backend = (int)a[0];
  if (a[1] > 0) opt.det_limit_side_len = (int)a[1];
  if (a[2] > 0) opt.rec_parallelism = (int)a[2];
  if (!std::isnan(a[3])) opt.det_threshold = (float)a[3];
  if (!std::isnan(a[4])) opt.det_box_threshold = (float)a[4];
  if (!std::isnan(a[5])) opt.det_unclip_ratio = (float)a[5];
  if (n > 6 && !std::isnan(a[6])) opt.use_cls = a[6] != 0.0 ? 1 : 0;
  if (n > 7 && !std::isnan(a[7])) opt.cls_thresh = (float)a[7];
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void* reserved) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8) != JNI_OK) {
    return JNI_ERR;
  }
  jclass local = env->FindClass("ai/rapidocr/lite/TextLine");
  if (!local) return JNI_ERR;
  g_textLineCls = (jclass)env->NewGlobalRef(local);
  env->DeleteLocalRef(local);
  g_textLineCtor = env->GetMethodID(g_textLineCls, "<init>",
                                    "(Ljava/lang/String;F[[FIIIIIF)V");
  if (!g_textLineCtor) return JNI_ERR;

  local = env->FindClass("[F");
  if (!local) return JNI_ERR;
  g_floatArrayCls = (jclass)env->NewGlobalRef(local);
  env->DeleteLocalRef(local);

  local = env->FindClass("[Lai/rapidocr/lite/TextLine;");
  if (!local) return JNI_ERR;
  g_textLineArrayCls = (jclass)env->NewGlobalRef(local);
  env->DeleteLocalRef(local);

  local = env->FindClass("ai/rapidocr/lite/CpuInfo");
  if (!local) return JNI_ERR;
  g_cpuInfoCls = (jclass)env->NewGlobalRef(local);
  env->DeleteLocalRef(local);
  g_cpuInfoCtor = env->GetMethodID(g_cpuInfoCls, "<init>",
                                   "(ZZZILjava/lang/String;)V");
  if (!g_cpuInfoCtor) return JNI_ERR;

  return JNI_VERSION_1_8;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* vm, void* reserved) {
  JNIEnv* env = nullptr;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_8) != JNI_OK) {
    return;
  }
  if (g_textLineCls) env->DeleteGlobalRef(g_textLineCls);
  if (g_floatArrayCls) env->DeleteGlobalRef(g_floatArrayCls);
  if (g_textLineArrayCls) env->DeleteGlobalRef(g_textLineArrayCls);
  if (g_cpuInfoCls) env->DeleteGlobalRef(g_cpuInfoCls);
  g_textLineCls = nullptr;
  g_floatArrayCls = nullptr;
  g_textLineArrayCls = nullptr;
  g_cpuInfoCls = nullptr;
}

// ---------------------------------------------------------------------------
// engine lifecycle
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT jlong JNICALL
Java_ai_rapidocr_lite_RapidOcr_nativeCreatePPOCR(JNIEnv* env, jclass,
                                                 jstring det, jstring rec,
                                                 jstring dict) {
  const char* d = GetUtf(env, det);
  const char* r = GetUtf(env, rec);
  const char* t = GetUtf(env, dict);

  rpocr_options opt;
  rpocr_options_init(&opt);
  opt.backend = RPOCR_BACKEND_CPU;
  opt.model_type = RPOCR_MODEL_PPOCR;

  rpocr_ocr* ocr = rpocr_create(d, r, t, &opt);
  if (!ocr) {
    ThrowIllegal(env, rpocr_last_error());
  }

  ReleaseUtf(env, det, d);
  ReleaseUtf(env, rec, r);
  ReleaseUtf(env, dict, t);
  return (jlong)(intptr_t)ocr;
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_rapidocr_lite_RapidOcr_nativeCreatePPOCR__Ljava_lang_String_2Ljava_lang_String_2Ljava_lang_String_2_3D(
    JNIEnv* env, jclass, jstring det, jstring rec, jstring dict,
    jdoubleArray options) {
  const char* d = GetUtf(env, det);
  const char* r = GetUtf(env, rec);
  const char* t = GetUtf(env, dict);

  jdouble* opts = options ? env->GetDoubleArrayElements(options, nullptr) : nullptr;

  rpocr_options opt;
  rpocr_options_init(&opt);
  opt.backend = RPOCR_BACKEND_CPU;
  opt.model_type = RPOCR_MODEL_PPOCR;
  ApplyOptions(opt, opts, options ? env->GetArrayLength(options) : 0);

  rpocr_ocr* ocr = rpocr_create(d, r, t, &opt);
  if (!ocr) {
    ThrowIllegal(env, rpocr_last_error());
  }

  if (options && opts) {
    env->ReleaseDoubleArrayElements(options, opts, JNI_ABORT);
  }
  ReleaseUtf(env, det, d);
  ReleaseUtf(env, rec, r);
  ReleaseUtf(env, dict, t);
  return (jlong)(intptr_t)ocr;
}

// PP-OCRv6 engine with the text-line orientation classifier model
// (ch_ppocr_mobile_v2.0_cls_mobile.onnx, rapidocr main Global.use_cls).
extern "C" JNIEXPORT jlong JNICALL
Java_ai_rapidocr_lite_RapidOcr_nativeCreatePPOCRCls(JNIEnv* env, jclass, jstring det, jstring rec,
                                                    jstring dict, jdoubleArray options,
                                                    jstring cls) {
  const char* d = GetUtf(env, det);
  const char* r = GetUtf(env, rec);
  const char* t = GetUtf(env, dict);
  const char* c = GetUtf(env, cls);

  jdouble* opts = options ? env->GetDoubleArrayElements(options, nullptr) : nullptr;

  rpocr_options opt;
  rpocr_options_init(&opt);
  opt.backend = RPOCR_BACKEND_CPU;
  opt.model_type = RPOCR_MODEL_PPOCR;
  ApplyOptions(opt, opts, options ? env->GetArrayLength(options) : 0);
  opt.cls_model = c;

  rpocr_ocr* ocr = rpocr_create(d, r, t, &opt);
  if (!ocr) {
    ThrowIllegal(env, rpocr_last_error());
  }

  if (options && opts) {
    env->ReleaseDoubleArrayElements(options, opts, JNI_ABORT);
  }
  ReleaseUtf(env, det, d);
  ReleaseUtf(env, rec, r);
  ReleaseUtf(env, dict, t);
  ReleaseUtf(env, cls, c);
  return (jlong)(intptr_t)ocr;
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_rapidocr_lite_RapidOcr_nativeCreateLite(JNIEnv* env, jclass,
                                                jstring dir) {
  const char* d = GetUtf(env, dir);

  rpocr_options opt;
  rpocr_options_init(&opt);
  opt.backend = RPOCR_BACKEND_CPU;
  opt.model_type = RPOCR_MODEL_LITE;

  rpocr_ocr* ocr = rpocr_create(d, nullptr, nullptr, &opt);
  if (!ocr) {
    ThrowIllegal(env, rpocr_last_error());
  }

  ReleaseUtf(env, dir, d);
  return (jlong)(intptr_t)ocr;
}

extern "C" JNIEXPORT jlong JNICALL
Java_ai_rapidocr_lite_RapidOcr_nativeCreateLite__Ljava_lang_String_2_3D(
    JNIEnv* env, jclass, jstring dir, jdoubleArray options) {
  const char* d = GetUtf(env, dir);

  jdouble* opts = options ? env->GetDoubleArrayElements(options, nullptr) : nullptr;

  rpocr_options opt;
  rpocr_options_init(&opt);
  opt.backend = RPOCR_BACKEND_CPU;
  opt.model_type = RPOCR_MODEL_LITE;
  ApplyOptions(opt, opts, options ? env->GetArrayLength(options) : 0);

  rpocr_ocr* ocr = rpocr_create(d, nullptr, nullptr, &opt);
  if (!ocr) {
    ThrowIllegal(env, rpocr_last_error());
  }

  if (options && opts) {
    env->ReleaseDoubleArrayElements(options, opts, JNI_ABORT);
  }
  ReleaseUtf(env, dir, d);
  return (jlong)(intptr_t)ocr;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_rapidocr_lite_RapidOcr_nativeDestroy(JNIEnv*, jclass, jlong handle) {
  rpocr_ocr* ocr = (rpocr_ocr*)(intptr_t)handle;
  if (ocr) {
    rpocr_destroy(ocr);
  }
}

// ---------------------------------------------------------------------------
// recognition
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT jobjectArray JNICALL
Java_ai_rapidocr_lite_RapidOcr_nativeRecognize(JNIEnv* env, jclass, jlong handle,
                                               jbyteArray rgb, jint width,
                                               jint height) {
  rpocr_ocr* ocr = (rpocr_ocr*)(intptr_t)handle;
  if (!ocr) {
    ThrowIllegal(env, "engine handle is null");
    return nullptr;
  }
  jbyte* px = env->GetByteArrayElements(rgb, nullptr);
  if (!px) {
    return nullptr;
  }

  rpocr_image image;
  image.width = width;
  image.height = height;
  image.rgb = (const uint8_t*)px;

  rpocr_result result;
  result.items = nullptr;
  result.count = 0;

  jobjectArray arr = nullptr;
  if (rpocr_recognize(ocr, &image, &result) == RPOCR_OK) {
    arr = ResultToArray(env, result);
  } else {
    ThrowIllegal(env, rpocr_last_error());
  }
  rpocr_result_free(&result);
  env->ReleaseByteArrayElements(rgb, px, JNI_ABORT);
  return arr;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_ai_rapidocr_lite_RapidOcr_nativeRecognizePpm(JNIEnv* env, jclass,
                                                  jlong handle, jstring path) {
  rpocr_ocr* ocr = (rpocr_ocr*)(intptr_t)handle;
  if (!ocr) {
    ThrowIllegal(env, "engine handle is null");
    return nullptr;
  }
  const char* p = GetUtf(env, path);

  rpocr_image_buffer buf;
  buf.width = 0;
  buf.height = 0;
  buf.rgb = nullptr;

  jobjectArray arr = nullptr;
  if (rpocr_load_ppm(p, &buf) != RPOCR_OK) {
    ThrowIllegal(env, rpocr_last_error());
  } else {
    rpocr_image image;
    image.width = buf.width;
    image.height = buf.height;
    image.rgb = buf.rgb;
    rpocr_result result;
    result.items = nullptr;
    result.count = 0;
    if (rpocr_recognize(ocr, &image, &result) == RPOCR_OK) {
      arr = ResultToArray(env, result);
    } else {
      ThrowIllegal(env, rpocr_last_error());
    }
    rpocr_result_free(&result);
    rpocr_image_free(&buf);
  }

  ReleaseUtf(env, path, p);
  return arr;
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_ai_rapidocr_lite_RapidOcr_nativeRecognizeBatch(JNIEnv* env, jclass,
                                                    jlong handle,
                                                    jobjectArray images,
                                                    jintArray widths,
                                                    jintArray heights) {
  rpocr_ocr* ocr = (rpocr_ocr*)(intptr_t)handle;
  if (!ocr) {
    ThrowIllegal(env, "engine handle is null");
    return nullptr;
  }
  const jsize n = env->GetArrayLength(images);
  if (n == 0 || env->GetArrayLength(widths) != n ||
      env->GetArrayLength(heights) != n) {
    ThrowIllegal(env, "batch arrays must be non-empty and equal length");
    return nullptr;
  }

  jint* ws = env->GetIntArrayElements(widths, nullptr);
  jint* hs = env->GetIntArrayElements(heights, nullptr);
  std::vector<jbyteArray> arrays((size_t)n, nullptr);
  std::vector<jbyte*> px((size_t)n, nullptr);
  std::vector<rpocr_image> imgs((size_t)n);
  bool bad = false;

  for (jsize i = 0; i < n; ++i) {
    arrays[(size_t)i] = (jbyteArray)env->GetObjectArrayElement(images, i);
    if (!arrays[(size_t)i]) {
      bad = true;
      break;
    }
    px[(size_t)i] = env->GetByteArrayElements(arrays[(size_t)i], nullptr);
    if (!px[(size_t)i]) {
      bad = true;
      break;
    }
    imgs[(size_t)i].width = ws[i];
    imgs[(size_t)i].height = hs[i];
    imgs[(size_t)i].rgb = (const uint8_t*)px[(size_t)i];
  }

  jobjectArray outer = nullptr;
  if (!bad) {
    std::vector<rpocr_result> outs((size_t)n);
    for (size_t i = 0; i < (size_t)n; ++i) {
      outs[i].items = nullptr;
      outs[i].count = 0;
    }
    if (rpocr_recognize_batch(ocr, imgs.data(), (size_t)n, outs.data()) == RPOCR_OK) {
      outer = env->NewObjectArray(n, g_textLineArrayCls, nullptr);
      if (outer) {
        for (jsize i = 0; i < n; ++i) {
          jobjectArray page = ResultToArray(env, outs[(size_t)i]);
          if (!page) {
            outer = nullptr;
            break;
          }
          env->SetObjectArrayElement(outer, i, page);
          env->DeleteLocalRef(page);
        }
      }
    } else {
      ThrowIllegal(env, rpocr_last_error());
    }
    for (size_t i = 0; i < (size_t)n; ++i) {
      rpocr_result_free(&outs[i]);
    }
  } else {
    ThrowIllegal(env, "failed to read batch input arrays");
  }

  for (jsize i = 0; i < n; ++i) {
    if (arrays[(size_t)i]) {
      if (px[(size_t)i]) {
        env->ReleaseByteArrayElements(arrays[(size_t)i], px[(size_t)i], JNI_ABORT);
      }
      env->DeleteLocalRef(arrays[(size_t)i]);
    }
  }
  if (ws) env->ReleaseIntArrayElements(widths, ws, JNI_ABORT);
  if (hs) env->ReleaseIntArrayElements(heights, hs, JNI_ABORT);
  return outer;
}

// ---------------------------------------------------------------------------
// info
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT jstring JNICALL
Java_ai_rapidocr_lite_RapidOcr_nativeLastError(JNIEnv* env, jclass) {
  const char* e = rpocr_last_error();
  return env->NewStringUTF(e ? e : "");
}

extern "C" JNIEXPORT jobject JNICALL
Java_ai_rapidocr_lite_RapidOcr_nativeCpuInfo(JNIEnv* env, jclass) {
  rpocr_cpu_info info;
  std::memset(&info, 0, sizeof(info));
  rpocr_query_cpu_info(&info);
  jstring isa = env->NewStringUTF(info.active_isa);
  if (!isa) return nullptr;
  jobject obj = env->NewObject(
      g_cpuInfoCls, g_cpuInfoCtor, (jboolean)(info.avx2_compiled != 0),
      (jboolean)(info.avx512_compiled != 0), (jboolean)(info.neon_compiled != 0),
      (jint)info.threads, isa);
  env->DeleteLocalRef(isa);
  return obj;
}
