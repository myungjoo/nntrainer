// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright 2023. Umberto Michieli <u.michieli@samsung.com>.
 * Copyright 2023. Kirill Paramonov <k.paramonov@samsung.com>.
 * Copyright 2023. Mete Ozay <m.ozay@samsung.com>.
 * Copyright 2023. JIJOONG MOON <jijoong.moon@samsung.com>.
 * Copyright 2023. HS.Kim <hs0207.kim@samsung.com>
 *
 * @file   simpleshot_jni.cpp
 * @date   24 Oct 2023
 * @brief  JNI apis for image recognition/detection
 * @author Umberto Michieli(u.michieli@samsung.com)
 * @author Kirill Paramonov(k.paramonov@samsung.com)
 * @author Mete Ozay(m.ozay@samsung.com)
 * @author JIJOONG MOON(jijoong.moon@samsung.com)
 * @author HS.Kim (hs0207.kim@samsung.com)
 * @bug    No known bugs
 */
#include "simpleshot_jni.h"
#include "simpleshot.h"
#include <android/bitmap.h>
#include <app_context.h>

namespace {
JavaVM *g_jvm = nullptr;
} // anonymous namespace

/**
 * Cache the JavaVM and force AppContext::Global() initialization at
 * library-load time. Without this, the first JVM thread to call into
 * nntrainer races the std::call_once that registers ~100 layer/optimizer
 * factories — a race that is benign on Linux (single-threaded main()
 * triggers init) but exposed under ART where dozens of pre-existing
 * threads may concurrently enter the .so.
 */
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void * /*reserved*/) {
  g_jvm = vm;
  try {
    (void)nntrainer::AppContext::Global();
  } catch (...) {
    // AppContext::initialize() is noexcept today; defensive catch so
    // JNI_OnLoad never propagates an exception across the JNI boundary.
  }
  return JNI_VERSION_1_6;
}

int cur_epoch = 0;
float val_accu = 0.0;

JNIEXPORT jlong JNICALL
Java_com_samsung_android_nndetector_MainActivity_createRecognitionModel(
  JNIEnv *env, jobject j_obj, jobjectArray args) {

  const int argc = env->GetArrayLength(args);
  char **argv = new char *[argc];
  for (unsigned int i = 0; i < argc; ++i) {
    jstring j_str = (jstring)(env->GetObjectArrayElement(args, i));
    const char *str = env->GetStringUTFChars(j_str, 0);
    size_t str_len = strlen(str);
    argv[i] = new char[str_len + 1];
    strcpy(argv[i], str);
    env->ReleaseStringUTFChars(j_str, str);
  }
  ml::train::Model *model_ = initialize_rec(argc, argv);

  for (unsigned int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;
  return reinterpret_cast<jlong>(model_);
}

JNIEXPORT jlong JNICALL
Java_com_samsung_android_nndetector_MainActivity_createDetectionModel(
  JNIEnv *env, jobject j_obj, jobjectArray args) {

  const int argc = env->GetArrayLength(args);
  char **argv = new char *[argc];
  for (unsigned int i = 0; i < argc; ++i) {
    jstring j_str = (jstring)(env->GetObjectArrayElement(args, i));
    const char *str = env->GetStringUTFChars(j_str, 0);
    size_t str_len = strlen(str);
    argv[i] = new char[str_len + 1];
    strcpy(argv[i], str);
    env->ReleaseStringUTFChars(j_str, str);
  }
  ml::train::Model *model_ = initialize_det(argc, argv);

  for (unsigned int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;
  return reinterpret_cast<jlong>(model_);
}

JNIEXPORT jint JNICALL
Java_com_samsung_android_nndetector_NNImageTrainer_trainPrototypes(
  JNIEnv *env, jobject j_obj, jobjectArray args, jlong det_model_pointer,
  jlong rec_model_pointer) {

  const int argc = env->GetArrayLength(args);
  char **argv = new char *[argc];
  for (unsigned int i = 0; i < argc; ++i) {
    jstring j_str = (jstring)(env->GetObjectArrayElement(args, i));
    const char *str = env->GetStringUTFChars(j_str, 0);
    size_t str_len = strlen(str);
    argv[i] = new char[str_len + 1];
    strcpy(argv[i], str);
    env->ReleaseStringUTFChars(j_str, str);
  }
  ml::train::Model *det_model_ =
    reinterpret_cast<ml::train::Model *>(det_model_pointer);
  ml::train::Model *rec_model_ =
    reinterpret_cast<ml::train::Model *>(rec_model_pointer);

  train_prototypes(argc, argv, det_model_, rec_model_);

  for (unsigned int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;
  return 0;
}

JNIEXPORT jstring JNICALL
Java_com_samsung_android_nndetector_NNImageTester_testPrototypes(
  JNIEnv *env, jobject j_obj, jobjectArray args, jlong det_model_pointer,
  jlong rec_model_pointer) {

  const int argc = env->GetArrayLength(args);
  char **argv = new char *[argc];
  for (unsigned int i = 0; i < argc; ++i) {
    jstring j_str = (jstring)(env->GetObjectArrayElement(args, i));
    const char *str = env->GetStringUTFChars(j_str, 0);
    size_t str_len = strlen(str);
    argv[i] = new char[str_len + 1];
    strcpy(argv[i], str);
    env->ReleaseStringUTFChars(j_str, str);
  }

  ml::train::Model *det_model_ =
    reinterpret_cast<ml::train::Model *>(det_model_pointer);
  ml::train::Model *rec_model_ =
    reinterpret_cast<ml::train::Model *>(rec_model_pointer);

  std::string result = test_prototypes(argc, argv, det_model_, rec_model_);
  jstring ret = (env)->NewStringUTF(result.c_str());

  for (unsigned int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;
  return ret;
}

JNIEXPORT jstring JNICALL
Java_com_samsung_android_nndetector_NNImageAnalyzer_runDetector(
  JNIEnv *env, jobject j_obj, jobjectArray args, jlong det_model_pointer) {
  const int argc = env->GetArrayLength(args);
  char **argv = new char *[argc];
  for (unsigned int i = 0; i < argc; ++i) {
    jstring j_str = (jstring)(env->GetObjectArrayElement(args, i));
    const char *str = env->GetStringUTFChars(j_str, 0);
    size_t str_len = strlen(str);
    argv[i] = new char[str_len + 1];
    strcpy(argv[i], str);
    env->ReleaseStringUTFChars(j_str, str);
  }

  ml::train::Model *det_model_ =
    reinterpret_cast<ml::train::Model *>(det_model_pointer);

  std::string result = run_detector(argc, argv, det_model_);
  jstring ret = (env)->NewStringUTF(result.c_str());

  for (unsigned int i = 0; i < argc; ++i) {
    delete[] argv[i];
  }
  delete[] argv;
  return ret;
}
