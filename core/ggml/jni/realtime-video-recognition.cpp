#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <unistd.h>
#include <inttypes.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <limits.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/types.h>

extern "C" {
#include "libavutil/avstring.h"
#include "libavutil/eval.h"
#include "libavutil/mathematics.h"
#include "libavutil/pixdesc.h"
#include "libavutil/imgutils.h"
#include "libavutil/dict.h"
#include "libavutil/parseutils.h"
#include "libavutil/avassert.h"
#include "libavutil/time.h"
#include "libavformat/avformat.h"
#include "libavcodec/avcodec.h"
#include "libswscale/swscale.h"
#include "libavcodec/avfft.h"
#include "libswresample/swresample.h"
#include "libavutil/log.h"
#include "libavutil/avutil.h"
#include "libavutil/opt.h"
#include "libavutil/samplefmt.h"
#include "libswresample/swresample.h"
#include "libavutil/myfifo.h"
#include "libavutil/cde_log.h"
#include "libavutil/cde_assert.h"
}

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <tuple>
#include <queue>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <memory>
#include <regex>
#include <random>
#include <functional>
#include <unordered_map>
#include <condition_variable>
#include <cassert>
#include <unordered_set>
#include <utility>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "kantv-asr.h"
#include "ggml-jni.h"

#include "whisper.h"

#include "llama.h"

#include "llamacpp/ggml/include/ggml-hexagon.h"

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>
#include <android/bitmap.h>
#include <android/log.h>

//ncnn
#include "platform.h"
#include "benchmark.h"
#include "net.h"
#include "gpu.h"

//opencv-android
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "ndkcamera.h"

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON


class MyNdkCamera;

static ncnn::Mutex      g_ncnn_lock;
static MyNdkCamera *    g_camera        = nullptr;
static long long        g_bmp_idx       = 0;
static char             g_bmp_filename[JNI_TMP_LEN];

class MyNdkCamera : public NdkCameraWindow {
public:
    virtual void on_image_render(cv::Mat &rgb) const;
};

static int draw_unsupported(cv::Mat &rgb) {
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat & rgb) {
    // resolve moving average
    float avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static float fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f) {
            t0 = t1;
            return 0;
        }

        float fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--) {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f) {
            return 0;
        }

        for (int i = 0; i < 10; i++) {
            avg_fps += fps_history[i];
        }
        avg_fps /= 10.f;
    }

    char text[32];
    snprintf(text, 32,"FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

void MyNdkCamera::on_image_render(cv::Mat & rgb) const {
    g_bmp_idx++;
    //TODO: stability issue between UI layer and native layer
    if (0 == (g_bmp_idx % 100)) { // 100 / 30 ~= 3 seconds
        GGML_JNI_NOTIFY("realtime-cam-reset");
        //snprintf(g_bmp_filename, JNI_TMP_LEN, "/sdcard/bmp-%04d.bmp", g_bmp_idx);
        snprintf(g_bmp_filename, JNI_TMP_LEN, "/sdcard/bmp-tmp.bmp");
        LOGGD("write bmp %s, width %d, height %d\n", g_bmp_filename, rgb.cols, rgb.rows);

        //this is a lazy/dirty method to implement "realtime"(not realtime at the moment) video recognition via MTMD feature in llama.cpp
        //using MTMD API directly is a better approach, can be seen in: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp#L4121
        write_bmp(g_bmp_filename, rgb.cols, rgb.rows, 24, rgb.data);
        inference_init_running_state();
        llava_inference("/sdcard/SmolVLM2-256M-Video-Instruct-f16.gguf",
                        "/sdcard/mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf",
                        g_bmp_filename, "what do you see in this image?",
                        GGML_BENCHMARK_LLM, 8, HEXAGON_BACKEND_GGML, HWACCEL_CDSP);
        inference_reset_running_state();
    }

    draw_fps(rgb);
}


extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM * vm, void * reserved) {
    LOGGD("JNI_OnLoad");

    ncnn::create_gpu_instance();

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM * vm, void * reserved) {
    LOGGD("JNI_OnUnload");
    ncnn::destroy_gpu_instance();
    delete g_camera;
    g_camera = NULL;
}

JNIEXPORT jboolean JNICALL
Java_kantvai_ai_ggmljava_openCamera(JNIEnv * env, jclass clazz, jint facing) {
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    LOGGD("openCamera %d", facing);

    g_camera->open((int) facing);

    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_kantvai_ai_ggmljava_closeCamera(JNIEnv * env, jclass clazz) {
    LOGGD("closeCamera");

    g_camera->close();
}

JNIEXPORT void JNICALL
Java_kantvai_ai_ggmljava_setOutputWindow(JNIEnv * env, jclass clazz, jobject surface) {
    ANativeWindow *win = ANativeWindow_fromSurface(env, surface);

    LOGGD("setOutputWindow %p", win);

    g_camera->set_window(win);
}

} //end extern "C" {