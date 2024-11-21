/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    Safety Recognition sample. Demonstrate MobileFaceNet recognition
 *           with antiSpoofing and fingerprint module together
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include <vector>
#include <string>
#include <cinttypes>
#include <deque>

#include "BoardInit.hpp"      /* Board initialisation */
#include "log_macros.h"      /* Logging macros (optional) */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "InputFiles.hpp"             /* Baked-in input (not needed for live data) */
#include "FaceMobileNetModel.hpp"       /* Model API */
#include "Labels.hpp"
#include "Recognizer.hpp"
#include "FaceRecognProcessing.hpp"
#include "FaceDetectorPostProcessing.hpp"

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"
#include "ModelFileReader.h"
#include "ff.h"

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

//#define __PROFILE__
#define __USE_CCAP__
//#define __USE_DISPLAY__
#define __USE_UVC__

//#define __USE_FINGERP__
// choose only one anti-spoof model
//#define __ANTI_MODEL_CNNH__
#define __ANTI_MODEL_MN3S__
#define __USE_RESULT_ACC_WINDOWS__

#include "Profiler.hpp"

#if defined (__USE_CCAP__)
    #include "ImageSensor.h"
#endif

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

#if defined (__USE_FINGERP__)
    #include "driver_as608_basic.h"
#endif

#include "Classifier.hpp"    /* Classifier for the result */
#include "ImgClassProcessing.hpp"
#include "Labels_cls.hpp"
#include "FaceAntiSpoofModel.hpp"

#define NUM_FRAMEBUF 2  //1 or 2
#define FACE_MOBILENET_MODEL_FILE "0:\\face_mobilenet.tflite"
#define FACE_MOBILENET_MODEL_AT_HYPERRAM_ADDR (0x820b0000)
#define FACE_PRESENCE_THRESHOLD                 (0.4)
#define FACE_LABEL_FILE             "0:\\face_labels.txt"

#if defined (__ANTI_MODEL_CNNH__)
#define FACE_ANTISPOOF_MODEL_FILE "0:\\AntiSpoofing_80_cnnhead_vela.tflite"
#define FACE_ANTISPOOF_MODEL_AT_HYPERRAM_ADDR (0x82500000)
static const char *labelsVec[] LABELS_ATTRIBUTE =
{
    "Live Face",
    "Print Attack",
    "Replay Attack"
};
constexpr size_t labelsSz = 3;
const float threshold_FAS = 0.6;
#endif

#if defined (__ANTI_MODEL_MN3S__)
#define FACE_ANTISPOOF_MODEL_FILE "0:\\anti-spoof-mn3_full_integer_quant_per_tensor_vela.tflite"
#define FACE_ANTISPOOF_MODEL_AT_HYPERRAM_ADDR (0x82400000)
static const char *labelsVec[] LABELS_ATTRIBUTE =
{
    "Live Face",
    "Fake Attack"
};
constexpr size_t labelsSz = 2;
const float threshold_FAS = 0.7;
#endif

typedef enum
{
    eFRAMEBUF_EMPTY,
    eFRAMEBUF_FULL,
    eFRAMEBUF_INF
} E_FRAMEBUF_STATE;

typedef struct
{
    E_FRAMEBUF_STATE eState;
    image_t frameImage;
    std::vector<arm::app::face_detection::DetectionResult> results_FD;
    arm::app::RecognitionResult result_FR;
    std::vector<arm::app::ClassificationResult> results_AS; // Basing on antispoof models, normal is 2.
} S_FRAMEBUF;


S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

namespace arm
{
namespace app
{
/* Tensor arena buffer */
#undef ACTIVATION_BUF_SZ
#define FACE_RECOGNITION_ACTIVATION_BUF_SZ   (460000)
#define FACE_DETECTION_ACTIVATION_BUF_SZ (460000)
#if defined (__ANTI_MODEL_MN3S__)
    #define FACE_ANTISPOOF_ACTIVATION_BUF_SZ (462848)
#endif
#if defined (__ANTI_MODEL_CNNH__)
    #define FACE_ANTISPOOF_ACTIVATION_BUF_SZ (800000)
#endif

static uint8_t tensorArena_FaceRecognition[FACE_RECOGNITION_ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
static uint8_t tensorArena_FaceDetection[FACE_DETECTION_ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
static uint8_t tensorArena_AntiSpoof[FACE_ANTISPOOF_ACTIVATION_BUF_SZ] ACTIVATION_BUF_HYPERRAM_ATTRIBUTE;

/* Optional getter function for the model pointer and its size. */
namespace face_detection
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace face_detection */

} /* namespace app */
} /* namespace arm */



// verify the fingerprint with AS608
static volatile bool finger_vail = true;

#if defined (__USE_FINGERP__)
/**
 * @brief     input fingerprint callback
 * @param[in] status is the current status
 * @param[in] fmt is the format data
 * @note      none
 */
static void a_callback(int8_t status, const char *const fmt, ...)
{
    switch (status)
    {
        case -1 :
        {
            /* output */
            as608_interface_debug_print("%s", fmt);
        }

        case 0 :
        {
            /* delay 2000ms */
            as608_interface_delay_ms(2000);

            /* output */
            as608_interface_debug_print("%s", fmt);
        }

        case 1 :
        {
            /* delay 2000ms */
            as608_interface_delay_ms(2000);

            /* output */
            as608_interface_debug_print("%s", fmt);
        }

        case 2 :
        {
            /* output */
            as608_interface_debug_print("%s", fmt);
        }

        default :
        {
            break;
        }
    }
}

static void verify_user_fingerprint()
{
    finger_vail = false;

    while (1)
    {
        /* as608 parameters*/
        uint8_t res_as608;
        uint16_t score_as608;
        as608_status_t status_as608;
        uint32_t addr = 0xFFFFFFFF;
        uint16_t found_page_as608;

        /* basic init */
        res_as608 = as608_basic_init(addr);

        if (res_as608 != 0)
        {
            as608_interface_debug_print("Error: basic_init");
            continue;
        }

        /* verify */
        res_as608 = as608_basic_verify(a_callback, &found_page_as608, &score_as608, &status_as608);

        if (res_as608 != 0)
        {
            (void)as608_basic_deinit();
            as608_interface_debug_print("Error: basic_verify");
            continue;
        }

        /* output */
        if (status_as608 == AS608_STATUS_OK)
        {
            as608_interface_debug_print("verify success with score %d in found page %d.\n", score_as608, found_page_as608);
            finger_vail = true;
            break;
        }
        else
        {
            as608_interface_debug_print("Verify fail! No fingerprint recognized!");
            as608_basic_print_status(status_as608);
        }

        /* basic deinit */
        (void)as608_basic_deinit();
    }
}
#endif

//frame buffer managemnet function
static S_FRAMEBUF *get_empty_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_EMPTY)
            return &s_asFramebuf[i];
    }

    return NULL;
}

static S_FRAMEBUF *get_full_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_FULL)
            return &s_asFramebuf[i];
    }

    return NULL;
}

static S_FRAMEBUF *get_inf_framebuf()
{
    int i;

    for (i = 0; i < NUM_FRAMEBUF; i ++)
    {
        if (s_asFramebuf[i].eState == eFRAMEBUF_INF)
            return &s_asFramebuf[i];
    }

    return NULL;
}

/* Image processing initiate function */
//Used by omv library
#if defined (__USE_CCAP__)
    #if defined(__USE_UVC__)
        //UVC only support QVGA, QQVGA
        #define GLCD_WIDTH  320
        #define GLCD_HEIGHT 240
    #else
        #define GLCD_WIDTH  224
        #define GLCD_HEIGHT 224
    #endif
#else
    #define GLCD_WIDTH  IMAGE_WIDTH
    #define GLCD_HEIGHT IMAGE_HEIGHT
#endif

//RGB565
#define IMAGE_FB_SIZE   (GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE (IMAGE_FB_SIZE + 1024)

#undef OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE   (1*1024)

__attribute__((section(".bss.sram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.sram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

#if (NUM_FRAMEBUF == 2)
    __attribute__((section(".bss.sram.data"), aligned(32))) static char frame_buf1[OMV_FB_SIZE];
#endif

char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

static void omv_init()
{
    image_t frameBuffer;
    int i;

    frameBuffer.w = GLCD_WIDTH;
    frameBuffer.h = GLCD_HEIGHT;
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

    _fb_base = fb_array;
    _fb_end =  fb_array + OMV_FB_SIZE - 1;
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = jpeg_array;

    fb_alloc_init0();

    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);

    for (i = 0 ; i < NUM_FRAMEBUF; i++)
    {
        s_asFramebuf[i].eState = eFRAMEBUF_EMPTY;
    }

    framebuffer_init_image(&s_asFramebuf[0].frameImage);

#if (NUM_FRAMEBUF == 2)
    s_asFramebuf[1].frameImage.w = GLCD_WIDTH;
    s_asFramebuf[1].frameImage.h = GLCD_HEIGHT;
    s_asFramebuf[1].frameImage.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    s_asFramebuf[1].frameImage.pixfmt = PIXFORMAT_RGB565;
    s_asFramebuf[1].frameImage.data = (uint8_t *)frame_buf1;
#endif
}

static void DrawDetectFace(
    std::vector<arm::app::face_detection::DetectionResult> &results,
    image_t *drawImg
)
{
    arm::app::face_detection::DetectionResult faceBox;
    int faceBoxSize = results.size();

    for (int i = 0; i < faceBoxSize; i ++)
    {
        faceBox = results[i];
        //info("Face detect on image (x,y,w,h) => (%d, %d, %d, %d)\n", faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h);
        imlib_draw_rectangle(drawImg, faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h, COLOR_B5_MAX, 2, false);
    }
}

#if defined (__USE_RESULT_ACC_WINDOWS__)
class FixedSizeBuffer
{
private:
    std::deque<int> buffer;
    const size_t maxSize;
    int sum;

public:
    FixedSizeBuffer(size_t size) : maxSize(size), sum(0) {}

    // Push a new element and maintain the buffer size and sum
    void push(int value)
    {
        // If buffer is full, remove the oldest element
        if (buffer.size() == maxSize)
        {
            sum -= buffer.front();
            buffer.pop_front();
        }

        // Add the new element
        buffer.push_back(value);
        sum += value;
    }

    int getSum() const
    {
        return sum;
    }

    void printBuffer() const
    {
        info("Buffer content: ");

        for (const auto &val : buffer)
        {
            info("%d ", val);
        }

        info("\nCurrent Sum: %d\n", sum);
    }

};
#endif

static void FaceAntiSpoof(
    S_FRAMEBUF *infFramebuf,
    arm::app::FaceAntiSpoofModel *faceAntiSpoofModel,
    arm::app::ImgClassPostProcess *postProcess,
    arm::app::Profiler *profiler
)
{
    arm::app::face_detection::DetectionResult faceBox;
    rectangle_t roi;
    TfLiteIntArray *inputShape = faceAntiSpoofModel->GetInputShape(0);
    TfLiteTensor *inputTensor   = faceAntiSpoofModel->GetInputTensor(0);

    const int inputImgCols = inputShape->data[arm::app::FaceAntiSpoofModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::FaceAntiSpoofModel::ms_inputRowsIdx];

    uint64_t u64StartCycle;
    uint64_t u64EndCycle;

    //Quantize input tensor data
    auto *req_data = static_cast<uint8_t *>(inputTensor->data.data);
    auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);

    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);

    faceBox = infFramebuf->results_FD[0];

    //resize face region image to input tensor
    image_t resizeImg;

    roi.x = faceBox.m_x0;
    roi.y = faceBox.m_y0;
    roi.w = faceBox.m_w;
    roi.h = faceBox.m_h;

    resizeImg.w = inputImgCols;
    resizeImg.h = inputImgRows;
    resizeImg.data = (uint8_t *)inputTensor->data.data; //direct resize to input tensor buffer
    resizeImg.pixfmt = PIXFORMAT_RGB888;

    if (profiler)
        u64StartCycle = pmu_get_systick_Count();

    imlib_nvt_scale(&infFramebuf->frameImage, &resizeImg, &roi);

    if (profiler)
    {
        u64EndCycle = pmu_get_systick_Count();
        info("face landmark resize cycles %llu \n", (u64EndCycle - u64StartCycle));
    }

    if (profiler)
    {
        u64StartCycle = pmu_get_systick_Count();
    }

#if defined (__ANTI_MODEL_CNNH__)

    /* face landmark/detection model preprocessing is image conversion from uint8 to [0,1] float values,
    * then quantize them with input quantization info. */
    for (size_t i = 0; i < inputTensor->bytes; i++)
    {
        //auto i_data_int8 = static_cast<int8_t>(((static_cast<float>(req_data[i]) / 255.0f) / inQuantParams.scale) + inQuantParams.offset);
        //signed_req_data[i] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
        signed_req_data[i] = static_cast<int8_t>(req_data[i]) - 128;
    }

#endif
#if defined (__ANTI_MODEL_MN3S__)

    for (size_t i = 0; i < inputTensor->bytes; i += 3)
    {
        const float mean[3] = {0.5931, 0.4690, 0.4229};
        const float std[3] = {0.2471, 0.2214, 0.2157};

        auto i_data_int8_r = ((static_cast<float>(req_data[i]) / 255.0f) - mean[0]) / std[0];
        auto i_data_int8_g = ((static_cast<float>(req_data[i + 1]) / 255.0f) - mean[1]) / std[1];
        auto i_data_int8_b = ((static_cast<float>(req_data[i + 2]) / 255.0f) - mean[2]) / std[2];

        auto i_data_int8 = static_cast<int8_t>((i_data_int8_r / inQuantParams.scale) + inQuantParams.offset);
        signed_req_data[i] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
        i_data_int8 = static_cast<int8_t>((i_data_int8_g / inQuantParams.scale) + inQuantParams.offset);
        signed_req_data[i + 1] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
        i_data_int8 = static_cast<int8_t>((i_data_int8_b / inQuantParams.scale) + inQuantParams.offset);
        signed_req_data[i + 2] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
    }

#endif


    if (profiler)
    {
        u64EndCycle = pmu_get_systick_Count();
        info("face landmark quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
    }

    if (profiler)
    {
        profiler->StartProfiling("Inference");
    }

    faceAntiSpoofModel->RunInference();

    if (profiler)
    {
        profiler->StopProfiling();
        profiler->PrintProfilingResult();
    }

    if (profiler)
    {
        u64StartCycle = pmu_get_systick_Count();
    }

#if defined (__ANTI_MODEL_MN3S__)
    postProcess->DoPostProcess_ASmodel(false);
#endif
#if defined (__ANTI_MODEL_CNNH__)
    postProcess->DoPostProcess_ASmodel(true);
#endif

    if (profiler)
    {
        u64EndCycle = pmu_get_systick_Count();
        info("face AntiSpoof post processing cycles %llu \n", (u64EndCycle - u64StartCycle));
    }

    infFramebuf->results_AS = postProcess->getTopkResults();
}

static void FaceRecognize(
    S_FRAMEBUF *infFramebuf,
    arm::app::FaceMobileNetModel *faceRecognitionModel,
    arm::app::FaceRecognPostProcess *postProcess,
    arm::app::Profiler *profiler
)
{
    arm::app::face_detection::DetectionResult faceBox;
    rectangle_t roi;
    TfLiteIntArray *inputShape = faceRecognitionModel->GetInputShape(0);
    TfLiteTensor *inputTensor   = faceRecognitionModel->GetInputTensor(0);

    const int inputImgCols = inputShape->data[arm::app::FaceMobileNetModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::FaceMobileNetModel::ms_inputRowsIdx];

    uint64_t u64StartCycle;
    uint64_t u64EndCycle;

    //Quantize input tensor data
    auto *req_data = static_cast<uint8_t *>(inputTensor->data.data);
    auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);

    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);

    faceBox = infFramebuf->results_FD[0];

    //resize face region image to input tensor
    image_t resizeImg;

    roi.x = faceBox.m_x0;
    roi.y = faceBox.m_y0;
    roi.w = faceBox.m_w;
    roi.h = faceBox.m_h;

    resizeImg.w = inputImgCols;
    resizeImg.h = inputImgRows;
    resizeImg.data = (uint8_t *)inputTensor->data.data; //direct resize to input tensor buffer
    resizeImg.pixfmt = PIXFORMAT_RGB888;

    if (profiler)
        u64StartCycle = pmu_get_systick_Count();

    imlib_nvt_scale(&infFramebuf->frameImage, &resizeImg, &roi);

    if (profiler)
    {
        u64EndCycle = pmu_get_systick_Count();
        info("face landmark resize cycles %llu \n", (u64EndCycle - u64StartCycle));
    }

    if (profiler)
    {
        u64StartCycle = pmu_get_systick_Count();
    }

    /* face landmark/detection model preprocessing is image conversion from uint8 to [0,1] float values,
    * then quantize them with input quantization info. */
    for (size_t i = 0; i < inputTensor->bytes; i++)
    {
        //          auto i_data_int8 = static_cast<int8_t>(((static_cast<float>(req_data[i]) / 255.0f) / inQuantParams.scale) + inQuantParams.offset);
        //          signed_req_data[i] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
        signed_req_data[i] = static_cast<int8_t>(req_data[i]) - 128;
    }

    if (profiler)
    {
        u64EndCycle = pmu_get_systick_Count();
        info("face landmark quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
    }

    if (profiler)
    {
        profiler->StartProfiling("Inference");
    }

    faceRecognitionModel->RunInference();

    if (profiler)
    {
        profiler->StopProfiling();
        profiler->PrintProfilingResult();
    }

    if (profiler)
    {
        u64StartCycle = pmu_get_systick_Count();
    }

    postProcess->RunPostProcess(infFramebuf->result_FR);

    if (profiler)
    {
        u64EndCycle = pmu_get_systick_Count();
        info("face landmark post processing cycles %llu \n", (u64EndCycle - u64StartCycle));
    }

    if (profiler)
    {
        u64StartCycle = pmu_get_systick_Count();
    }

    DrawDetectFace(infFramebuf->results_FD, &infFramebuf->frameImage);

    if (profiler)
    {
        u64EndCycle = pmu_get_systick_Count();
        info("draw face region cycles %llu \n", (u64EndCycle - u64StartCycle));
    }
}

static void DetectFaceRegion(
    S_FRAMEBUF *infFramebuf,
    arm::app::FaceDetectionModel *faceDetectionModel,
    arm::app::FaceDetectorPostProcess *postProcess,
    arm::app::Profiler *profiler
)
{
    TfLiteIntArray *inputShape = faceDetectionModel->GetInputShape(0);
    TfLiteTensor *inputTensor   = faceDetectionModel->GetInputTensor(0);
    rectangle_t roi;

    const int inputImgCols = inputShape->data[arm::app::FaceDetectionModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::FaceDetectionModel::ms_inputRowsIdx];

    uint64_t u64StartCycle;
    uint64_t u64EndCycle;

    auto *req_data = static_cast<uint8_t *>(inputTensor->data.data);
    auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);

    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);

    //resize full image to input tensor
    image_t resizeImg;

    roi.x = 0;
    roi.y = 0;
    roi.w = infFramebuf->frameImage.w;
    roi.h = infFramebuf->frameImage.h;

    resizeImg.w = inputImgCols;
    resizeImg.h = inputImgRows;
    resizeImg.data = (uint8_t *)inputTensor->data.data; //direct resize to input tensor buffer
    resizeImg.pixfmt = PIXFORMAT_GRAYSCALE;

    if (profiler)
        u64StartCycle = pmu_get_systick_Count();

    imlib_nvt_scale(&infFramebuf->frameImage, &resizeImg, &roi);

    if (profiler)
    {
        u64EndCycle = pmu_get_systick_Count();
        info("face detect resize cycles %llu \n", (u64EndCycle - u64StartCycle));
    }

    if (profiler)
    {
        u64StartCycle = pmu_get_systick_Count();
    }

    /* face landmark/detection model preprocessing is image conversion from uint8 to [0,1] float values,
     * then quantize them with input quantization info. */
    for (size_t i = 0; i < inputTensor->bytes; i++)
    {
        //      auto i_data_int8 = static_cast<int8_t>(((static_cast<float>(req_data[i]) / 255.0f) / inQuantParams.scale) + inQuantParams.offset);
        //      signed_req_data[i] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
        signed_req_data[i] = static_cast<int8_t>(req_data[i]) - 128;
    }

    if (profiler)
    {
        u64EndCycle = pmu_get_systick_Count();
        info("face detect quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
    }

    if (profiler)
    {
        profiler->StartProfiling("Inference");
    }

    faceDetectionModel->RunInference();

    if (profiler)
    {
        profiler->StopProfiling();
        profiler->PrintProfilingResult();
    }

    if (profiler)
    {
        u64StartCycle = pmu_get_systick_Count();
    }

    postProcess->RunPostProcess(infFramebuf->results_FD);

    if (profiler)
    {
        u64EndCycle = pmu_get_systick_Count();
        info("face detect post processing cycles %llu \n", (u64EndCycle - u64StartCycle));
    }

    float scaleFactoryW = 1.1;
    float scaleFactoryH = 1.05;
    arm::app::face_detection::DetectionResult *faceBox;

    //fine tune face region
    for (int i = 0 ; i < infFramebuf->results_FD.size(); i ++)
    {
        float scaleW;
        float scaleH;
        int newX;
        int newY;

        int newW;
        int newH;

        faceBox = &(infFramebuf->results_FD[i]);

        scaleH = scaleFactoryH * faceBox->m_h;
        scaleW = scaleFactoryW * faceBox->m_w;
        //scaleW = scaleH;
        newW = scaleW;
        newH = scaleH;

        newX = faceBox->m_x0 - ((scaleW - faceBox->m_w) / 2);
        newY = faceBox->m_y0 - ((scaleH - faceBox->m_h) / 2);

        if (newX < 0)
            newX = 0;

        if (newY < 0)
            newY = 0;

        if (newX + newW >= infFramebuf->frameImage.w)
            newW = infFramebuf->frameImage.w - newX;

        if (newY + newH >= infFramebuf->frameImage.h)
            newH = infFramebuf->frameImage.h - newY;

        faceBox->m_x0 = newX;
        faceBox->m_y0 = newY;
        faceBox->m_w = newW;
        faceBox->m_h = newH;
    }
}

static int32_t PrepareModelToHyperRAM(const char *model_file, uint32_t model_addr)
{
#define EACH_READ_SIZE 512

    TCHAR sd_path[] = { '0', ':', 0 };    /* SD drive started from 0 */
    f_chdrive(sd_path);          /* set default path */

    int32_t i32FileSize;
    int32_t i32FileReadIndex = 0;
    int32_t i32Read;

    if (!ModelFileReader_Initialize(model_file))
    {
        printf_err("Unable open model %c\n", *model_file);
        return -1;
    }

    i32FileSize = ModelFileReader_FileSize();
    info("Model file size %i \n", i32FileSize);

    while (i32FileReadIndex < i32FileSize)
    {
        i32Read = ModelFileReader_ReadData((BYTE *)(model_addr + i32FileReadIndex), EACH_READ_SIZE);

        if (i32Read < 0)
            break;

        i32FileReadIndex += i32Read;
    }

    if (i32FileReadIndex < i32FileSize)
    {
        printf_err("Read Model file size is not enough\n");
        return -2;
    }

#if 0
    /* verify */
    i32FileReadIndex = 0;
    ModelFileReader_Rewind();
    BYTE au8TempBuf[EACH_READ_SIZE];

    while (i32FileReadIndex < i32FileSize)
    {
        i32Read = ModelFileReader_ReadData((BYTE *)au8TempBuf, EACH_READ_SIZE);

        if (i32Read < 0)
            break;

        if (std::memcmp(au8TempBuf, (void *)(FACE_MOBILENET_MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), i32Read) != 0)
        {
            printf_err("verify the model file content is incorrect at %i \n", i32FileReadIndex);
            return -3;
        }

        i32FileReadIndex += i32Read;
    }

#endif
    ModelFileReader_Finish();

    return i32FileSize;
}

int main()
{
    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

    /* Model object creation and initialisation. */
    /* Copy model file from SD to HyperRAM*/
    int32_t i32ModelSize;

    i32ModelSize = PrepareModelToHyperRAM(FACE_MOBILENET_MODEL_FILE, FACE_MOBILENET_MODEL_AT_HYPERRAM_ADDR);

    if (i32ModelSize <= 0)
    {
        printf_err("Failed to prepare model\n");
        return 1;
    }

    arm::app::FaceMobileNetModel faceRecognitionModel;

    if (!faceRecognitionModel.Init(arm::app::tensorArena_FaceRecognition,
                                   sizeof(arm::app::tensorArena_FaceRecognition),
                                   (unsigned char *)FACE_MOBILENET_MODEL_AT_HYPERRAM_ADDR,
                                   i32ModelSize))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }

    /* Model object creation and initialisation. */
    arm::app::FaceDetectionModel faceDetectionModel;

    if (!faceDetectionModel.Init(arm::app::tensorArena_FaceDetection,
                                 sizeof(arm::app::tensorArena_FaceDetection),
                                 arm::app::face_detection::GetModelPointer(),
                                 arm::app::face_detection::GetModelLen()))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }

    /* Model object creation and initialisation. */
    /* Copy model file from SD to HyperRAM*/
    i32ModelSize = PrepareModelToHyperRAM(FACE_ANTISPOOF_MODEL_FILE, FACE_ANTISPOOF_MODEL_AT_HYPERRAM_ADDR);

    if (i32ModelSize <= 0)
    {
        printf_err("Failed to prepare model\n");
        return 1;
    }

    arm::app::FaceAntiSpoofModel faceAntiSpoofModel;

    if (!faceAntiSpoofModel.Init(arm::app::tensorArena_AntiSpoof,
                                 sizeof(arm::app::tensorArena_AntiSpoof),
                                 (unsigned char *)FACE_ANTISPOOF_MODEL_AT_HYPERRAM_ADDR,
                                 i32ModelSize))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }

    /* Setup cache poicy of tensor arean buffer */
    info("Set tesnor arena cache policy to WTRA \n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena_FaceRecognition),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena_FaceRecognition) + FACE_RECOGNITION_ACTIVATION_BUF_SZ - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - Write-Through, Read-allocate
        },
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena_FaceDetection),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena_FaceDetection) + FACE_DETECTION_ACTIVATION_BUF_SZ - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - Write-Through, Read-allocate
        },
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena_AntiSpoof),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena_AntiSpoof) + FACE_ANTISPOOF_ACTIVATION_BUF_SZ - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - Write-Through, Read-allocate
        },
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)fb_array),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)fb_array) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
#if (NUM_FRAMEBUF == 2)
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)frame_buf1),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)frame_buf1) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
#endif
    };

    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

    TfLiteIntArray *inputShape_FD = faceDetectionModel.GetInputShape(0);

    const int inputImgCols_FD = inputShape_FD->data[arm::app::FaceDetectionModel::ms_inputColsIdx];
    const int inputImgRows_FD = inputShape_FD->data[arm::app::FaceDetectionModel::ms_inputRowsIdx];

    TfLiteTensor *outputTensor0_FD = faceDetectionModel.GetOutputTensor(0);
    TfLiteTensor *outputTensor1_FD = faceDetectionModel.GetOutputTensor(1);

    //display framebuffer
    image_t frameBuffer;
    rectangle_t roi;

    //omv library init
    omv_init();
    framebuffer_init_image(&frameBuffer);

    //label information
    std::vector<std::string> labels;
    std::vector<S_LABEL_INFO> labelInfo;

#if 0
    GetLabelsVector(labels);
    ParserLabelVector(labels, labelInfo, nullptr);
#else
    ParserLabelVectorFromFile(FACE_LABEL_FILE, labelInfo, nullptr);
#endif

    // Set up post-Process
    // face_detection
    const arm::app::face_detection::PostProcessParams postProcessParams
    {
        inputImgRows_FD,
        inputImgCols_FD,
        (int)s_asFramebuf[0].frameImage.h,
        (int)s_asFramebuf[0].frameImage.w,
        anchor1,
        anchor2};

    arm::app::FaceDetectorPostProcess postProcess_FD =
        arm::app::FaceDetectorPostProcess(outputTensor0_FD, outputTensor1_FD, s_asFramebuf[0].results_FD, postProcessParams);

    // face_Recognizer
    arm::app::Recognizer recognizer;  /* Classifier object. */
    arm::app::FaceRecognPostProcess postProcess_FR(recognizer, &faceRecognitionModel, labelInfo);

    // face_AntiSpoof
    //label information
    std::vector <std::string> labels_AS;
    GetLabelsVector(labels_AS, labelsVec, labelsSz);


    arm::app::Classifier classifier;  /* Classifier object. */
    arm::app::ImgClassPostProcess postProcess_AS = arm::app::ImgClassPostProcess(classifier, &faceAntiSpoofModel, labels_AS);


#if defined(__PROFILE__)

    arm::app::Profiler profiler;
    uint64_t u64StartCycle;
    uint64_t u64EndCycle;
    uint64_t u64CCAPStartCycle;
    uint64_t u64CCAPEndCycle;
#else
    pmu_reset_counters();
#endif

#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = pmu_get_systick_Count();
    u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);

    S_FRAMEBUF *infFramebuf;
    S_FRAMEBUF *fullFramebuf;
    S_FRAMEBUF *emptyFramebuf;

#if !defined (__USE_CCAP__)
    uint8_t u8ImgIdx = 0;
    char chStdIn;
#endif

#if defined (__USE_CCAP__)
    //Setup image senosr
    ImageSensor_Init();
    ImageSensor_Config(eIMAGE_FMT_RGB565, frameBuffer.w, frameBuffer.h, true);
#endif

#if defined (__USE_DISPLAY__)
    char szDisplayText[100];
    S_DISP_RECT sDispRect;

    Display_Init();
    Display_ClearLCD(C_WHITE);
#endif

#if defined (__USE_UVC__)
    UVC_Init();
    HSUSBD_Start();
#endif

#if defined (__USE_FINGERP__)
    verify_user_fingerprint();
#endif

    std::string predictLabelInfo;

#if defined (__USE_RESULT_ACC_WINDOWS__)
    FixedSizeBuffer sliwin_res_buf(10); // Buffer of size 10
#endif

    while (1 && finger_vail)
    {
        emptyFramebuf = get_empty_framebuf();

        if (emptyFramebuf)
        {
#if !defined (__USE_CCAP__)
            info("Press 'n' to run next image inference \n");
            info("Press 'q' to exit program \n");

            while ((chStdIn = getchar()))
            {
                if (chStdIn == 'q')
                {
                    goto prog_done;
                }
                else if (chStdIn != 'n')
                {
                    break;
                }
            }

            const uint8_t *pu8ImgSrc = get_img_array(u8ImgIdx);

            if (nullptr == pu8ImgSrc)
            {
                printf_err("Failed to get image index %" PRIu32 " (max: %u)\n", u8ImgIdx,
                           NUMBER_OF_FILES - 1);
                goto prog_done;
            }

            u8ImgIdx ++;

            if (u8ImgIdx >= NUMBER_OF_FILES)
                u8ImgIdx = 0;


            //copy source image to frame buffer
            image_t srcImg;

            srcImg.w = IMAGE_WIDTH;
            srcImg.h = IMAGE_HEIGHT;
            srcImg.data = (uint8_t *)pu8ImgSrc;
            srcImg.pixfmt = PIXFORMAT_RGB888;

            roi.x = 0;
            roi.y = 0;
            roi.w = IMAGE_WIDTH;
            roi.h = IMAGE_HEIGHT;

            imlib_nvt_scale(&srcImg, &emptyFramebuf->frameImage, &roi);

#endif

            //capture frame from CCAP
#if defined(__PROFILE__)
            u64CCAPStartCycle = pmu_get_systick_Count();
#endif

#if defined (__USE_CCAP__)
            ImageSensor_TriggerCapture((uint32_t)(emptyFramebuf->frameImage.data));
#endif
        }

        fullFramebuf = get_full_framebuf();

        if (fullFramebuf)
        {
#if defined(__PROFILE__)
            DetectFaceRegion(
                fullFramebuf,
                &faceDetectionModel,
                &postProcess_FD,
                &profiler);
#else
            DetectFaceRegion(
                fullFramebuf,
                &faceDetectionModel,
                &postProcess_FD,
                nullptr);
#endif
            fullFramebuf->eState = eFRAMEBUF_INF;
        }

        infFramebuf = get_inf_framebuf();

        if (infFramebuf)
        {

#if defined(__PROFILE__)

            if (infFramebuf->results_FD.size())
                FaceRecognize(
                    infFramebuf,
                    &faceRecognitionModel,
                    &postProcess_FR,
                    &profiler);

#else

            if (infFramebuf->results_FD.size())
                FaceRecognize(
                    infFramebuf,
                    &faceRecognitionModel,
                    &postProcess_FR,
                    nullptr);

#endif


            //display result image
#if defined (__USE_DISPLAY__)
            //Display image on LCD
            sDispRect.u32TopLeftX = 0;
            sDispRect.u32TopLeftY = 0;
            sDispRect.u32BottonRightX = (infFramebuf->frameImage.w - 1);
            sDispRect.u32BottonRightY = (infFramebuf->frameImage.h - 1);

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif

            Display_FillRect((uint16_t *)infFramebuf->frameImage.data, &sDispRect);

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("display image cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#endif

#if defined (__USE_UVC__)

            if (UVC_IsConnect())
            {
#if (UVC_Color_Format == UVC_Format_YUY2)
                image_t RGB565Img;
                image_t YUV422Img;

                RGB565Img.w = infFramebuf->frameImage.w;
                RGB565Img.h = infFramebuf->frameImage.h;
                RGB565Img.data = (uint8_t *)infFramebuf->frameImage.data;
                RGB565Img.pixfmt = PIXFORMAT_RGB565;

                YUV422Img.w = RGB565Img.w;
                YUV422Img.h = RGB565Img.h;
                YUV422Img.data = (uint8_t *)infFramebuf->frameImage.data;
                YUV422Img.pixfmt = PIXFORMAT_YUV422;

                roi.x = 0;
                roi.y = 0;
                roi.w = RGB565Img.w;
                roi.h = RGB565Img.h;
                imlib_nvt_scale(&RGB565Img, &YUV422Img, &roi);

#else
                image_t origImg;
                image_t vflipImg;

                origImg.w = infFramebuf->frameImage.w;
                origImg.h = infFramebuf->frameImage.h;
                origImg.data = (uint8_t *)infFramebuf->frameImage.data;
                origImg.pixfmt = PIXFORMAT_RGB565;

                vflipImg.w = origImg.w;
                vflipImg.h = origImg.h;
                vflipImg.data = (uint8_t *)infFramebuf->frameImage.data;
                vflipImg.pixfmt = PIXFORMAT_RGB565;

                imlib_nvt_vflip(&origImg, &vflipImg);
#endif
                UVC_SendImage((uint32_t)infFramebuf->frameImage.data, IMAGE_FB_SIZE, uvcStatus.StillImage);

            }

#endif

            if (infFramebuf->results_FD.size())
            {

                if (infFramebuf->result_FR.m_recognize) // if recognize the face, go to antispoof test
                {
                    FaceAntiSpoof(infFramebuf, &faceAntiSpoofModel, &postProcess_AS, nullptr);

                    // Debug
                    // info("results_AS[0] idx: %d, val: %f \n", infFramebuf->results_AS[0].m_labelIdx, infFramebuf->results_AS[0].m_normalisedVal);
                    // info("results_AS[1] idx: %d, val: %f \n", infFramebuf->results_AS[1].m_labelIdx, infFramebuf->results_AS[1].m_normalisedVal);

                    // if the Softmax value of real (IDX=0) class is lower than threshold, it is fake
                    if (infFramebuf->results_AS[0].m_labelIdx == 0 && infFramebuf->results_AS[0].m_normalisedVal > threshold_FAS)
                    {
#if defined (__USE_RESULT_ACC_WINDOWS__)
                        sliwin_res_buf.push(1); // 1 => live, 0 => fake
#else
                        predictLabelInfo =  infFramebuf->result_FR.m_label + std::string(":") + std::to_string(infFramebuf->result_FR.m_predict) +
                                            std::string(": Real ") + std::to_string(infFramebuf->results_AS[0].m_normalisedVal);
#endif
                    }
                    else
                    {
#if defined (__USE_RESULT_ACC_WINDOWS__)
                        sliwin_res_buf.push(0); // 1 => live, 0 => fake
#else
                        predictLabelInfo =  infFramebuf->result_FR.m_label + std::string(":") + std::to_string(infFramebuf->result_FR.m_predict) +
                                            std::string(": Fake! ") + std::to_string(infFramebuf->results_AS[0].m_normalisedVal);
#endif
                    }

#if defined (__USE_RESULT_ACC_WINDOWS__)
                    info("Final results (10 counts):\n");
                    float percent_count;
                    percent_count = static_cast<float>(sliwin_res_buf.getSum()) / 10.0;

                    if (percent_count > 0.6)
                    {
                        predictLabelInfo =  infFramebuf->result_FR.m_label + std::string(":") + std::to_string(infFramebuf->result_FR.m_predict) +
                                            std::string(": Real ") + std::to_string(percent_count);
                    }
                    else
                    {
                        predictLabelInfo =  infFramebuf->result_FR.m_label + std::string(":") + std::to_string(infFramebuf->result_FR.m_predict) +
                                            std::string(": Fake! ") + std::to_string(percent_count);
                    }

#endif
                }
                else
                {
                    predictLabelInfo = std::string("???") + std::string(":") + std::to_string(infFramebuf->result_FR.m_predict);
                }

                //show result
#if defined (__USE_RESULT_ACC_WINDOWS__)
                info("Final results (10 counts):\n");
#else
                info("Final results:\n");
#endif
                info("%s\n", predictLabelInfo.c_str());

            } // result_FD.size()

#if defined (__USE_DISPLAY__)

            sDispRect.u32TopLeftX = 0;
            sDispRect.u32TopLeftY = frameBuffer.h;
            sDispRect.u32BottonRightX = (frameBuffer.w);
            sDispRect.u32BottonRightY = (frameBuffer.h + FONT_HTIGHT - 1);

            Display_ClearRect(C_WHITE, &sDispRect);

            if (infFramebuf->results_FD.size())
            {
                sprintf(szDisplayText, "%s", predictLabelInfo.c_str());

                Display_PutText(
                    szDisplayText,
                    strlen(szDisplayText),
                    0,
                    frameBuffer.h,
                    C_BLUE,
                    C_WHITE,
                    true
                );
            }

#endif


            u64PerfFrames ++;

            if ((uint64_t) pmu_get_systick_Count() > u64PerfCycle)
            {
                info("Total inference rate: %llu\n", u64PerfFrames / EACH_PERF_SEC);
#if defined (__USE_DISPLAY__)
                sprintf(szDisplayText, "Frame Rate %llu", u64PerfFrames / EACH_PERF_SEC);
                //sprintf(szDisplayText,"Time %llu",(uint64_t) pmu_get_systick_Count() / (uint64_t)SystemCoreClock);
                //info("Running %s sec \n", szDisplayText);

                sDispRect.u32TopLeftX = 0;
                sDispRect.u32TopLeftY = frameBuffer.h + FONT_HTIGHT;
                sDispRect.u32BottonRightX = (frameBuffer.w);
                sDispRect.u32BottonRightY = (frameBuffer.h + (2 * FONT_HTIGHT) - 1);

                Display_ClearRect(C_WHITE, &sDispRect);
                Display_PutText(
                    szDisplayText,
                    strlen(szDisplayText),
                    0,
                    frameBuffer.h + FONT_HTIGHT,
                    C_BLUE,
                    C_WHITE,
                    false
                );
#endif
                u64PerfCycle = (uint64_t)pmu_get_systick_Count() + (uint64_t)(SystemCoreClock * EACH_PERF_SEC);
                u64PerfFrames = 0;
            }

            infFramebuf->eState = eFRAMEBUF_EMPTY;
        }

        //Wait CCAP ready
        if (emptyFramebuf)
        {
#if defined (__USE_CCAP__)
            //Capture new image
            ImageSensor_WaitCaptureDone();
#endif
#if defined(__PROFILE__)
            u64CCAPEndCycle = pmu_get_systick_Count();
            info("ccap capture cycles %llu \n", (u64CCAPEndCycle - u64CCAPStartCycle));
#endif
            emptyFramebuf->eState = eFRAMEBUF_FULL;
        }
    }

prog_done:

    return 0;
}