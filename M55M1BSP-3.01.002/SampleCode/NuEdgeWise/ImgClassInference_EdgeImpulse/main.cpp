/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    Image Classification model inference sample with Edge Impulse SDK. Demonstrate network infereence
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2025 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "BoardInit.hpp"      /* Board initialisation */

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

#include "ModelFileReader.h"
#include "ff.h"

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"    /* EI SDK */
#include "export_tensor_arena.h"

//#define LOG_LEVEL_TRACE       0
//#define LOG_LEVEL_DEBUG       1
//#define LOG_LEVEL_INFO        2
//#define LOG_LEVEL_WARN        3
//#define LOG_LEVEL_ERROR       4

#define LOG_LEVEL             2
#include "log_macros.h"      /* Logging macros (optional) */

//#define __LOAD_MODEL_FROM_SD__

#define MODEL_AT_HYPERRAM_ADDR 0x82400000

#include "Profiler.hpp"
#include "ImageSensor.h"

//#define __PROFILE__
#define __USE_UVC__    /* Real Time Test */

#if defined (__USE_UVC__)    /* Real Time Test */
    #include "UVC.h"
#else
    #include "InputFiles.hpp"    /* Test Data */
#endif

/* Image processing initiate function */
//Used by omv library
#if defined(__USE_UVC__)
    //UVC only support QVGA, QQVGA
    #define GLCD_WIDTH  320
    #define GLCD_HEIGHT 240
#else
    #define GLCD_WIDTH 320
    #define GLCD_HEIGHT 240
#endif

//RGB565
#define IMAGE_FB_SIZE   (GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE (IMAGE_FB_SIZE + 1024)

__attribute__((section(".bss.vram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

static void omv_init()
{
    image_t frameBuffer;

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
}

static int32_t PrepareModelToHyperRAM(void)
{
#define MODEL_FILE "0:\\nn_model.tflite"
#define EACH_READ_SIZE 512

    TCHAR sd_path[] = { '0', ':', 0 };    /* SD drive started from 0 */
    f_chdrive(sd_path);          /* set default path */

    int32_t i32FileSize;
    int32_t i32FileReadIndex = 0;
    int32_t i32Read;

    if (!ModelFileReader_Initialize(MODEL_FILE))
    {
        printf_err("Unable open model %s\n", MODEL_FILE);
        return -1;
    }

    i32FileSize = ModelFileReader_FileSize();
    info("Model file size %i \n", i32FileSize);

    while (i32FileReadIndex < i32FileSize)
    {
        i32Read = ModelFileReader_ReadData((BYTE *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), EACH_READ_SIZE);

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

        if (std::memcmp(au8TempBuf, (void *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), i32Read) != 0)
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

#if !defined(__USE_UVC__)
// Callback: fill a section of the out_ptr buffer when requested
static int get_signal_data(size_t offset, size_t length, float *out_ptr)
{
    for (size_t i = 0; i < length; i++)
    {
        out_ptr[i] = (input_buf + offset)[i];
    }

    return EIDSP_OK;
}
#else
// Setup Input Buf
static uint8_t ei_buf_array[EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT * 3];

static int raw_feature_get_data(size_t offset, size_t out_len, float *signal_ptr)
{
    size_t pixel_ix = offset * 3;
    size_t bytes_left = out_len;
    size_t out_ptr_ix = 0;

    // read byte for byte
    while (bytes_left != 0)
    {
        // grab the values and convert to r/g/b
        uint8_t r, g, b;
        r = ei_buf_array[pixel_ix];
        g = ei_buf_array[pixel_ix + 1];
        b = ei_buf_array[pixel_ix + 2];

        // then convert to out_ptr format
        float pixel_f = (r << 16) + (g << 8) + b;
        signal_ptr[out_ptr_ix] = pixel_f;

        // and go to the next pixel
        out_ptr_ix++;
        pixel_ix += 3;
        bytes_left--;
    }

    return EIDSP_OK;
}
#endif

int main()
{

    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

#if defined(__LOAD_MODEL_FROM_SD__)

    /* Copy model file from SD to HyperRAM*/
    int32_t i32ModelSize;

    printf("==================== Load model file from SD card =================================\n");
    printf("Please copy NN_ModelInference/Model/xxx_vela.tflite to SDCard:/nn_model.tflite     \n");
    printf("===================================================================================\n");
    i32ModelSize = PrepareModelToHyperRAM();

    if (i32ModelSize <= 0)
    {
        printf_err("Failed to prepare model\n");
        return 1;
    }

    /* Model object creation and initialisation. */
    arm::app::NNModel model;

    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    (unsigned char *)MODEL_AT_HYPERRAM_ADDR,
                    i32ModelSize))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }

#else

    /* Model object creation and initialisation. */
    // summary of inferencing settings (from model_metadata.h)
    printf("Edge Impulse Inferencing settings:\n");
    printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

#endif



    //Edge Impulse structure initial
    signal_t signal;            // Wrapper for raw input buffer
    ei_impulse_result_t result = {0}; // Used to store inference output
    EI_IMPULSE_ERROR res;       // Return code from inference

#if defined(__PROFILE__) && defined(__USE_UVC__)
    arm::app::Profiler profiler;
    uint64_t u64StartCycle;
    uint64_t u64EndCycle;
    uint64_t u64CCAPStartCycle;
    uint64_t u64CCAPEndCycle;
#else
    pmu_reset_counters();
#endif

#if !defined(__USE_UVC__)       // inference with inputFile test data

    // Calculate the length of the buffer
    size_t buf_len = sizeof(input_buf) / sizeof(input_buf[0]);

    // Make sure that the length of the buffer matches expected input length
    if (buf_len != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE)
    {
        printf("ERROR: The size of the input buffer is not correct.\r\n");
        printf("Expected %d items, but got %d\r\n",
               EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE,
               (int)buf_len);
        return 1;
    }

    // Assign callback function to fill buffer used for preprocessing/inference
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &get_signal_data;
    //numpy::signal_from_buffer(&input_buf[0], EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);

    // Perform DSP pre-processing and inference
    printf("\r\n");
    printf("run_classifier\r\n");
    res = run_classifier(&signal, &result, false);

    // Print return code and how long it took to perform inference
    printf("run_classifier returned: %d\r\n", res);
    printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
           result.timing.dsp,
           result.timing.classification,
           result.timing.anomaly);

    // Print the prediction results (classification)
    printf("Predictions:\r\n");

    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
    {
        printf("  %s: ", ei_classifier_inferencing_categories[i]);
        printf("%.5f\r\n", result.classification[i].value);
    }

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

#else
    /* Setup cache poicy of tensor arean buffer */
    info("Set tesnor arena cache policy to WTRA \n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
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
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)tensor_arena),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)tensor_arena) + 149680 - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - Write-Through, Read-allocate
        },
    };
    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

    const int inputImgCols = EI_CLASSIFIER_INPUT_WIDTH;
    const int inputImgRows = EI_CLASSIFIER_INPUT_HEIGHT;
    const uint32_t inputChannels = EI_CLASSIFIER_NN_INPUT_FRAME_SIZE / EI_CLASSIFIER_RAW_SAMPLE_COUNT;

    //display framebuffer
    image_t frameBuffer;
    rectangle_t roi;

    //omv library init
    omv_init();
    framebuffer_init_image(&frameBuffer);

#endif



#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = pmu_get_systick_Count();
    u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);

#if defined (__USE_UVC__)
    //Setup image senosr
    ImageSensor_Init();
    ImageSensor_Config(eIMAGE_FMT_RGB565, frameBuffer.w, frameBuffer.h, true);

    UVC_Init();
    HSUSBD_Start();
#endif



    while (1)
    {

#if defined (__USE_UVC__)
        //resize framebuffer image to model input
        image_t resizeImg;

        roi.x = 0;
        roi.y = 0;
        roi.w = frameBuffer.w;
        roi.h = frameBuffer.h;

        resizeImg.w = inputImgCols;
        resizeImg.h = inputImgRows;
        resizeImg.data = (uint8_t *)ei_buf_array; //direct resize to input tensor buffer
        resizeImg.pixfmt = PIXFORMAT_RGB888;

#if defined(__PROFILE__)
        u64StartCycle = pmu_get_systick_Count();
#endif
        imlib_nvt_scale(&frameBuffer, &resizeImg, &roi);
#if defined(__PROFILE__)
        u64EndCycle = pmu_get_systick_Count();
        info("resize cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

        signal.total_length = inputImgCols * inputImgRows;
        signal.get_data = &raw_feature_get_data;


#if defined (__USE_UVC__)

        if (UVC_IsConnect())
        {
#if (UVC_Color_Format == UVC_Format_YUY2)
            image_t RGB565Img;
            image_t YUV422Img;

            RGB565Img.w = frameBuffer.w;
            RGB565Img.h = frameBuffer.h;
            RGB565Img.data = (uint8_t *)frameBuffer.data;
            RGB565Img.pixfmt = PIXFORMAT_RGB565;

            YUV422Img.w = RGB565Img.w;
            YUV422Img.h = RGB565Img.h;
            YUV422Img.data = (uint8_t *)frameBuffer.data;
            YUV422Img.pixfmt = PIXFORMAT_YUV422;

            roi.x = 0;
            roi.y = 0;
            roi.w = RGB565Img.w;
            roi.h = RGB565Img.h;
            imlib_nvt_scale(&RGB565Img, &YUV422Img, &roi);

#else
            image_t origImg;
            image_t vflipImg;

            origImg.w = frameBuffer.w;
            origImg.h = frameBuffer.h;
            origImg.data = (uint8_t *)frameBuffer.data;
            origImg.pixfmt = PIXFORMAT_RGB565;

            vflipImg.w = origImg.w;
            vflipImg.h = origImg.h;
            vflipImg.data = (uint8_t *)frameBuffer.data;
            vflipImg.pixfmt = PIXFORMAT_RGB565;

            imlib_nvt_vflip(&origImg, &vflipImg);
#endif
            UVC_SendImage((uint32_t)frameBuffer.data, IMAGE_FB_SIZE, uvcStatus.StillImage);
        }

#endif

        //Capture new image
#if defined(__PROFILE__)
        u64CCAPStartCycle = pmu_get_systick_Count();
#endif
        ImageSensor_TriggerCapture((uint32_t)frameBuffer.data);

#if defined(__PROFILE__)
        u64StartCycle = pmu_get_systick_Count();
#endif

#if defined(__PROFILE__)
        profiler.StartProfiling("Inference");
#endif

        // Perform DSP pre-processing and inference
        res = run_classifier(&signal, &result, false);

#if defined(__PROFILE__)
        profiler.StopProfiling();
#endif

        //Capture new image
        ImageSensor_WaitCaptureDone();
#if defined(__PROFILE__)
        u64CCAPEndCycle = pmu_get_systick_Count();
        info("ccap capture cycles %llu \n", (u64CCAPEndCycle - u64CCAPStartCycle));
#endif


        // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

        u64PerfFrames ++;

        if (pmu_get_systick_Count() > u64PerfCycle)
        {
            info("Model inference rate: %llu inf/s \n", u64PerfFrames / EACH_PERF_SEC);
            info("Accumulated time: %llu (s) \n", pmu_get_systick_Count() / SystemCoreClock);
            u64PerfCycle = pmu_get_systick_Count();
            u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);
            u64PerfFrames = 0;

            // Print the prediction results (classification)
            info("Predictions:\r\n");

            for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
            {
                printf("  %s: ", ei_classifier_inferencing_categories[i]);
                printf("%.5f\r\n", result.classification[i].value);
            }

#if defined(__PROFILE__)
            profiler.PrintProfilingResult();
#endif
        }


#else

        // Perform DSP pre-processing and inference
        res = run_classifier(&signal, &result, false);
        //res = run_classifier_continuous(&signal, &result, false);

        u64PerfFrames ++;

        if (pmu_get_systick_Count() > u64PerfCycle)
        {
            info("Model inference rate: %llu inf/s \n", u64PerfFrames / EACH_PERF_SEC);
            info("Accumulated time: %llu (s) \n", pmu_get_systick_Count() / SystemCoreClock);
            u64PerfCycle = pmu_get_systick_Count();
            u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);
            u64PerfFrames = 0;

            // Print the prediction results (classification)
            info("Predictions:\r\n");

            for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
            {
                printf("  %s: ", ei_classifier_inferencing_categories[i]);
                printf("%.5f\r\n", result.classification[i].value);
            }

            // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
            printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

        }

#endif
    }  //while(1)
}