/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    YOLOX-nano network inference sample. Demonstrate object detection.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

/****************************************************************************
 * Includes
 ****************************************************************************/
#include <inttypes.h>
#include <string>
#include <stdio.h>
#include <vector>
#include <zephyr/kernel.h>
#include <zephyr/sys/time_units.h>
#include <zephyr/arch/arm/mpu/arm_mpu.h>
#include <zephyr/cache.h>
#include <zephyr/logging/log.h>

#include "BoardInit.hpp"
#include "ModelFileReader.h"
#include "ff.h"
#include "YoloXnanoNu.hpp"       /* Model API */
#include "DetectorPostProcessing.hpp"
#include "Profiler.hpp"
#include "InferenceTask.hpp"
#include "Labels.hpp"

#include "ImageSensor.h"

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"

#include "Profiler.hpp"

#define IMAGE_REAL_FRAMRATE		16  
#include "BYTETracker.h"

// Define activation buffer size for model inference
#undef ACTIVATION_BUF_SZ
#define ACTIVATION_BUF_SZ (512 * 1024)  //512KB
#include "BufAttributes.hpp" /* Buffer attributes to be applied */

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

#if defined (__USE_LCD__)
    #include "Display.h"
#endif

// Model location when loaded from SD card to HyperRAM
#define MODEL_AT_HYPERRAM_ADDR 0x82400000

#define IMAGE_DISP_UPSCALE_FACTOR 1
#if defined(LT7381_LCD_PANEL)
#define FONT_DISP_UPSCALE_FACTOR 2
#else
#define FONT_DISP_UPSCALE_FACTOR 1
#endif

//Used by omv library
#if defined(__PREVIEW_SUPPORT__)
    //UVC only support QVGA, QQVGA
    #define GLCD_WIDTH  320
    #define GLCD_HEIGHT 240
    #define IMGAE_COLOR_PIXEL_FORMAT    PIXFORMAT_RGB565
    #define IMGAE_COLOR_BPP 2
#else
    #define GLCD_WIDTH  320
    #define GLCD_HEIGHT 320
    #define IMGAE_COLOR_PIXEL_FORMAT    PIXFORMAT_RGB888
    #define IMGAE_COLOR_BPP 3
#endif

#define IMAGE_FB_SIZE	(GLCD_WIDTH * GLCD_HEIGHT * IMGAE_COLOR_BPP)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE ( IMAGE_FB_SIZE + 1024)

#undef OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE	(1*1024)

// Stack size and priority for the new thread
#define MAINLOOP_TASK_STACK_SIZE    (48 * 1024)
#define INFERENCE_TASK_STACK_SIZE   (4 * 1024)
#define MAINLOOP_TASK_PRIO  4
#define INFERENCE_TASK_PRIO 3

#define NUM_FRAMEBUF 2  //1 or 2

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
    std::vector<arm::app::object_detection::DetectionResult> results;
} S_FRAMEBUF;

LOG_MODULE_REGISTER(main, LOG_LEVEL_INF);

// Frame buffer static allocation
__attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t s_au8FrameBuf0[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t s_au8JpegBuf[OMV_JPEG_BUF_SIZE];

#if (NUM_FRAMEBUF >= 2)
    __attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t s_au8FrameBuf1[OMV_FB_SIZE];
#endif

#if (NUM_FRAMEBUF >= 3)
    __attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t s_au8FrameBuf2[OMV_FB_SIZE];
#endif

// Global variables for omv frame buffer management
char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

// Frame buffer array
S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

// Define a stack and thread data structure
K_THREAD_STACK_DEFINE(s_sMainTaskStack, MAINLOOP_TASK_STACK_SIZE);
static struct k_thread s_sMainTask;

K_THREAD_STACK_DEFINE(s_sInfTaskStack, INFERENCE_TASK_STACK_SIZE);
static struct k_thread s_sInfTask;

namespace arm
{
namespace app
{
/* Tensor arena buffer */
static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

/* Optional getter function for the model pointer and its size. */
namespace yoloxnanonu
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace nn */
} /* namespace app */
} /* namespace arm */

void readMPUConifg(void)
{
	z_mpu_context_retained mpu_ctx;
	z_arm_save_mpu_context(&mpu_ctx);

	uint32_t i;

	for(i = 0; i < mpu_ctx.num_valid_regions; i++) {
		LOG_INF("MPU Region %d: RBAR=0x%08" PRIx32 ", RASR/RLAR=0x%08" PRIx32 "", i,
		       mpu_ctx.rbar[i], mpu_ctx.rasr_rlar[i]);
	}
	LOG_INF("mpu_ctx.mair[0] is %x ", mpu_ctx.mair[0]);
	LOG_INF("mpu_ctx.mair[1] is %x ", mpu_ctx.mair[1]);
}

// Load model file from SD card to HyperRAM
static int32_t PrepareModelToHyperRAM(void)
{
#define EACH_READ_SIZE 512

#if defined(__NUMAKER_M55M1__)
#define MODEL_FILE "0:\\nn_model.tflite"
#endif
#if defined(__NUGESTUREAI_M55M1__)
#define MODEL_FILE "1:\\nn_model.tflite"
#endif

#if defined(__NUMAKER_M55M1__)
    TCHAR sd_path[] = { '0', ':', 0 };    /* SD drive started from 0 */	
#endif
#if defined(__NUGESTUREAI_M55M1__)
    TCHAR sd_path[] = { '1', ':', 0 };    /* SD drive started from 1 */    
#endif
    f_chdrive(sd_path);          /* set default path */

	int32_t i32FileSize;
	int32_t i32FileReadIndex = 0;
	int32_t i32Read;
	
	if(!ModelFileReader_Initialize(MODEL_FILE))
	{
        LOG_ERR("Unable open model %s", MODEL_FILE);		
		return -1;
	}
	
	i32FileSize = ModelFileReader_FileSize();
    LOG_INF("Model file size %i ", i32FileSize);

	while(i32FileReadIndex < i32FileSize)
	{
		i32Read = ModelFileReader_ReadData((BYTE *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), EACH_READ_SIZE);
		if(i32Read < 0)
			break;
		i32FileReadIndex += i32Read;
	}
	
	if(i32FileReadIndex < i32FileSize)
	{
        LOG_ERR("Read Model file size is not enough");		
		return -2;
	}
	
#if 0
	/* verify */
	i32FileReadIndex = 0;
	ModelFileReader_Rewind();
	BYTE au8TempBuf[EACH_READ_SIZE];
	
	while(i32FileReadIndex < i32FileSize)
	{
		i32Read = ModelFileReader_ReadData((BYTE *)au8TempBuf, EACH_READ_SIZE);
		if(i32Read < 0)
			break;
		
		if(std::memcmp(au8TempBuf, (void *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), i32Read)!= 0)
		{
			LOG_ERR("verify the model file content is incorrect at %i", i32FileReadIndex);		
			return -3;
		}
		i32FileReadIndex += i32Read;
	}
	
#endif
	ModelFileReader_Finish();
	
	return i32FileSize;
}	

static void omv_init()
{
    image_t frameBuffer;
    int i;

    frameBuffer.w = GLCD_WIDTH;
    frameBuffer.h = GLCD_HEIGHT;
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * IMGAE_COLOR_BPP;
    frameBuffer.pixfmt = IMGAE_COLOR_PIXEL_FORMAT;

    _fb_base = (char *)s_au8FrameBuf0;
    _fb_end =  (char *)(s_au8FrameBuf0 + OMV_FB_SIZE - 1);
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = (char *)s_au8JpegBuf;

    fb_alloc_init0();

    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);

    for (i = 0 ; i < NUM_FRAMEBUF; i++)
    {
        s_asFramebuf[i].eState = eFRAMEBUF_EMPTY;
    }

    framebuffer_init_image(&s_asFramebuf[0].frameImage);

#if (NUM_FRAMEBUF >= 2)
    s_asFramebuf[1].frameImage.w = GLCD_WIDTH;
    s_asFramebuf[1].frameImage.h = GLCD_HEIGHT;
    s_asFramebuf[1].frameImage.size = GLCD_WIDTH * GLCD_HEIGHT * IMGAE_COLOR_BPP;
    s_asFramebuf[1].frameImage.pixfmt = IMGAE_COLOR_PIXEL_FORMAT;
    s_asFramebuf[1].frameImage.data = (uint8_t *)s_au8FrameBuf1;
#endif

#if (NUM_FRAMEBUF >= 3)
    s_asFramebuf[2].frameImage.w = GLCD_WIDTH;
    s_asFramebuf[2].frameImage.h = GLCD_HEIGHT;
    s_asFramebuf[2].frameImage.size = GLCD_WIDTH * GLCD_HEIGHT * IMGAE_COLOR_BPP;
    s_asFramebuf[2].frameImage.pixfmt = IMGAE_COLOR_PIXEL_FORMAT;
    s_asFramebuf[2].frameImage.data = (uint8_t *)s_au8FrameBuf2;
#endif

}

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

static void apply_person_track_ID(
    std::vector<arm::app::object_detection::DetectionResult> &results,
    BYTETracker *tracker
)
{
    //prepare detection objects for tracker
    arm::app::object_detection::DetectionResult detectBox;
	int boxSize = results.size();
    std::vector<struct Object> detObjects;
    int b = 0;

    //only process person class
    for(b = 0; b < boxSize; b ++)
    {
        struct Object detObject;
        detectBox = results[b];

        if(detectBox.m_cls == LABELS_PERSON_ID)
        {
            detObject.rect.x = detectBox.m_x0;
            detObject.rect.y = detectBox.m_y0;
            detObject.rect.w = detectBox.m_w;
            detObject.rect.h = detectBox.m_h;
            detObject.label = detectBox.m_cls;
            detObject.prob = static_cast<float>(detectBox.m_normalisedVal);
            detObjects.push_back(detObject);
        }
    }

    //remove previous person class detection results
    b = 0;
    while(b < results.size())
    {
        if(results[b].m_cls == LABELS_PERSON_ID)
        {
            results.erase(results.begin() + b);
        }
        else
        {
            b++;
        }
    }

    std::vector<STrack> trackedStracks = tracker->update(detObjects);

    int trackSize = trackedStracks.size();

    if(trackSize == 0)
        return;
 
    //append tracking results
    for(int t = 0; t < trackSize; t ++)
    {
        STrack track = trackedStracks[t];

        arm::app::object_detection::DetectionResult trackBox;
        trackBox.m_x0 = static_cast<int>(track.tlwh[0]);
        trackBox.m_y0 = static_cast<int>(track.tlwh[1]);
        trackBox.m_w  = static_cast<int>(track.tlwh[2]);
        trackBox.m_h  = static_cast<int>(track.tlwh[3]);
        trackBox.m_cls = LABELS_PERSON_ID;
        trackBox.m_normalisedVal = track.score;
        trackBox.m_trackID = track.track_id;
        results.push_back(trackBox);
    }

}

static void DrawImageDetectionBoxes(
    const std::vector<arm::app::object_detection::DetectionResult> &results,
    image_t *drawImg,
    std::vector<std::string> &labels)
{
    for (const auto &result : results)
    {
        imlib_draw_rectangle(drawImg, result.m_x0, result.m_y0, result.m_w, result.m_h, COLOR_B5_MAX, 1, false);
        imlib_draw_string(drawImg, result.m_x0, result.m_y0 - 16, labels[result.m_cls].c_str(), COLOR_B5_MAX, 2, 0, 0, false,
                          false, false, false, 0, false, false);
    }
}

static bool PresentInferenceResult(const std::vector<arm::app::object_detection::DetectionResult> &results,
                                   std::vector<std::string> &labels)
{
    /* If profiling is enabled, and the time is valid. */
    //LOG_INF("Final results: %d detected objects", results.size());

    for (uint32_t i = 0; i < results.size(); ++i)
    {
        LOG_INF("%" PRIu32 ") %s(%f) %s {x=%d,y=%d,w=%d,h=%d, id=%d}", i,
             labels[results[i].m_cls].c_str(),
             results[i].m_normalisedVal, "Box:",
             results[i].m_x0, results[i].m_y0, results[i].m_w, results[i].m_h, results[i].m_trackID);
    }

    return true;
}

// Resize source image to model input tensor size
static void ResizeImageToInputTensor(image_t *srcImg, TfLiteTensor *inputTensor, int inputImgCols, int inputImgRows)
{
    image_t resizeImg;
    rectangle_t ROIRect;

    ROIRect.x = 0;
    ROIRect.y = 0;
    ROIRect.w = srcImg->w;
    ROIRect.h = srcImg->h;

    resizeImg.w = inputImgCols;
    resizeImg.h = inputImgRows;
    resizeImg.data = (uint8_t *)inputTensor->data.data; //direct resize to input tensor buffer
    resizeImg.pixfmt = PIXFORMAT_RGB888;

    imlib_nvt_scale(srcImg, &resizeImg, &ROIRect);
}

// Copy and quantize image data to int8 tensor buffer
static void CopyImgToTensorInt8(void* srcData, void *destData, const size_t maxImageSize)
{
    auto* tmp_req_data = static_cast<uint8_t*>(srcData);
    auto* tmp_signed_req_data = static_cast<int8_t*>(destData);

    for (size_t i = 0; i < maxImageSize; i++) {
        tmp_signed_req_data[i] = (int8_t) (
                (int32_t) (tmp_req_data[i]) - 128);
    }
}

#if defined (__USE_UVC__)
static void UVCShowResultImage(image_t *Img)
{
#if (UVC_Color_Format == UVC_Format_YUY2)
    rectangle_t roi;

    image_t RGB565Img;
    image_t YUV422Img;

    RGB565Img.w = Img->w;
    RGB565Img.h = Img->h;
    RGB565Img.data = (uint8_t *)Img->data;
    RGB565Img.pixfmt = img->pixfmt;

    YUV422Img.w = RGB565Img.w;
    YUV422Img.h = RGB565Img.h;
    YUV422Img.data = (uint8_t *)Img->data;
    YUV422Img.pixfmt = PIXFORMAT_YUV422;

    roi.x = 0;
    roi.y = 0;
    roi.w = RGB565Img.w;
    roi.h = RGB565Img.h;
    imlib_nvt_scale(&RGB565Img, &YUV422Img, &roi);

#else
    image_t origImg;
    image_t vflipImg;

    origImg.w = Img->w;
    origImg.h = Img->h;
    origImg.data = (uint8_t *)Img->data;
    origImg.pixfmt = Img->pixfmt;

    vflipImg.w = origImg.w;
    vflipImg.h = origImg.h;
    vflipImg.data = (uint8_t *)Img->data;
    vflipImg.pixfmt = PIXFORMAT_RGB565;

    imlib_nvt_vflip(&origImg, &vflipImg);
#endif
    UVC_SendImage((uint32_t)Img->data, IMAGE_FB_SIZE, uvcStatus.StillImage);
}
#endif

K_MSGQ_DEFINE(infProcMsgQueue, sizeof(xInferenceJob *), 1, 4);
K_MSGQ_DEFINE(infRespMsgQueue, sizeof(xInferenceJob *), 1, 4);

// Main task
void main_task(void *pvArgs1, void *pvArgs2, void *pvArgs3)
{
    //display framebuffer
    image_t dispImage;
    rectangle_t ROIRect;

#if defined(__LOAD_MODEL_FROM_SD__)

	// Prepare model file to HyperRAM
	int32_t i32ModelSize;

	LOG_INF("==================== Load model file from SD card ================================="); 
	LOG_INF("Please copy NN_ModelInference/Model/xxx_vela.tflite to SDCard:/nn_model.tflite     "); 
	LOG_INF("==================================================================================="); 
	i32ModelSize = 	PrepareModelToHyperRAM();

	if(i32ModelSize <= 0 )
	{
        LOG_ERR("Failed to prepare model");
        return;
	}

    /* Model object creation and initialisation. */
    arm::app::YoloXnanoNu model;

    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    (unsigned char *)MODEL_AT_HYPERRAM_ADDR,
                    i32ModelSize))
    {
        LOG_ERR("Failed to initialise model");
        return;
    }
#else

	/* Model object creation and initialisation. */
    arm::app::YoloXnanoNu model;

    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::yoloxnanonu::GetModelPointer(),
                    arm::app::yoloxnanonu::GetModelLen()))
    {
        LOG_ERR("Failed to initialise model");
        return;
    }
#endif
    // Setup inference resource and create task
    struct ProcessTaskParams taskParam;

    taskParam.model = &model;
    taskParam.queueHandle = &infProcMsgQueue;

	k_tid_t infTID = k_thread_create(&s_sInfTask, s_sInfTaskStack,
			INFERENCE_TASK_STACK_SIZE,
			inferenceProcessTask,
			&taskParam, NULL, NULL,
			INFERENCE_TASK_PRIO,
			0, K_NO_WAIT);

	if(infTID == NULL)
	{
		LOG_ERR("Failed to create inference task");
		return;
	}

	// Get model input and output tensor information
    TfLiteTensor *inputTensor = model.GetInputTensor(0);
    TfLiteTensor *outputTensor = model.GetOutputTensor(0);

    if (!inputTensor->dims)
    {
        LOG_ERR("Invalid input tensor dims");
        return;
    }
    else if (inputTensor->dims->size < 3)
    {
        LOG_ERR("Input tensor dimension should be >= 3");
        return;
    }

	// Get input shape
	TfLiteIntArray *inputShape = model.GetInputShape(0);

	// Input image dimensions
	const int inputImgCols = inputShape->data[arm::app::YoloXnanoNu::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::YoloXnanoNu::ms_inputRowsIdx];

    // postProcess
    arm::app::object_detection::DetectorPostprocessing postProcess(0.6, 0.65, numClasses, 0);

    //label information
    std::vector<std::string> labels;
    GetLabelsVector(labels);

	//omv library init
    omv_init();
    framebuffer_init_image(&dispImage);

#if defined(__PROFILE__)
    arm::app::Profiler profiler;
    uint64_t u64StartCycle;
    uint64_t u64EndCycle;
    uint64_t u64CCAPStartCycle;
    uint64_t u64CCAPEndCycle;

    uint64_t u64EachStartCycle = 0;
    uint64_t u64EachEndCycle = 0;

#else
    pmu_reset_counters();
#endif

#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = pmu_get_systick_Count();
    u64PerfCycle += (k_sec_to_cyc_floor32(1) * EACH_PERF_SEC);

    S_FRAMEBUF *infFramebuf;
    S_FRAMEBUF *fullFramebuf;
    S_FRAMEBUF *emptyFramebuf;

    // Create inference job object
    struct xInferenceJob *inferenceJob = new (struct xInferenceJob);
    if(inferenceJob == nullptr)
    {
        LOG_ERR("Failed to create inference job");
        return;
    }   

    //Setup image senosr
    ImageSensor_Init(dispImage.w, dispImage.h);

#if defined(__PREVIEW_SUPPORT__)
    ImageSensor_Config(eIMAGE_FMT_RGB565, dispImage.w, dispImage.h, true);
#else
    ImageSensor_Config(eIMAGE_FMT_RGB888_U8, dispImage.w, dispImage.h, true);
#endif

#if defined (__USE_UVC__)
    // UVC init and HSUSBD start
    UVC_Init();
    HSUSBD_Start();
#endif

#if defined (__USE_LCD__)
    char szDisplayText[160];
    S_DISP_RECT sDispRect;

    Display_Init();
    Display_ClearLCD(C_WHITE);
#endif

    BYTETracker tracker(IMAGE_REAL_FRAMRATE, 30);

	while(1)
	{
        // Get empty frame buffer to store captured image
        emptyFramebuf = get_empty_framebuf();

        if (emptyFramebuf)
        {
#if defined(__PROFILE__)
            u64CCAPStartCycle = pmu_get_systick_Count();
#endif
            ImageSensor_TriggerCapture((uint32_t)(emptyFramebuf->frameImage.data));
        }

        // Get inferenced frame buffer and wait for inference done
        infFramebuf = get_inf_framebuf();

        if (infFramebuf)
        {
#if defined(__PROFILE__)
            u64EachStartCycle = pmu_get_systick_Count();
#endif
			// wait for inference process done 
			k_msgq_get(&infRespMsgQueue, &inferenceJob, K_FOREVER);
#if defined(__PROFILE__)
			u64EachEndCycle = pmu_get_systick_Count();
            LOG_INF("Wait inference done cycles %llu ", (u64EachEndCycle - u64EachStartCycle));
#endif
		}

        // Get CCAP filled frame buffer for new inference
        fullFramebuf = get_full_framebuf();

        if (fullFramebuf)
        {
            image_t *frameImg = &fullFramebuf->frameImage;
            uint8_t *qualizedDataPtr = (uint8_t*)inputTensor->data.data;

            if((frameImg->w != inputImgCols) || (frameImg->h != inputImgRows) || (frameImg->pixfmt != PIXFORMAT_RGB888))
            {
#if defined(__PROFILE__)
                u64StartCycle = pmu_get_systick_Count();
#endif
                ResizeImageToInputTensor(frameImg, inputTensor, inputImgCols, inputImgRows);
#if defined(__PROFILE__)
                u64EndCycle = pmu_get_systick_Count();
                LOG_INF("resize cycles %llu ", (u64EndCycle - u64StartCycle));
#endif

#if defined(__PROFILE__)
                u64StartCycle = pmu_get_systick_Count();
#endif
                arm::app::image::ConvertImgToInt8(inputTensor->data.data, inputTensor->bytes);
#if defined(__PROFILE__)
                u64EndCycle = pmu_get_systick_Count();
                LOG_INF("quantize cycles %llu ", (u64EndCycle - u64StartCycle));
#endif
            }
            else
            {
                //direct copy and quantize
                CopyImgToTensorInt8(frameImg->data, inputTensor->data.data, inputTensor->bytes);
            }

            //trigger inference
            inferenceJob->responseQueue = &infRespMsgQueue;
            inferenceJob->pPostProc = &postProcess;
            inferenceJob->modelCols = inputImgCols;
            inferenceJob->mode1Rows = inputImgRows;
            inferenceJob->srcImgWidth = fullFramebuf->frameImage.w;
            inferenceJob->srcImgHeight = fullFramebuf->frameImage.h;
            inferenceJob->results = &fullFramebuf->results;

			k_msgq_put(&infProcMsgQueue, &inferenceJob, K_FOREVER);
            fullFramebuf->eState = eFRAMEBUF_INF;
        }

        // Process inferenced frame buffer to display result
        if (infFramebuf)
        {
            //apply person track ID
            apply_person_track_ID(infFramebuf->results, &tracker);

#if defined(__PREVIEW_SUPPORT__)
            //draw bbox and render
            /* Draw boxes. */
            DrawImageDetectionBoxes(infFramebuf->results, &infFramebuf->frameImage, labels);
#endif

#if defined (__USE_LCD__)
            //Display image on LCD
            sDispRect.u32TopLeftX = 0;
            sDispRect.u32TopLeftY = 0;
            sDispRect.u32BottonRightX = ((dispImage.w * IMAGE_DISP_UPSCALE_FACTOR) - 1);
            sDispRect.u32BottonRightY = ((dispImage.h * IMAGE_DISP_UPSCALE_FACTOR) - 1);

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif

            Display_FillRect((uint16_t *)infFramebuf->frameImage.data, &sDispRect, IMAGE_DISP_UPSCALE_FACTOR);

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            LOG_INF("display image cycles %llu ", (u64EndCycle - u64StartCycle));
#endif

#endif

#if defined (__USE_UVC__)

            if (UVC_IsConnect())
            {
#if defined(__PROFILE__)
                u64StartCycle = pmu_get_systick_Count();
#endif
                UVCShowResultImage(&infFramebuf->frameImage);

#if defined(__PROFILE__)
                u64EndCycle = pmu_get_systick_Count();
                LOG_INF("UVC show image cycles %llu ", (u64EndCycle - u64StartCycle));
#endif
            }
#endif            
            u64PerfFrames ++;

            if ((uint64_t) pmu_get_systick_Count() > u64PerfCycle)
            {
                LOG_INF("Total inference rate: %llu", u64PerfFrames / EACH_PERF_SEC);
                u64PerfCycle = (uint64_t)pmu_get_systick_Count() + (uint64_t)(k_sec_to_cyc_floor32(1) * EACH_PERF_SEC);
                u64PerfFrames = 0;
            }

            PresentInferenceResult(infFramebuf->results, labels);
            infFramebuf->eState = eFRAMEBUF_EMPTY;
        }

        // Wait for CCAP capture done and mark frame buffer full
        if (emptyFramebuf)
        {
            ImageSensor_WaitCaptureDone();

#if defined(__PROFILE__)
            u64CCAPEndCycle = pmu_get_systick_Count();
            LOG_INF("ccap capture cycles %llu ", (u64CCAPEndCycle - u64CCAPStartCycle));
#endif
            emptyFramebuf->results.clear();
            emptyFramebuf->eState = eFRAMEBUF_FULL;
		}

		k_yield();
	}

}

/* Zephyr application. NOTE: Additional tasks may require increased heap size. */
int main()
{
    // Initialise HyperRAM and SDH hardware source
	BoardInit();

    // Configure SRAM2 MPU region
	//SRAM2MPUConifg(); //Unnecessary if configured by sram2_region.overlay's "zephyr,memory-attr = <DT_MEM_ARM_MPU_RAM_NOCACHE>"
	// Make sure SRAM2 MPU config is correct
	readMPUConifg();

	// Create main task thread
	k_tid_t mainTID = k_thread_create(&s_sMainTask, s_sMainTaskStack,
			MAINLOOP_TASK_STACK_SIZE,
			main_task,
			NULL, NULL, NULL,
			MAINLOOP_TASK_PRIO,
			0, K_NO_WAIT);

	if(mainTID == NULL)
	{
		LOG_ERR("Failed to create main task");
		return 1;
	}

	return 0;
}
