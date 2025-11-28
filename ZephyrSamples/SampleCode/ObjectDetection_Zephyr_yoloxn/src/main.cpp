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

// Define activation buffer size for model inference
#undef ACTIVATION_BUF_SZ
#define ACTIVATION_BUF_SZ (0x00CC000)
#include "BufAttributes.hpp" /* Buffer attributes to be applied */

//#define __PROFILE__
//#define __LOAD_MODEL_FROM_SD__
//#define __USE_UVC__

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
#if defined(__USE_UVC__)
    //UVC only support QVGA, QQVGA
    #define GLCD_WIDTH  320
    #define GLCD_HEIGHT 240
#else
    #define GLCD_WIDTH  320
    #define GLCD_HEIGHT 240
#endif

#define IMAGE_FB_SIZE	(GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE ( IMAGE_FB_SIZE + 1024)

#undef OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE	(1*1024)

// Stack size and priority for the new thread
#define THREAD_STACK_SIZE (2*1024)
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

// Frame buffer static allocation
__attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t s_au8FrameBuf0[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t s_au8JpegBuf[OMV_JPEG_BUF_SIZE];

#if (NUM_FRAMEBUF == 2)
    __attribute__((section(".bss.vram.data"), aligned(32))) static uint8_t s_au8FrameBuf1[OMV_FB_SIZE];
#endif

// Global variables for omv frame buffer management
char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

// Frame buffer array
S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

// Define a stack and thread data structure
K_THREAD_STACK_DEFINE(s_sMainTaskStack, THREAD_STACK_SIZE);
static struct k_thread s_sMainTask;

K_THREAD_STACK_DEFINE(s_sInfTaskStack, THREAD_STACK_SIZE);
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
		printf("MPU Region %d: RBAR=0x%08" PRIx32 ", RASR/RLAR=0x%08" PRIx32 "\n", i,
		       mpu_ctx.rbar[i], mpu_ctx.rasr_rlar[i]);
	}
	printf("mpu_ctx.mair[0] is %x \n", mpu_ctx.mair[0]);
	printf("mpu_ctx.mair[1] is %x \n", mpu_ctx.mair[1]);
}

/*Configure sram2 MPU attribute by manaual, beacuse not support configu MPU by DT */
extern uint8_t   __sram2_noinit_start;
extern uint8_t   __sram2_noinit_end;

/* Configure MPU to map SRAM2 as Non-cacheable region */
void SRAM2MPUConifg(void)
{
	z_mpu_context_retained mpu_ctx;
	z_arm_save_mpu_context(&mpu_ctx);

	uint32_t i;

	for(i = 0; i < mpu_ctx.num_valid_regions; i++) {
		if ((mpu_ctx.rasr_rlar[i] & 0x1U) == 0U) {
			break;
		}
	}

	if(i == mpu_ctx.num_valid_regions) {
		printf("All regions are used \n");
		return;
	}

	mpu_ctx.rbar[i] = ARM_MPU_RBAR((uint32_t)&__sram2_noinit_start,
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         0,                 // Non-Privileged
                         1),                // eXecute Never enabled
	mpu_ctx.rasr_rlar[i] = ARM_MPU_RLAR((uint32_t)(&__sram2_noinit_end) - 1U
								 ,MPU_MAIR_INDEX_SRAM_NOCACHE); //size 128KB, enable

	printf("Config MPU Region %d: RBAR=0x%08" PRIx32 ", RASR/RLAR=0x%08" PRIx32 "\n", i,
		       mpu_ctx.rbar[i], mpu_ctx.rasr_rlar[i]);
	printf("NORMAL_OUTER_INNER_NON_CACHEABLE is %x \n", MPU_MAIR_INDEX_SRAM_NOCACHE);
	printf("mpu_ctx.mair[0] is %x \n", mpu_ctx.mair[0]);
	printf("mpu_ctx.mair[1] is %x \n", mpu_ctx.mair[1]);
	printf("num_valid_regions is %d \n", mpu_ctx.num_valid_regions);

	z_arm_restore_mpu_context(&mpu_ctx);
}

// Load model file from SD card to HyperRAM
static int32_t PrepareModelToHyperRAM(void)
{
#define MODEL_FILE "0:\\nn_model.tflite"
#define EACH_READ_SIZE 512
	
    TCHAR sd_path[] = { '0', ':', 0 };    /* SD drive started from 0 */	
    f_chdrive(sd_path);          /* set default path */

	int32_t i32FileSize;
	int32_t i32FileReadIndex = 0;
	int32_t i32Read;
	
	if(!ModelFileReader_Initialize(MODEL_FILE))
	{
        printf("Unable open model %s\n", MODEL_FILE);		
		return -1;
	}
	
	i32FileSize = ModelFileReader_FileSize();
    printf("Model file size %i \n", i32FileSize);

	while(i32FileReadIndex < i32FileSize)
	{
		i32Read = ModelFileReader_ReadData((BYTE *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), EACH_READ_SIZE);
		if(i32Read < 0)
			break;
		i32FileReadIndex += i32Read;
	}
	
	if(i32FileReadIndex < i32FileSize)
	{
        printf("Read Model file size is not enough\n");		
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
			printf_err("verify the model file content is incorrect at %i \n", i32FileReadIndex);		
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
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

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

#if (NUM_FRAMEBUF == 2)
    s_asFramebuf[1].frameImage.w = GLCD_WIDTH;
    s_asFramebuf[1].frameImage.h = GLCD_HEIGHT;
    s_asFramebuf[1].frameImage.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    s_asFramebuf[1].frameImage.pixfmt = PIXFORMAT_RGB565;
    s_asFramebuf[1].frameImage.data = (uint8_t *)s_au8FrameBuf1;
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
    //info("Final results:\n");

    for (uint32_t i = 0; i < results.size(); ++i)
    {
        printf("%" PRIu32 ") %s(%f) -> %s {x=%d,y=%d,w=%d,h=%d}\n", i,
             labels[results[i].m_cls].c_str(),
             results[i].m_normalisedVal, "Detection box:",
             results[i].m_x0, results[i].m_y0, results[i].m_w, results[i].m_h);
    }

    return true;
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
    RGB565Img.pixfmt = PIXFORMAT_RGB565;

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
    origImg.pixfmt = PIXFORMAT_RGB565;

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

	printf("==================== Load model file from SD card =================================\n"); 
	printf("Please copy NN_ModelInference/Model/xxx_vela.tflite to SDCard:/nn_model.tflite     \n"); 
	printf("===================================================================================\n"); 
	i32ModelSize = 	PrepareModelToHyperRAM();

	if(i32ModelSize <= 0 )
	{
        printf("Failed to prepare model\n");
        return;
	}

    /* Model object creation and initialisation. */
    arm::app::YoloXnanoNu model;

    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    (unsigned char *)MODEL_AT_HYPERRAM_ADDR,
                    i32ModelSize))
    {
        printf("Failed to initialise model\n");
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
        printf("Failed to initialise model\n");
        return;
    }
#endif
    // Setup inference resource and create task
    struct ProcessTaskParams taskParam;

    taskParam.model = &model;
    taskParam.queueHandle = &infProcMsgQueue;

	k_tid_t infTID = k_thread_create(&s_sInfTask, s_sInfTaskStack,
			THREAD_STACK_SIZE,
			inferenceProcessTask,
			&taskParam, NULL, NULL,
			INFERENCE_TASK_PRIO,
			0, K_NO_WAIT);

	if(infTID == NULL)
	{
		printf("Failed to create inference task\n");
		return;
	}

	// Get model input and output tensor information
    TfLiteTensor *inputTensor = model.GetInputTensor(0);
    TfLiteTensor *outputTensor = model.GetOutputTensor(0);

    if (!inputTensor->dims)
    {
        printf("Invalid input tensor dims\n");
        return;
    }
    else if (inputTensor->dims->size < 3)
    {
        printf("Input tensor dimension should be >= 3\n");
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
        printf("Failed to create inference job\n");
        return;
    }   

    //Setup image senosr
    ImageSensor_Init();
    ImageSensor_Config(eIMAGE_FMT_RGB565, dispImage.w, dispImage.h, true);

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
            printf("Wait inference done cycles %llu \n", (u64EachEndCycle - u64EachStartCycle));
#endif
		}

        // Get CCAP filled frame buffer for new inference
        fullFramebuf = get_full_framebuf();

        if (fullFramebuf)
        {
            //resize full image to input tensor
            image_t resizeImg;

            ROIRect.x = 0;
            ROIRect.y = 0;
            ROIRect.w = fullFramebuf->frameImage.w;
            ROIRect.h = fullFramebuf->frameImage.h;

            resizeImg.w = inputImgCols;
            resizeImg.h = inputImgRows;
            resizeImg.data = (uint8_t *)inputTensor->data.data; //direct resize to input tensor buffer
			resizeImg.pixfmt = PIXFORMAT_RGB888;

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif
            imlib_nvt_scale(&fullFramebuf->frameImage, &resizeImg, &ROIRect);

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            printf("resize cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif

            /* If the data is signed. */
            if (model.IsDataSigned())
            {
                arm::app::image::ConvertImgToInt8(inputTensor->data.data, inputTensor->bytes);
            }

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            printf("quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif
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
            //draw bbox and render
            /* Draw boxes. */
            DrawImageDetectionBoxes(infFramebuf->results, &infFramebuf->frameImage, labels);

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
            info("display image cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#endif




#if defined (__USE_UVC__)

            if (UVC_IsConnect())
            {
                UVCShowResultImage(&infFramebuf->frameImage);
            }
#endif            
            u64PerfFrames ++;

            if ((uint64_t) pmu_get_systick_Count() > u64PerfCycle)
            {
                printf("Total inference rate: %llu\n", u64PerfFrames / EACH_PERF_SEC);
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
            printf("ccap capture cycles %llu \n", (u64CCAPEndCycle - u64CCAPStartCycle));
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
			THREAD_STACK_SIZE,
			main_task,
			NULL, NULL, NULL,
			MAINLOOP_TASK_PRIO,
			0, K_NO_WAIT);

	if(mainTID == NULL)
	{
		printf("Failed to create main task\n");
		return 1;
	}

	return 0;
}
