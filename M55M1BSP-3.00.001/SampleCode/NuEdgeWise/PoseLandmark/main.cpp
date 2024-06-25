/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    Pose landmark network sample. Demonstrate hand landmark detect.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include "BoardInit.hpp"      /* Board initialisation */
#include "log_macros.h"      /* Logging macros (optional) */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "PoseLandmarkModel.hpp"       /* Model API */
#include "PoseLandmarkPostProcessing.hpp"

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

//#define __PROFILE__
#define __USE_DISPLAY__

#include "Profiler.hpp"

#include "ImageSensor.h"

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#define NUM_FRAMEBUF 2  //1 or 2

#define POSE_LANDMARK_SCREEN_TENSOR_INDEX		3
#define POSE_PRESENCE_TENSOR_INDEX				2
#define POSE_SEGMENTATION_TENSOR_INDEX			0
#define POSE_HEATMAP_TENSOR_INDEX				1
#define POSE_LANDMARK_WORLD_TENSOR_INDEX		4

#define POSE_PRESENCE_THRESHOLD  				(0.9)

enum{
	ePOSE_KP_INDEX_NOSE,				//0
	ePOSE_KP_INDEX_LEFT_EYE_INNER,		//1
	ePOSE_KP_INDEX_LEFT_EYE,			//2
	ePOSE_KP_INDEX_LEFT_EYE_OUTER,		//3
	ePOSE_KP_INDEX_RIGHT_EYE_INNER,  	//4
	ePOSE_KP_INDEX_RIGHT_EYE,			//5
	ePOSE_KP_INDEX_RIGHT_EYE_OUTER,		//6
	ePOSE_KP_INDEX_LEFT_EAR,			//7
	ePOSE_KP_INDEX_RIGHT_EAR,			//8
	ePOSE_KP_INDEX_LEFT_MOUTH,			//9
	ePOSE_KP_INDEX_RIGHT_MOUTH,			//10
	ePOSE_KP_INDEX_LEFT_SHOULDER,		//11
	ePOSE_KP_INDEX_RIGHT_SHOULDER,		//12
	ePOSE_KP_INDEX_LEFT_ELBOW,			//13
	ePOSE_KP_INDEX_RIGHT_ELBOW,			//14
	ePOSE_KP_INDEX_LEFT_WRIST,			//15
	ePOSE_KP_INDEX_RIGHT_WRIST,			//16
	ePOSE_KP_INDEX_LEFT_PINKY,			//17
	ePOSE_KP_INDEX_RIGHT_PINKY,			//18
	ePOSE_KP_INDEX_LEFT_INDEX,			//19
	ePOSE_KP_INDEX_RIGHT_INDEX,			//20
	ePOSE_KP_INDEX_LEFT_THUMB,			//21
	ePOSE_KP_INDEX_RIGHT_THUMB,			//22
	ePOSE_KP_INDEX_LEFT_HIP,			//23
	ePOSE_KP_INDEX_RIGHT_HIP,			//24
	ePOSE_KP_INDEX_LEFT_KNEE,			//25
	ePOSE_KP_INDEX_RIGHT_KNEE,			//26
	ePOSE_KP_INDEX_LEFT_ANKLE,			//27
	ePOSE_KP_INDEX_RIGHT_ANKLE,			//28
	ePOSE_KP_INDEX_LEFT_HEEL,			//29
	ePOSE_KP_INDEX_RIGHT_HEEL,			//30
	ePOSE_KP_INDEX_LEFT_FOOT_INDEX,		//31
	ePOSE_KP_INDEX_RIGHT_FOOT_INDEX,	//32
	ePOSE_KP_NUMS,						//33
}E_POSE_KP_INDEX;


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
    std::vector<arm::app::pose_landmark::KeypointResult> results;
} S_FRAMEBUF;


S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

namespace arm
{
namespace app
{
/* Tensor arena buffer */
static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

/* Optional getter function for the model pointer and its size. */
namespace pose_landmark
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace hand_landmark */
	
} /* namespace app */
} /* namespace arm */

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
#define GLCD_WIDTH	320
#define GLCD_HEIGHT	240

#undef OMV_FB_SIZE
#define OMV_FB_SIZE ((GLCD_WIDTH * GLCD_HEIGHT * 2) + 1024)

#undef OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE	(1*1024)

__attribute__((section(".bss.sram.data"), aligned(16))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.sram.data"), aligned(16))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

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

static void DrawPoseLandmark(
    const std::vector<arm::app::pose_landmark::KeypointResult> &results,
    image_t *drawImg
)
{
	int i;
	
	arm::app::pose_landmark::KeypointResult keyPoint;
	arm::app::pose_landmark::KeypointResult keyPointTemp;
	
	int lineColor = COLOR_R5_G6_B5_TO_RGB565(0,COLOR_G6_MAX, 0);
	
	for(i = 0; i < ePOSE_KP_NUMS; i ++)
	{
		keyPoint = results[i];
		
		if(keyPoint.m_visibility < POSE_PRESENCE_THRESHOLD)
			continue;

		//draw points
		imlib_draw_circle(drawImg, keyPoint.m_x, keyPoint.m_y, 1, COLOR_B5_MAX, 1, true);

		//draw lines
		if( i == ePOSE_KP_INDEX_NOSE || i == ePOSE_KP_INDEX_LEFT_MOUTH || i == ePOSE_KP_INDEX_LEFT_SHOULDER)
		{
			//Don't draw line
		}
		else if(i == ePOSE_KP_INDEX_RIGHT_EYE_INNER)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_NOSE];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_LEFT_EAR)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_EYE_OUTER];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_EAR)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_EYE_OUTER];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_SHOULDER)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_SHOULDER];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_LEFT_ELBOW)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_SHOULDER];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_ELBOW)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_SHOULDER];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_LEFT_WRIST)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_ELBOW];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_WRIST)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_ELBOW];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_LEFT_PINKY)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_WRIST];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_PINKY)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_WRIST];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_LEFT_INDEX)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_WRIST];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_PINKY];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_INDEX)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_WRIST];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_PINKY];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_LEFT_THUMB)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_WRIST];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_THUMB)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_WRIST];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_LEFT_HIP)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_SHOULDER];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_HIP)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_HIP];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_SHOULDER];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_LEFT_KNEE)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_HIP];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_KNEE)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_HIP];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_LEFT_ANKLE)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_KNEE];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_ANKLE)		//28
		{
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_KNEE];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_LEFT_HEEL)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_ANKLE];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_HEEL)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_ANKLE];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_LEFT_FOOT_INDEX)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_ANKLE];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
			keyPointTemp = results[ePOSE_KP_INDEX_LEFT_HEEL];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else if( i == ePOSE_KP_INDEX_RIGHT_FOOT_INDEX)
		{
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_ANKLE];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
			keyPointTemp = results[ePOSE_KP_INDEX_RIGHT_HEEL];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		else
		{
			keyPointTemp = results[i - 1];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, lineColor, 1);
		}
		
	}
}



int main()
{
    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

    /* Model object creation and initialisation. */
    arm::app::PoseLandmarkModel model;

    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::pose_landmark::GetModelPointer(),
                    arm::app::pose_landmark::GetModelLen()))
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
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena) + ACTIVATION_BUF_SZ - 1),        // Limit
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

    TfLiteTensor *inputTensor   = model.GetInputTensor(0);

    if (!inputTensor->dims)
    {
        printf_err("Invalid input tensor dims\n");
        return 2;
    }
    else if (inputTensor->dims->size < 3)
    {
        printf_err("Input tensor dimension should be >= 3\n");
        return 3;
    }

    TfLiteIntArray *inputShape = model.GetInputShape(0);

    const int inputImgCols = inputShape->data[arm::app::PoseLandmarkModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::PoseLandmarkModel::ms_inputRowsIdx];
    const uint32_t nChannels = inputShape->data[arm::app::PoseLandmarkModel::ms_inputChannelsIdx];

    /* Hand landmark model preprocessing is image conversion from uint8 to [0,1] float values,
     * then quantize them with input quantization info. */
    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);

    // postProcess
    arm::app::pose_landmark::PoseLandmarkPostProcessing postProcess(POSE_PRESENCE_THRESHOLD);
	
    //display framebuffer
    image_t frameBuffer;
    rectangle_t roi;

    //omv library init
    omv_init();
    framebuffer_init_image(&frameBuffer);

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

    //Setup image senosr
    ImageSensor_Init();
    ImageSensor_Config(eIMAGE_FMT_RGB565, frameBuffer.w, frameBuffer.h, true);

#if defined (__USE_DISPLAY__)
    char szDisplayText[100];
    S_DISP_RECT sDispRect;

    Display_Init();
    Display_ClearLCD(C_WHITE);
#endif

    while (1)
    {
        emptyFramebuf = get_empty_framebuf();

        if (emptyFramebuf)
        {
            //capture frame from CCAP
#if defined(__PROFILE__)
            u64CCAPStartCycle = pmu_get_systick_Count();
#endif

            ImageSensor_TriggerCapture((uint32_t)(emptyFramebuf->frameImage.data));
		}
		
        fullFramebuf = get_full_framebuf();

        if (fullFramebuf)
        {
            //resize full image to input tensor
            image_t resizeImg;

            roi.x = 0;
            roi.y = 0;
            roi.w = fullFramebuf->frameImage.w;
            roi.h = fullFramebuf->frameImage.h;

            resizeImg.w = inputImgCols;
            resizeImg.h = inputImgRows;
            resizeImg.data = (uint8_t *)inputTensor->data.data; //direct resize to input tensor buffer
            resizeImg.pixfmt = PIXFORMAT_RGB888;

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif
            imlib_nvt_scale(&fullFramebuf->frameImage, &resizeImg, &roi);

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("resize cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#if defined(__PROFILE__)
            u64StartCycle = pmu_get_systick_Count();
#endif
			//Quantize input tensor data
			auto *req_data = static_cast<uint8_t *>(inputTensor->data.data);
			auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);

			for (size_t i = 0; i < inputTensor->bytes; i++)
			{
//				auto i_data_int8 = static_cast<int8_t>(((static_cast<float>(req_data[i]) / 255.0f) / inQuantParams.scale) + inQuantParams.offset);
//				signed_req_data[i] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
				signed_req_data[i] = static_cast<int8_t>(req_data[i]) - 128;
			}

#if defined(__PROFILE__)
            u64EndCycle = pmu_get_systick_Count();
            info("quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#if defined(__PROFILE__)
			profiler.StartProfiling("Inference");
#endif

			model.RunInference();

#if defined(__PROFILE__)
			profiler.StopProfiling();
			profiler.PrintProfilingResult();
#endif

            fullFramebuf->eState = eFRAMEBUF_INF;
        }
        infFramebuf = get_inf_framebuf();

        if (infFramebuf)
        {
			//post process
			TfLiteTensor *modelOutput0 = model.GetOutputTensor(POSE_LANDMARK_SCREEN_TENSOR_INDEX);
			TfLiteTensor *modelOutput1 = model.GetOutputTensor(POSE_PRESENCE_TENSOR_INDEX);

#if defined(__PROFILE__)
			u64StartCycle = pmu_get_systick_Count();
#endif
			postProcess.RunPostProcessing(
				inputImgCols,
				inputImgRows,
				infFramebuf->frameImage.w,
				infFramebuf->frameImage.h,
				modelOutput0,
				modelOutput1,
				infFramebuf->results);

#if defined(__PROFILE__)
			u64EndCycle = pmu_get_systick_Count();
			info("post processing cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

            //draw bbox and render
            /* Draw boxes. */
			if(infFramebuf->results.size())
			{
#if defined(__PROFILE__)
				u64StartCycle = pmu_get_systick_Count();
#endif
				DrawPoseLandmark(infFramebuf->results, &infFramebuf->frameImage);
#if defined(__PROFILE__)
				u64EndCycle = pmu_get_systick_Count();
				info("draw hand landmark cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif
			}

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

            u64PerfFrames ++;
			if ((uint64_t) pmu_get_systick_Count() > u64PerfCycle)
            {
                info("Total inference rate: %llu\n", u64PerfFrames / EACH_PERF_SEC);
#if defined (__USE_DISPLAY__)
                sprintf(szDisplayText, "Frame Rate %llu", u64PerfFrames / EACH_PERF_SEC);
                //sprintf(szDisplayText,"Time %llu",(uint64_t) pmu_get_systick_Count() / (uint64_t)SystemCoreClock);
                //info("Running %s sec \n", szDisplayText);

                sDispRect.u32TopLeftX = 0;
                sDispRect.u32TopLeftY = frameBuffer.h;
                sDispRect.u32BottonRightX = (frameBuffer.w);
                sDispRect.u32BottonRightY = (frameBuffer.h + (FONT_HTIGHT) - 1);

                Display_ClearRect(C_WHITE, &sDispRect);
                Display_PutText(
                    szDisplayText,
                    strlen(szDisplayText),
                    0,
                    frameBuffer.h,
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
			//Capture new image

			ImageSensor_WaitCaptureDone();
#if defined(__PROFILE__)
			u64CCAPEndCycle = pmu_get_systick_Count();
			info("ccap capture cycles %llu \n", (u64CCAPEndCycle - u64CCAPStartCycle));
#endif
            emptyFramebuf->eState = eFRAMEBUF_FULL;		
		}
    }

    return 0;
}
