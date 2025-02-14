/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    Hand pose recognition sample. Demonstrate hand pose recognition
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include "BoardInit.hpp"      /* Board initialisation */
#include "log_macros.h"      /* Logging macros (optional) */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "HandLandmarkModel.hpp"       /* Model API */
#include "PointHistoryClassifierModel.hpp"
#include "HandLandmarkPostProcessing.hpp"
#include "Labels.hpp"
#include "PointHistoryProcessing.hpp"
#include "Classifier.hpp"    /* Classifier for the result */
#include "ClassificationResult.hpp"

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"
#include "ModelFileReader.h"
#include "ff.h"

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

//#define __PROFILE__
#define __USE_DISPLAY__
//#define __USE_UVC__

#include "Profiler.hpp"

#include "ImageSensor.h"

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

#define NUM_FRAMEBUF 2  //1 or 2
#define ACTIVATION_HL_BUF_SZ ACTIVATION_BUF_SZ
#define HAND_POSE_KEYPOINT 8 //INDEX_FINGER_TIP

#define MODEL_AT_HYPERRAM_ADDR (0x82400000)

#define HAND_LANDMARK_SCREEN_TENSOR_INDEX    3
#define HAND_PRESENCE_TENSOR_INDEX           2
#define HANDEDNESS_TENSOR_INDEX              0
#define HAND_LANDMARK_WORLD_TENSOR_INDEX     1

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
    std::vector<arm::app::hand_landmark::KeypointResult> results;
} S_FRAMEBUF;


S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

using PointHistoryClassifier = arm::app::Classifier;

namespace arm
{
namespace app
{
/* Tensor arena buffer for handlandmark*/
static uint8_t tensorArenaHandLandmark[ACTIVATION_HL_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

/* Tensor arena buffer for point history*/
static uint8_t tensorArenaPointHistory[ACTIVATION_PH_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

/* Optional getter function for the model pointer and its size. */
namespace point_history
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
#if defined(__USE_UVC__)
//UVC only support QVGA, QQVGA
#define GLCD_WIDTH	320
#define GLCD_HEIGHT	240
#else
#define GLCD_WIDTH 320
#define GLCD_HEIGHT 240
#endif

//RGB565
#define IMAGE_FB_SIZE	(GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE ((IMAGE_FB_SIZE) + 1024)

#undef OMV_FB_ALLOC_SIZE
#define OMV_FB_ALLOC_SIZE	(1*1024)

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

static void DrawHandLandmark(
    const std::vector<arm::app::hand_landmark::KeypointResult> &results,
    image_t *drawImg
)
{
	int i;
	
	arm::app::hand_landmark::KeypointResult keyPoint;
	arm::app::hand_landmark::KeypointResult keyPointTemp;
	
	for(i = 0; i < results.size(); i ++)
	{
		keyPoint = results[i];
		
		//draw points
		imlib_draw_circle(drawImg, keyPoint.m_x, keyPoint.m_y, 1, COLOR_B5_MAX, 1, true);
		
		//draw lines
		if (i == 0)
		{
			//Don't draw line
		}
		else if(i == 5)
		{
			keyPointTemp = results[0];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, COLOR_B5_MAX, 1);
		}
		else if(i == 9)
		{
			keyPointTemp = results[5];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, COLOR_B5_MAX, 1);
		}
		else if(i == 13)
		{
			keyPointTemp = results[9];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, COLOR_B5_MAX, 1);
		}
		else if(i == 17)
		{
			keyPointTemp = results[13];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, COLOR_B5_MAX, 1);
			keyPointTemp = results[0];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, COLOR_B5_MAX, 1);
		}
		else
		{
			keyPointTemp = results[ i - 1 ];
			imlib_draw_line(drawImg, keyPoint.m_x, keyPoint.m_y, keyPointTemp.m_x, keyPointTemp.m_y, COLOR_B5_MAX, 1);
		}
	}
}

static int32_t PrepareModelToHyperRAM(void)
{
#define MODEL_FILE "0:\\hand_landmark.tflite"
#define EACH_READ_SIZE 512
	
    TCHAR sd_path[] = { '0', ':', 0 };    /* SD drive started from 0 */	
    f_chdrive(sd_path);          /* set default path */

	int32_t i32FileSize;
	int32_t i32FileReadIndex = 0;
	int32_t i32Read;
	
	if(!ModelFileReader_Initialize(MODEL_FILE))
	{
        printf_err("Unable open model %s\n", MODEL_FILE);		
		return -1;
	}
	
	i32FileSize = ModelFileReader_FileSize();
    info("Model file size %i \n", i32FileSize);

	while(i32FileReadIndex < i32FileSize)
	{
		i32Read = ModelFileReader_ReadData((BYTE *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), EACH_READ_SIZE);
		if(i32Read < 0)
			break;
		i32FileReadIndex += i32Read;
	}
	
	if(i32FileReadIndex < i32FileSize)
	{
        printf_err("Read Model file size is not enough\n");		
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

static int32_t s_i32PrevLabelIndex = -1;

bool JudgePoseDetect(arm::app::ClassificationResult &result)
{
	
	if((s_i32PrevLabelIndex == -1) || (s_i32PrevLabelIndex == 0)) //if previous pose is unknown or stop
	{
		s_i32PrevLabelIndex = (int32_t)result.m_labelIdx;
		return true;
	}

	if(s_i32PrevLabelIndex != result.m_labelIdx)
	{
		s_i32PrevLabelIndex = -1;
		return false;
	}
	
	s_i32PrevLabelIndex = (int32_t)result.m_labelIdx;
	return true;
}

int main()
{
    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

	/* Copy model file from SD to HyperRAM*/
	int32_t i32ModelSize;
	
	
	i32ModelSize = PrepareModelToHyperRAM();
	
	if(i32ModelSize <= 0 )
	{
        printf_err("Failed to prepare model\n");
        return 1;
	}

    /* Model object creation and initialisation. */
    arm::app::HandLandmarkModel handLandmarkModel;
    arm::app::PointHistoryClassifierModel pointHistoryClassifierModel;

    if (!handLandmarkModel.Init(arm::app::tensorArenaHandLandmark,
                    sizeof(arm::app::tensorArenaHandLandmark),
                    (unsigned char *)MODEL_AT_HYPERRAM_ADDR,
                    i32ModelSize))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }

    if (!pointHistoryClassifierModel.Init(arm::app::tensorArenaPointHistory ,
                    sizeof(arm::app::tensorArenaPointHistory),
                    arm::app::point_history::GetModelPointer(),
                    arm::app::point_history::GetModelLen()))
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
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArenaHandLandmark),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArenaHandLandmark) + ACTIVATION_HL_BUF_SZ - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - Write-Through, Read-allocate
        },
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArenaPointHistory),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArenaPointHistory) + ACTIVATION_PH_BUF_SZ - 1),        // Limit
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

    TfLiteTensor *inputTensorHandLandmark = handLandmarkModel.GetInputTensor(0);

    if (!inputTensorHandLandmark->dims)
    {
        printf_err("Invalid input tensor dims\n");
        return 2;
    }
    else if (inputTensorHandLandmark->dims->size < 3)
    {
        printf_err("Input tensor dimension should be >= 3\n");
        return 3;
    }

    TfLiteIntArray *inputShape = handLandmarkModel.GetInputShape(0);

    const int inputImgCols = inputShape->data[arm::app::HandLandmarkModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::HandLandmarkModel::ms_inputRowsIdx];
    const uint32_t nChannels = inputShape->data[arm::app::HandLandmarkModel::ms_inputChannelsIdx];

    /* HandLandmark model preprocessing is image conversion from uint8 to [0,1] float values,
     * then quantize them with input quantization info. */
    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensorHandLandmark);

    // postProcess
    arm::app::hand_landmark::HandLandmarkPostProcessing postProcessHandLandmark(0.5);
	
    //label information
    std::vector <std::string> labels;
    GetLabelsVector(labels);

    /* Set up pre and post-processing. */
    arm::app::PointHistoryPreProcess preProcessPointHistory = arm::app::PointHistoryPreProcess(&pointHistoryClassifierModel);

    std::vector<arm::app::ClassificationResult> results;
    std::string predictLabelInfo;
    PointHistoryClassifier classifier;  /* Classifier object. */
    arm::app::PointHistoryPostProcess postProcessPointHistory = arm::app::PointHistoryPostProcess(classifier, &pointHistoryClassifierModel,
                                                                              labels, results);
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
	S_DISP_RECT sDispRect;

    Display_Init();
    Display_ClearLCD(C_WHITE);
#endif

#if defined (__USE_UVC__)
	UVC_Init();
    HSUSBD_Start();
#endif

    char szPoseText[100];
    char szFrameRateText[50];
	bool bCollectPointDone;

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
            resizeImg.data = (uint8_t *)inputTensorHandLandmark->data.data; //direct resize to input tensor buffer
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
			auto *req_data = static_cast<uint8_t *>(inputTensorHandLandmark->data.data);
			auto *signed_req_data = static_cast<int8_t *>(inputTensorHandLandmark->data.data);

			for (size_t i = 0; i < inputTensorHandLandmark->bytes; i++)
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

			handLandmarkModel.RunInference();

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
			TfLiteTensor *modelOutput0 = handLandmarkModel.GetOutputTensor(HAND_LANDMARK_SCREEN_TENSOR_INDEX);
			TfLiteTensor *modelOutput1 = handLandmarkModel.GetOutputTensor(HAND_PRESENCE_TENSOR_INDEX);

#if defined(__PROFILE__)
			u64StartCycle = pmu_get_systick_Count();
#endif
			postProcessHandLandmark.RunPostProcessing(
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

			if(infFramebuf->results.size())
			{
				//draw hand landmark
				DrawHandLandmark(infFramebuf->results, &infFramebuf->frameImage);

				//collect point
				arm::app::PointHistoryPreProcess::S_POINT_COORD sPointCoord;
				sPointCoord.i32X = infFramebuf->results[HAND_POSE_KEYPOINT].m_x;
				sPointCoord.i32Y = infFramebuf->results[HAND_POSE_KEYPOINT].m_y;

				bCollectPointDone = preProcessPointHistory.CollectPoint(&sPointCoord, infFramebuf->frameImage.w, infFramebuf->frameImage.h);
			}
			else
			{
				bCollectPointDone = false;
				preProcessPointHistory.ResetPointHistory();
				s_i32PrevLabelIndex = -1;
			}

			
			if(bCollectPointDone)
			{
				//run point history model 
				preProcessPointHistory.DoPreProcess(nullptr, 0);
				pointHistoryClassifierModel.RunInference();
				postProcessPointHistory.DoPostProcess();
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

#if defined (__USE_UVC__)
			if(UVC_IsConnect())
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

            u64PerfFrames ++;

			bool bPoseDetect = false;

			if(bCollectPointDone)
			{
				bPoseDetect = JudgePoseDetect(results[0]);
				if(bPoseDetect == false)
				{
					preProcessPointHistory.ResetPointHistory();
				}
			}
			
			if ((uint64_t) pmu_get_systick_Count() > u64PerfCycle)
            {

				sprintf(szFrameRateText, "frame rate: %llu", u64PerfFrames / EACH_PERF_SEC);
				info("%s\n", szFrameRateText);

                u64PerfCycle = (uint64_t)pmu_get_systick_Count() + (uint64_t)(SystemCoreClock * EACH_PERF_SEC);
                u64PerfFrames = 0;
			}

			if(bPoseDetect)
			{
				sprintf(szPoseText, "Pose: %s(%f)", results[0].m_label.c_str(), results[0].m_normalisedVal);
				info("%s\n", szPoseText);
			}
			else
			{
				sprintf(szPoseText, " ");
			}
				
			
#if defined (__USE_DISPLAY__)

			sDispRect.u32TopLeftX = 0;
			sDispRect.u32TopLeftY = frameBuffer.h;
			sDispRect.u32BottonRightX = Disaplay_GetLCDWidth();
			sDispRect.u32BottonRightY = (frameBuffer.h + (FONT_HTIGHT * 2) - 1);
			
			Display_ClearRect(C_WHITE, &sDispRect);
			Display_PutText(
				szPoseText,
				strlen(szPoseText),
				0,
				frameBuffer.h,
				C_BLUE,
				C_WHITE,
				false
			);

			Display_PutText(
				szFrameRateText,
				strlen(szFrameRateText),
				0,
				frameBuffer.h + FONT_HTIGHT,
				C_BLUE,
				C_WHITE,
				false
			);

				
#endif

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
