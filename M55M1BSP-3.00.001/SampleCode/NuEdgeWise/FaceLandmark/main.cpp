/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    face landmark network sample. Demonstrate face landmark detect.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include "BoardInit.hpp"      /* Board initialisation */
#include "log_macros.h"      /* Logging macros (optional) */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "FaceLandmarkModel.hpp"       /* Model API */
#include "FaceDetectionModel.hpp"       /* Model API */
#include "FaceLandmarkPostProcessing.hpp"
#include "FaceDetectorPostProcessing.hpp"

#include "imlib.h"          /* Image processing */
#include "framebuffer.h"
#include "ModelFileReader.h"
#include "ff.h"

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

//#define __PROFILE__
#define __USE_DISPLAY__
//#define __USE_UVC__

#if defined (FACE_LANDMARK_ATTENTION_MODEL)
#define __LOAD_MODEL_FROM_SD__
#endif

#include "Profiler.hpp"

#include "ImageSensor.h"

#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

#define NUM_FRAMEBUF 2  //1 or 2
#define MODEL_AT_HYPERRAM_ADDR (0x82400000)
#define FACE_PRESENCE_THRESHOLD  				(0.4)

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
    std::vector<arm::app::face_landmark::KeypointResult> results_KP;
    std::vector<arm::app::face_detection::DetectionResult> results_FD;	
} S_FRAMEBUF;


S_FRAMEBUF s_asFramebuf[NUM_FRAMEBUF];

namespace arm
{
namespace app
{
/* Tensor arena buffer */
#undef ACTIVATION_BUF_SZ
#if defined(FACE_LANDMARK_ATTENTION_MODEL)
#define FACE_LANDMARK_ACTIVATION_BUF_SZ	 (500000)
#else
#define FACE_LANDMARK_ACTIVATION_BUF_SZ	 (460000)
#endif
#define FACE_DETECTION_ACTIVATION_BUF_SZ (460000)
static uint8_t tensorArena_FaceLandmark[FACE_LANDMARK_ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
static uint8_t tensorArena_FaceDetection[FACE_DETECTION_ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

#if !defined(__LOAD_MODEL_FROM_SD__)
/* Optional getter function for the model pointer and its size. */
namespace face_landmark
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace face_landmark */

namespace face_detection
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace face_detection */

#endif

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
#define GLCD_WIDTH	240
#define GLCD_HEIGHT	240
#endif

#define IMAGE_FB_SIZE	(GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE (IMAGE_FB_SIZE + 1024)

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

static void DrawFaceLandmark(
    const std::vector<arm::app::face_landmark::KeypointResult> &results,
	int posOffsetX,
	int posOffsetY,		
    image_t *drawImg
)
{
	int i;
	arm::app::face_landmark::KeypointResult keyPoint;
	int keypointSize = results.size();
	
	for(i = 0; i < keypointSize; i ++)
	{
		keyPoint = results[i];
		//draw points
		if(i < 468)
		{
			imlib_draw_circle(drawImg, posOffsetX + keyPoint.m_x, posOffsetY + keyPoint.m_y, 1, COLOR_R5_G6_B5_TO_RGB565(0, COLOR_G6_MAX, 0), 1, true);	
		}
		else
		{	
			imlib_draw_circle(drawImg, posOffsetX + keyPoint.m_x, posOffsetY + keyPoint.m_y, 1, COLOR_R5_G6_B5_TO_RGB565(0, 0, COLOR_B5_MAX), 1, true);	
		}
	}
}

static void DrawDetectFace(
    std::vector<arm::app::face_detection::DetectionResult> &results,
    image_t *drawImg
)
{
	arm::app::face_detection::DetectionResult faceBox;
	int faceBoxSize = results.size();
	
	for(int i = 0; i < faceBoxSize; i ++)
	{
		faceBox = results[i];
		imlib_draw_rectangle(drawImg, faceBox.m_x0, faceBox.m_y0, faceBox.m_w, faceBox.m_h, COLOR_B5_MAX, 2, false);
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

	if(profiler)
		u64StartCycle = pmu_get_systick_Count();

	imlib_nvt_scale(&infFramebuf->frameImage, &resizeImg, &roi);

	if(profiler){
		u64EndCycle = pmu_get_systick_Count();
		info("face detect resize cycles %llu \n", (u64EndCycle - u64StartCycle));
	}

	if(profiler){
		u64StartCycle = pmu_get_systick_Count();
	}
		
    /* face landmark/detection model preprocessing is image conversion from uint8 to [0,1] float values,
     * then quantize them with input quantization info. */
	for (size_t i = 0; i < inputTensor->bytes; i++)
	{
//		auto i_data_int8 = static_cast<int8_t>(((static_cast<float>(req_data[i]) / 255.0f) / inQuantParams.scale) + inQuantParams.offset);
//		signed_req_data[i] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
		signed_req_data[i] = static_cast<int8_t>(req_data[i]) - 128;
	}

	if(profiler){
		u64EndCycle = pmu_get_systick_Count();
		info("face detect quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
	}

	if(profiler){
		profiler->StartProfiling("Inference");
	}

	faceDetectionModel->RunInference();

	if(profiler){
		profiler->StopProfiling();
		profiler->PrintProfilingResult();
	}

	if(profiler){
		u64StartCycle = pmu_get_systick_Count();
	}
	
	postProcess->RunPostProcess(infFramebuf->results_FD);

	if(profiler){
		u64EndCycle = pmu_get_systick_Count();
		info("face detect post processing cycles %llu \n", (u64EndCycle - u64StartCycle));
	}

	float scaleFactoryW = 1.4;
	float scaleFactoryH = 1.4;
	arm::app::face_detection::DetectionResult *faceBox;
	//fine tune face region
	for(int i = 0 ; i < infFramebuf->results_FD.size(); i ++)
	{
		float scaleW;
		float scaleH;
		int newX;
		int newY;
		
		int newW;
		int newH;

		faceBox = &(infFramebuf->results_FD[i]);
	
		scaleH = scaleFactoryH * faceBox->m_h;
//		scaleW = scaleFactoryW * faceBox->m_w;
		scaleW = scaleH;
		newW = scaleW;
		newH = scaleH;

		newX = faceBox->m_x0 - ((scaleW - faceBox->m_w) / 2);
		newY = faceBox->m_y0 - ((scaleH - faceBox->m_h) / 2);
		
		if(newX < 0)
			newX = 0;
		
		if(newY < 0)
			newY = 0;

		if(newX + newW >= infFramebuf->frameImage.w)
			newW = infFramebuf->frameImage.w - newX;
		
		if(newY + newH >= infFramebuf->frameImage.h)
			newH = infFramebuf->frameImage.h - newY;

		faceBox->m_x0 = newX;
		faceBox->m_y0 = newY;
		faceBox->m_w = newW;
		faceBox->m_h = newH;
	}
}

static void DetectFaceLandmark_DrawResult(
    S_FRAMEBUF *infFramebuf,
    arm::app::FaceLandmarkModel *faceLandmarkModel,	
	arm::app::face_landmark::FaceLandmarkPostProcessing *postProcess,
	arm::app::Profiler *profiler
)
{
	int i;
	arm::app::face_detection::DetectionResult faceBox;
    rectangle_t roi;
    TfLiteIntArray *inputShape = faceLandmarkModel->GetInputShape(0);
    TfLiteTensor *inputTensor   = faceLandmarkModel->GetInputTensor(0);

    const int inputImgCols = inputShape->data[arm::app::FaceLandmarkModel::ms_inputColsIdx];
    const int inputImgRows = inputShape->data[arm::app::FaceLandmarkModel::ms_inputRowsIdx];

	uint64_t u64StartCycle;
	uint64_t u64EndCycle;

	//Quantize input tensor data
	auto *req_data = static_cast<uint8_t *>(inputTensor->data.data);
	auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);

    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);

	TfLiteTensor *modelOutput0 = faceLandmarkModel->GetOutputTensor(FACE_LANDMARK_MESH_TENSOR_INDEX);

	#if defined(FACE_LANDMARK_LEFT_IRIS_TENSOR_INDEX)
		TfLiteTensor *modelOutput1 = faceLandmarkModel->GetOutputTensor(FACE_LANDMARK_LEFT_IRIS_TENSOR_INDEX);
	#else
		TfLiteTensor *modelOutput1 = NULL;
	#endif

	#if defined(FACE_LANDMARK_RIGHT_IRIS_TENSOR_INDEX)
		TfLiteTensor *modelOutput2 = faceLandmarkModel->GetOutputTensor(FACE_LANDMARK_RIGHT_IRIS_TENSOR_INDEX);			
	#else
		TfLiteTensor *modelOutput2 = NULL;
	#endif

	TfLiteTensor *modelOutput3 = faceLandmarkModel->GetOutputTensor(FACE_LANDMARK_FACE_FLAG_TENSOR_INDEX);
	
	for(i = 0 ; i < infFramebuf->results_FD.size(); i ++)
	{
		faceBox = infFramebuf->results_FD[i];

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

		if(profiler)
			u64StartCycle = pmu_get_systick_Count();

        imlib_nvt_scale(&infFramebuf->frameImage, &resizeImg, &roi);

		if(profiler){
			u64EndCycle = pmu_get_systick_Count();
			info("face landmark resize cycles %llu \n", (u64EndCycle - u64StartCycle));
		}
			
		if(profiler){
			u64StartCycle = pmu_get_systick_Count();
		}
			
		/* face landmark/detection model preprocessing is image conversion from uint8 to [0,1] float values,
		* then quantize them with input quantization info. */
		for (size_t i = 0; i < inputTensor->bytes; i++)
		{
//			auto i_data_int8 = static_cast<int8_t>(((static_cast<float>(req_data[i]) / 255.0f) / inQuantParams.scale) + inQuantParams.offset);
//			signed_req_data[i] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
			signed_req_data[i] = static_cast<int8_t>(req_data[i]) - 128;
		}

		if(profiler){
			u64EndCycle = pmu_get_systick_Count();
			info("face landmark quantize cycles %llu \n", (u64EndCycle - u64StartCycle));
		}

		if(profiler){
			profiler->StartProfiling("Inference");
		}

		faceLandmarkModel->RunInference();

		if(profiler){
			profiler->StopProfiling();
			profiler->PrintProfilingResult();
		}

		if(profiler){
			u64StartCycle = pmu_get_systick_Count();
		}

		postProcess->RunPostProcessing(
			inputImgCols,
			inputImgRows,
			roi.w,
			roi.h,
			modelOutput0,
			modelOutput1,
			modelOutput2,
			modelOutput3,
			infFramebuf->results_KP);

		if(profiler){
			u64EndCycle = pmu_get_systick_Count();
			info("face landmark post processing cycles %llu \n", (u64EndCycle - u64StartCycle));
		}

		//Draw face landmark keypoint
		if(infFramebuf->results_KP.size())
		{
			if(profiler){
				u64StartCycle = pmu_get_systick_Count();
			}

			DrawFaceLandmark(infFramebuf->results_KP, roi.x, roi.y, &infFramebuf->frameImage);

			if(profiler){
				u64EndCycle = pmu_get_systick_Count();
				info("draw face landmark cycles %llu \n", (u64EndCycle - u64StartCycle));
			}
		}
	}

	if(profiler){
		u64StartCycle = pmu_get_systick_Count();
	}

	DrawDetectFace(infFramebuf->results_FD, &infFramebuf->frameImage);

	if(profiler){
		u64EndCycle = pmu_get_systick_Count();
		info("draw face region cycles %llu \n", (u64EndCycle - u64StartCycle));
	}
}

static int32_t PrepareModelToHyperRAM(void)
{
#define MODEL_FILE "0:\\face_landmark.tflite"
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

int main()
{
    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

#if defined(__LOAD_MODEL_FROM_SD__)

	/* Copy model file from SD to HyperRAM*/
	int32_t i32ModelSize;

	i32ModelSize = PrepareModelToHyperRAM();
	
	if(i32ModelSize <= 0 )
	{
        printf_err("Failed to prepare model\n");
        return 1;
	}

    /* Model object creation and initialisation. */
    arm::app::FaceLandmarkModel faceLandmarkModel;

    if (!faceLandmarkModel.Init(arm::app::tensorArena_FaceLandmark,
                    sizeof(arm::app::tensorArena_FaceLandmark),
                    (unsigned char *)MODEL_AT_HYPERRAM_ADDR,
                    i32ModelSize))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }

#else
    /* Model object creation and initialisation. */
    arm::app::FaceLandmarkModel faceLandmarkModel;

    if (!faceLandmarkModel.Init(arm::app::tensorArena_FaceLandmark,
                    sizeof(arm::app::tensorArena_FaceLandmark),
					arm::app::face_landmark::GetModelPointer(),
                    arm::app::face_landmark::GetModelLen()))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }
	
#endif

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



    /* Setup cache poicy of tensor arean buffer */
    info("Set tesnor arena cache policy to WTRA \n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena_FaceLandmark),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena_FaceLandmark) + FACE_LANDMARK_ACTIVATION_BUF_SZ - 1),        // Limit
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

    TfLiteTensor* outputTensor0_FD = faceDetectionModel.GetOutputTensor(0);
    TfLiteTensor* outputTensor1_FD = faceDetectionModel.GetOutputTensor(1);
	
    //display framebuffer
    image_t frameBuffer;
    rectangle_t roi;

    //omv library init
    omv_init();
    framebuffer_init_image(&frameBuffer);

    // postProcess
    arm::app::face_landmark::FaceLandmarkPostProcessing postProcess_FL(FACE_PRESENCE_THRESHOLD);

    const arm::app::face_detection::PostProcessParams postProcessParams{
            inputImgRows_FD,
            inputImgCols_FD,
            (int)s_asFramebuf[0].frameImage.h,
            (int)s_asFramebuf[0].frameImage.w,
            anchor1,
            anchor2};

	arm::app::FaceDetectorPostProcess postProcess_FD =
            arm::app::FaceDetectorPostProcess(outputTensor0_FD, outputTensor1_FD, s_asFramebuf[0].results_FD, postProcessParams);

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

#if defined (__USE_UVC__)
	UVC_Init();
    HSUSBD_Start();
#endif

	bool bDoFaceLandmark = false;

    while(1)
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
			if(infFramebuf->results_FD.size())
				DetectFaceLandmark_DrawResult(
						infFramebuf,
						&faceLandmarkModel,
						&postProcess_FL,
						&profiler);
#else
			if(infFramebuf->results_FD.size())
				DetectFaceLandmark_DrawResult(
						infFramebuf,
						&faceLandmarkModel,
						&postProcess_FL,
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
