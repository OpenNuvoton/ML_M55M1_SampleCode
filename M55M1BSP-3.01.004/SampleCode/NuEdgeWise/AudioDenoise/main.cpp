/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    RNNoise network sample. Demonstrate audio denoise
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include "BoardInit.hpp"      	/* Board initialisation */
#include "log_macros.h"      	/* Logging macros (optional) */
#include "Profiler.hpp"
#include "BufAttributes.hpp" 	/* Buffer attributes to be applied */

#include "RNNoiseModel.hpp"
#include "RNNoiseFeatureProcessor.hpp"
#include "RNNoiseProcessing.hpp"
#include "AudioCodec.h"

#include "AudioUtils.hpp"
#include "ff.h"

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

//#define __PROFILE__
//#define __WAV_FILE_REC__

#if defined(__WAV_FILE_REC__)
#include "WavFileUtil.h"
#endif


#define TEST_SAMPLERATE			eAUDIOCODEC_SAMPLERATE_16000
#define TEST_BLOCK_COUNT		(8)
#define TEST_CHANNELS			eAUDIOCODEC_CHANNEL_MONO
#define TEST_BLOCK_SAMPLES		(RNNOISE_FRAME_SIZE)	

#define WAV_REC_SOURCE			"0:\\source.wav"
#define WAV_REC_DENOISE			"0:\\denoise.wav"
#define WAV_REC_SEC				5

static int16_t s_i16AudioSampleBlock[TEST_BLOCK_SAMPLES * TEST_CHANNELS];
static bool s_bRunDeNoise = true;
static bool s_bStartRecWav = false;

namespace arm
{
namespace app
{
/* Tensor arena buffer */
static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
	
/* Optional getter function for the model pointer and its size. */
namespace rnnoise
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace rnnoise */

} /* namespace app */
} /* namespace arm */

#ifdef __cplusplus
extern "C" {
#endif

/* Using PH.1(BTN1) as Input mode and enable interrupt by rising edge trigger. In order to switch denoise or not*/

void GPH_IRQHandler(void)
{
    volatile uint32_t temp;

    /* To check if PB.4 interrupt occurred */
    if (GPIO_GET_INT_FLAG(PH, BIT1))
    {
        GPIO_CLR_INT_FLAG(PH, BIT1);
#if !defined(__WAV_FILE_REC__)
		if(s_bRunDeNoise)
		{
			printf("Disable denoise\n");
			s_bRunDeNoise = false;
		}
		else
		{
			printf("Enable denoise\n");
			s_bRunDeNoise = true;
		}
#else
		s_bStartRecWav = true;
#endif
	}
    else
    {
        /* Un-expected interrupt. Just clear all PB interrupts */
        temp = PH->INTSRC;
        PH->INTSRC = temp;
        printf("Un-expected interrupts.\n");
    }
}

#ifdef __cplusplus
}
#endif


int main()
{
	S_AUDIOCODEC_RES *psAudioCodecRes = NULL;
    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

    info("*************************************************************** \n");
    info("* Audio denoise live demo.                                    * \n"); 
	info("* Using PH.1(BTN1) pin to switch denoise or not.              * \n");
    info("*************************************************************** \n");
	

#if defined(__WAV_FILE_REC__)

    TCHAR sdPath[] = { '0', ':', 0 };    /* SD drive started from 0 */	
    f_chdrive(sdPath);          /* set default path */

	S_WavFileWriteInfo sSourceWavFileInfo;
	S_WavFileWriteInfo sSDenoiseWavFileInfo;

	if(!WavFileUtil_Write_Initialize(&sSourceWavFileInfo, WAV_REC_SOURCE))
	{
        printf_err("Unable create source wav file %s\n", WAV_REC_SOURCE);
		return -1;
	}
	
	WavFileUtil_Write_SetFormat(
		&sSourceWavFileInfo,
		eWAVE_FORMAT_PCM,
		TEST_CHANNELS,
		TEST_SAMPLERATE,
		16
	);

	if(!WavFileUtil_Write_Initialize(&sSDenoiseWavFileInfo, WAV_REC_DENOISE))
	{
        printf_err("Unable create source wav file %s\n", WAV_REC_DENOISE);
		return -1;
	}

	
	WavFileUtil_Write_SetFormat(
		&sSDenoiseWavFileInfo,
		eWAVE_FORMAT_PCM,
		TEST_CHANNELS,
		TEST_SAMPLERATE,
		16
	);

#endif
	
    /* Model object creation and initialisation. */
    arm::app::RNNoiseModel rnnNoiseModel;

    if (!rnnNoiseModel.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
					arm::app::rnnoise::GetModelPointer(),
                    arm::app::rnnoise::GetModelLen()))
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
    };

    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

#if defined(__PROFILE__)
    arm::app::Profiler profiler;
    uint64_t u64StartCycle;
    uint64_t u64EndCycle;
#endif

    pmu_reset_counters();

	size_t audioFrameLen = RNNOISE_FRAME_SIZE;
	size_t audioFrameStride = RNNOISE_FRAME_STRIDE;
	
	TfLiteTensor* inputTensor = rnnNoiseModel.GetInputTensor(0);
	TfLiteTensor* outputTensor = rnnNoiseModel.GetOutputTensor(rnnNoiseModel.m_indexForModelOutput);

	/* Set up pre and post-processing. */
	std::shared_ptr<arm::app::rnn::RNNoiseFeatureProcessor> featureProcessor =
	std::make_shared<arm::app::rnn::RNNoiseFeatureProcessor>();
	std::shared_ptr<arm::app::rnn::FrameFeatures> frameFeatures =
	std::make_shared<arm::app::rnn::FrameFeatures>();

    arm::app::RNNoisePreProcess preProcess = arm::app::RNNoisePreProcess(inputTensor, featureProcessor, frameFeatures);

	std::vector<int16_t> denoisedAudioFrame(audioFrameLen);
	arm::app::RNNoisePostProcess postProcess = arm::app::RNNoisePostProcess(outputTensor, denoisedAudioFrame,
			featureProcessor, frameFeatures);

	psAudioCodecRes = AudioCodec_Init(
		TEST_SAMPLERATE,
		TEST_CHANNELS,
		TEST_BLOCK_SAMPLES,
		TEST_BLOCK_COUNT,
		true,
		true
	);

	if(psAudioCodecRes == NULL)
	{
		printf("Unable create audio codec resource \n");
		return -1;
	}

	bool resetGRU = true;
    uint64_t u64EndRecCycle = 0;

	while(1)
	{
		if(AudioCodec_RecvPCMBlockData(psAudioCodecRes, s_i16AudioSampleBlock) > 0)
		{

#if defined(__WAV_FILE_REC__)
			if(s_bStartRecWav)
			{
				if(u64EndRecCycle == 0)
				{
					u64EndRecCycle = pmu_get_systick_Count();
				    u64EndRecCycle += (SystemCoreClock * WAV_REC_SEC);
				}

				//terminate program if record timeout
				if((uint64_t)pmu_get_systick_Count() > u64EndRecCycle)
					break;

				//SCB_CleanDCache_by_Addr((void *)s_i16AudioSampleBlock, sizeof(s_i16AudioSampleBlock));
				WavFileUtil_Write_WriteData(
					&sSourceWavFileInfo,
					(const BYTE *) s_i16AudioSampleBlock,
					sizeof(s_i16AudioSampleBlock));

			}
#endif			

			if(s_bRunDeNoise)
			{
				/* Creating a sliding window through the audio. */
				auto audioDataSlider = arm::app::audio::SlidingWindow<const int16_t>(
							s_i16AudioSampleBlock,
							TEST_BLOCK_SAMPLES, audioFrameLen,
							audioFrameStride);
				int i = 0;
				while (audioDataSlider.HasNext()) {
					const int16_t* inferenceWindow = audioDataSlider.Next();

#if defined(__PROFILE__)
					u64StartCycle = pmu_get_systick_Count();
#endif

					if (!preProcess.DoPreProcess(inferenceWindow, audioFrameLen)) {
						printf_err("Pre-processing failed.");
						break;
					}

#if defined(__PROFILE__)
					u64EndCycle = pmu_get_systick_Count();
					info("pre process cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif
					
					/* Reset or copy over GRU states first to avoid TFLu memory overlap issues. */
					if (resetGRU){
						rnnNoiseModel.ResetGruState();
					} else {
						/* Copying gru state outputs to gru state inputs.
						 * Call ResetGruState in between the sequence of inferences on unrelated input data. */
						rnnNoiseModel.CopyGruStates();
					}

#if defined(__PROFILE__)
					profiler.StartProfiling("Inference");
#endif

					/* Run inference over this feature sliding window. */
					if (!rnnNoiseModel.RunInference()) {
						printf_err("Inference failed.");
						break;
					}

#if defined(__PROFILE__)
					profiler.StopProfiling();
					profiler.PrintProfilingResult();
#endif

					resetGRU = false;

#if defined(__PROFILE__)
					u64StartCycle = pmu_get_systick_Count();
#endif
					/* Carry out post-processing. */
					if (!postProcess.DoPostProcess()) {
						printf_err("Post-processing failed.");
						break;
					}

#if defined(__PROFILE__)
					u64EndCycle = pmu_get_systick_Count();
					info("post process cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

					for(int j = 0; j < audioFrameLen; j++)
					{
						s_i16AudioSampleBlock[i] = denoisedAudioFrame[j];
						i ++;
					}
				}
			}
			else
			{
				resetGRU = true;
			}

			if(AudioCodec_SendPCMBlockData(psAudioCodecRes, s_i16AudioSampleBlock) <= 0)
			{
				printf("Unable send audio codec resource \n");
			}

#if defined(__WAV_FILE_REC__)

			if(s_bStartRecWav)
			{
				//SCB_CleanDCache_by_Addr((void *)s_i16AudioSampleBlock, sizeof(s_i16AudioSampleBlock));
				WavFileUtil_Write_WriteData(
					&sSDenoiseWavFileInfo,
					(const BYTE *) s_i16AudioSampleBlock,
					sizeof(s_i16AudioSampleBlock));
			}
#endif			

		}
	}

#if defined(__WAV_FILE_REC__)
	WavFileUtil_Write_Finish(&sSourceWavFileInfo, WAV_REC_SOURCE); 
	
	WavFileUtil_Write_Finish(&sSDenoiseWavFileInfo, WAV_REC_DENOISE); 
#endif
	SDH_Close_Disk(SDH0);

	AudioCodec_UnInit(psAudioCodecRes);
	printf("Program done \n");
	return 0;
}
