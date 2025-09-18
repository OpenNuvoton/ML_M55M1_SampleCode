/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    network inference sample with Edge Impulse SDK. Demonstrate network infereence
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2025 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "BoardInit.hpp"      /* Board initialisation */

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "export_tensor_arena.h"
#include "InputFiles.hpp"

//#define LOG_LEVEL_TRACE       0
//#define LOG_LEVEL_DEBUG       1
//#define LOG_LEVEL_INFO        2
//#define LOG_LEVEL_WARN        3
//#define LOG_LEVEL_ERROR       4

#define LOG_LEVEL             2
#include "log_macros.h"      /* Logging macros (optional) */

//#define __PROFILE__
//#define USE_PERF_CALIB  // Use EI's Selected config or not
#define USE_DMIC
#include "DMICRecord.h"
#include "usbd_audio.h"

#include "Profiler.hpp"

#if defined(USE_DMIC)

static int16_t audioSlidingBuf[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];
static int get_signal_data_from_dmic(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&audioSlidingBuf[offset], out_ptr, length);

    return 0;
}
#else
// Callback: fill a section of the out_ptr buffer when requested
static int get_signal_data(size_t offset, size_t length, float *out_ptr)
{
    for (size_t i = 0; i < length; i++)
    {
        out_ptr[i] = (input_buf + offset)[i];
    }

    return EIDSP_OK;
}
#endif

int main()
{

    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

    /* Model object creation and initialisation. */
    // summary of inferencing settings (from model_metadata.h)
    printf("Edge Impulse Inferencing settings:\n");
    printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    /* Setup cache poicy of tensor arean buffer */
    info("Set tesnor arena cache policy to WTRA \n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)tensor_arena),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)tensor_arena) + 66208 - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - Write-Through, Read-allocate
        },
    };

    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());
		

    //Edge Impulse structure initial
    signal_t signal;            // Wrapper for raw input buffer
    ei_impulse_result_t result = {0}; // Used to store inference output
    EI_IMPULSE_ERROR res;       // Return code from inference

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

#if defined(__PROFILE__)
    arm::app::Profiler profiler;
    uint64_t u64StartCycle;
    uint64_t u64EndCycle;
    uint64_t u64CCAPStartCycle;
    uint64_t u64CCAPEndCycle;
#else
    pmu_reset_counters();
#endif		


#if defined(USE_DMIC)
		// Assign callback function to fill buffer used for preprocessing/inference
		signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &get_signal_data_from_dmic;
		
#if defined(USE_PERF_CALIB)			
		run_classifier_init();
#endif		
		
#define AUDIO_SAMPLE_BLOCK  4		
#define AUDIO_CHANNEL       1
const auto audioSlidingSamples = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;		

		int32_t ret;
		ret = DMICRecord_Init(EI_CLASSIFIER_FREQUENCY, AUDIO_CHANNEL, audioSlidingSamples, AUDIO_SAMPLE_BLOCK);
    if (ret)
    {
        printf_err("Unable init DMIC record error(%d)\n", ret);
        return 4;
    }		
		
#else
		// Assign callback function to fill buffer used for preprocessing/inference
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &get_signal_data;
    //numpy::signal_from_buffer(&input_buf[0], EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    
		// Perform DSP pre-processing and inference		
    printf("\r\n");
    printf("run_classifier\r\n");
    res = run_classifier(&signal, &result, false);
    if (res != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", res);
        return 1;
    }
		
    // Print return code and how long it took to perform inference
    printf("run_classifier returned: %d\r\n", res);
    printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
           result.timing.dsp,
           result.timing.classification,
           result.timing.anomaly);
		
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
		
#endif	// defined(USE_DMIC)	

#define EACH_PERF_SEC 1
    uint64_t u64PerfCycle;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = pmu_get_systick_Count();
    u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);


    while (1)
    {

#if defined(USE_DMIC)
			
			  DMICRecord_ReadSamples(audioSlidingBuf, audioSlidingSamples);

        DMICRecord_UpdateReadSampleIndex(audioSlidingSamples);
			
			  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
        signal.get_data = &get_signal_data_from_dmic;
			
#if defined(__PROFILE__)
			  u64StartCycle = pmu_get_systick_Count();
        profiler.StartProfiling("Inference");
#endif			
			
#if defined(USE_PERF_CALIB)			
			  res = run_classifier_continuous(&signal, &result, false);
#else			
			  res = run_classifier(&signal, &result, false);
#endif		
			
#else			
        // Perform DSP pre-processing and inference
        res = run_classifier(&signal, &result, false);
#endif

#if defined(__PROFILE__)
        profiler.StopProfiling();
#endif
        
			  if (res != EI_IMPULSE_OK) {
            ei_printf("ERR: Failed to run classifier (%d)\n", res);
            return 1;
        }

        u64PerfFrames ++;

        if (pmu_get_systick_Count() > u64PerfCycle)
        {
            info("Model inference rate: %llu inf/s \n", u64PerfFrames / EACH_PERF_SEC);
            info("Accumulated time: %llu (s) \n", pmu_get_systick_Count() / SystemCoreClock);
            u64PerfCycle = pmu_get_systick_Count();
            u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);
            u64PerfFrames = 0;

            // output
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
						
#if defined(__PROFILE__)
            profiler.PrintProfilingResult();
#endif						

        }
    }
}