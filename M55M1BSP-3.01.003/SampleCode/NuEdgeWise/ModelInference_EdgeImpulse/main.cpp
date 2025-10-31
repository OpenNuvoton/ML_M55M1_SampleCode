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

#include "ModelFileReader.h"
#include "ff.h"


#define EI_CLASSIFIER_ALLOCATION_STATIC // use static memory 
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/classifier/inferencing_engines/tflite_micro_extern.h" // need extern addr to set MPU
#include "InputFiles.hpp"

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

// Callback function declaration
static int get_signal_data(size_t offset, size_t length, float *out_ptr);

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

// Callback: fill a section of the out_ptr buffer when requested
static int get_signal_data(size_t offset, size_t length, float *out_ptr)
{
    for (size_t i = 0; i < length; i++)
    {
        out_ptr[i] = (input_buf + offset)[i];
    }

    return EIDSP_OK;
}

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

    //run_classifier_init();
#endif

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
            ARM_MPU_RLAR((((unsigned int)tensor_arena) + EI_CLASSIFIER_TFLITE_LARGEST_ARENA_SIZE - 1),        // Limit
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

    pmu_reset_counters();

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

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    printf("Object detection bounding boxes:\r\n");

    for (uint32_t i = 0; i < EI_CLASSIFIER_OBJECT_DETECTION_COUNT; i++)
    {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];

        if (bb.value == 0)
        {
            continue;
        }

        printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
               bb.label,
               bb.value,
               bb.x,
               bb.y,
               bb.width,
               bb.height);
    }

    // Print the prediction results (classification)
#else
    printf("Predictions:\r\n");

    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
    {
        printf("  %s: ", ei_classifier_inferencing_categories[i]);
        printf("%.5f\r\n", result.classification[i].value);
    }

#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = pmu_get_systick_Count();
    u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);


    while (1)
    {
        //__NOP();

        // Perform DSP pre-processing and inference
        res = run_classifier(&signal, &result, false);

        u64PerfFrames ++;

        if (pmu_get_systick_Count() > u64PerfCycle)
        {
            info("Model inference rate: %llu inf/s \n", u64PerfFrames / EACH_PERF_SEC);
            info("Accumulated time: %llu (s) \n", pmu_get_systick_Count() / SystemCoreClock);
            u64PerfCycle = pmu_get_systick_Count();
            u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);
            u64PerfFrames = 0;

            // output
            // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
            printf("Object detection bounding boxes:\r\n");

            for (uint32_t i = 0; i < EI_CLASSIFIER_OBJECT_DETECTION_COUNT; i++)
            {
                ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];

                if (bb.value == 0)
                {
                    continue;
                }

                printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                       bb.label,
                       bb.value,
                       bb.x,
                       bb.y,
                       bb.width,
                       bb.height);
            }

#else
            // Print the prediction results (classification)
            info("Predictions:\r\n");

            for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
            {
                printf("  %s: ", ei_classifier_inferencing_categories[i]);
                printf("%.5f\r\n", result.classification[i].value);
            }

#endif

            // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
            printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

        }
    }
}