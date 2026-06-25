/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    Battery SOH estimation sample. Demonstrate network infereence
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2026 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include "BoardInit.hpp"      /* Board initialisation */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "NNModel.hpp"       /* Model API */

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

#include "ModelFileReader.h"
#include "ff.h"

#include "Pattern/SOH_test_data.h"

//#define LOG_LEVEL_TRACE       0
//#define LOG_LEVEL_DEBUG       1
//#define LOG_LEVEL_INFO        2
//#define LOG_LEVEL_WARN        3
//#define LOG_LEVEL_ERROR       4

#define LOG_LEVEL             2
#include "log_macros.h"      /* Logging macros (optional) */

//#define __PROFILE__

#if !defined(__WITHOUT_HYPERRAM__)
#define __LOAD_MODEL_FROM_SD__
#endif

#undef ACTIVATION_BUF_SZ
#define ACTIVATION_BUF_SZ (219136)

#define MODEL_AT_HYPERRAM_ADDR 0x82400000

#include "Profiler.hpp"

namespace arm
{
namespace app
{
/* Tensor arena buffer */
static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

/* Optional getter function for the model pointer and its size. */
namespace nn
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace nn */
} /* namespace app */
} /* namespace arm */

static int32_t PrepareModelToHyperRAM(void)
{
#if !defined(__USE_SD1__)
#define MODEL_FILE "0:\\nn_model.tflite"
#else
#define MODEL_FILE "1:\\nn_model.tflite"
#endif

#define EACH_READ_SIZE 512

#if !defined(__USE_SD1__)
    TCHAR sd_path[] = { '0', ':', 0 };    /* SD drive started from 0 */
#else
    TCHAR sd_path[] = { '1', ':', 0 };    /* SD drive started from 0 */
#endif

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

    printf("==================== Load model file from SD card =================================\n");
    printf("Please copy NN_ModelInference/Model/xxx_vela.tflite to SDCard:/nn_model.tflite     \n");
    printf("===================================================================================\n");
    i32ModelSize = PrepareModelToHyperRAM();

    if(i32ModelSize <= 0 )
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
    arm::app::NNModel model;

    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::nn::GetModelPointer(),
                    arm::app::nn::GetModelLen()))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }
#endif

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

    size_t numOutput = model.GetNumOutputs();

    TfLiteTensor *inputTensor   = model.GetInputTensor(0);
    TfLiteTensor *outputTensor = model.GetOutputTensor(0);

    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);

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

    float *pfInputTestData = (float *)test_x;
    float *pfOutputTestData = (float *)test_y;

    //Sampling data during charge cycle * (Voltage, Current, Temperature)
    int32_t i32InputDataSize = test_x_dim[1] * test_x_dim[2];
    int32_t i32InputDataIndex = 0;
    int32_t i32InputDataItems = test_x_dim[2];
    int32_t i32OutputDataIndex = 0;
    int32_t i32Cycle = 0;
    int32_t i32ToatlCycles = test_x_dim[0];
    //Summation of absolute error
    float fPredicAbsErrorSum = 0.;
    float fInputData;

    std::vector<float> vNormScale;

    for(int v= 0; v < i32InputDataItems; v++)
    {
        vNormScale.push_back(normalize_scale_max[v] - normalize_scale_min[v]);
    }

    while (i32Cycle < i32ToatlCycles)
    {
        //Quantize input tensor data.
        auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);
        for (size_t i = 0; i < inputTensor->bytes; i++)
        {
            fInputData = pfInputTestData[i32InputDataIndex + i];
            //If the input data is raw data, perform the normalize step.
            fInputData = (fInputData - normalize_scale_min[i % i32InputDataItems]) / (vNormScale[i % i32InputDataItems]);

            auto i_data_int8 = static_cast<int8_t>(((fInputData) / inQuantParams.scale) + inQuantParams.offset);
            signed_req_data[i] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
        }

        i32InputDataIndex += i32InputDataSize;

#if defined(__PROFILE__)
        profiler.StartProfiling("Inference");
#endif

        if (!model.RunInference())
        {
            printf_err("Inference failed.");
            return 5;
        }

#if defined(__PROFILE__)
        profiler.StopProfiling();
#endif

#if defined(__PROFILE__)
        profiler.PrintProfilingResult();
#endif

        u64PerfFrames ++;

        if (pmu_get_systick_Count() > u64PerfCycle)
        {
            info("Model inference rate: %llu inf/s \n", u64PerfFrames / EACH_PERF_SEC);

            u64PerfCycle = pmu_get_systick_Count();
            u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);
            u64PerfFrames = 0;
        }

        //Dequantize output tensor data
        for(int i = 0; i < numOutput; i ++)
        {
            outputTensor = model.GetOutputTensor(i);
            arm::app::QuantParams outputQuantParams = arm::app::GetTensorQuantParams(outputTensor);
            int8_t *tensorOutputData = outputTensor->data.int8;
            float fPredicData;

            for(int j =0; j < outputTensor->bytes; j ++)
            {
                fPredicData = outputQuantParams.scale * static_cast<float>(tensorOutputData[i] - outputQuantParams.offset);
                fPredicAbsErrorSum += std::abs(pfOutputTestData[i32OutputDataIndex] - fPredicData);
                i32OutputDataIndex ++;
                info("Tensor output[%d][%d]: %f\n",i,j, fPredicData);
            }
        }

        i32Cycle ++;
    }

    info("Prediction MAE %f\n",  fPredicAbsErrorSum / i32OutputDataIndex);
}