/**************************************************************************//**
 * @file     InferenceTask.cpp
 * @version  V0.10
 * @brief    Inference process source code
 * * SPDX-License-Identifier: Apache-2.0
 * @copyright (C) 2022 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include "InferenceTask.hpp"

namespace InferenceProcess
{

InferenceProcess::InferenceProcess(
    Model *model)
    :   m_model(model)
{}

bool InferenceProcess::RunJob(
    object_detection::DetectorPostprocessing *pPostProc,
    int modelCols,
    int mode1Rows,
    int srcImgWidth,
    int srcImgHeight,
    std::vector<object_detection::DetectionResult> *results
)
{
//    printf("Inference process task run job...\n");

#if defined(__PROFILE__)
    uint64_t u64StartCycle;
    uint64_t u64EndCycle;

    profiler.StartProfiling("Inference");
#endif

    bool runInf = m_model->RunInference();

#if defined(__PROFILE__)
    profiler.StopProfiling();
    profiler.PrintProfilingResult();
#endif

    TfLiteTensor *modelOutput0 = m_model->GetOutputTensor(0);

#if defined(__PROFILE__)
    u64StartCycle = pmu_get_systick_Count();
#endif

    pPostProc->RunPostProcessing(
        mode1Rows,
        modelCols,
        srcImgHeight,
        srcImgWidth,
        modelOutput0,
        *results);

#if defined(__PROFILE__)
    u64EndCycle = pmu_get_systick_Count();
    printf("post processing cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

    return runInf;
}

}// namespace InferenceProcess


void inferenceProcessTask(void *pvArgs1, void *pvArgs2, void *pvArgs3)
{
    struct ProcessTaskParams params = *reinterpret_cast<struct ProcessTaskParams *>(pvArgs1);

    InferenceProcess::InferenceProcess inferenceProcess(params.model);

    for (;;)
    {
//		printf("Inference task is running...\n");
        xInferenceJob *xJob;
		k_msgq_get(params.queueHandle, &xJob, K_FOREVER);

        inferenceProcess.RunJob(
            xJob->pPostProc,
            xJob->modelCols,
            xJob->mode1Rows,
            xJob->srcImgWidth,
            xJob->srcImgHeight,
            xJob->results
        );

		k_msgq_put(xJob->responseQueue, &xJob, K_FOREVER);
    }
}




