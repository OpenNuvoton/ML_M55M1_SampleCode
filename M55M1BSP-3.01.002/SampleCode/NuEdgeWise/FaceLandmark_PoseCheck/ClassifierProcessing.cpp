/*
 * Copyright (c) 2022 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "ClassifierProcessing.hpp"
#include "ImageUtils.hpp"
#include "log_macros.h"
#include "arm_math.h"

#define __USE_ARMMVE_DSP /* Use Helium */
#define __USE_CMSIS_DSP

#if defined (__USE_ARMMVE_DSP)
    #include <arm_mve.h>
#endif

namespace arm
{
namespace app
{

ClassifierPreProcess::ClassifierPreProcess(Model *model)
{
    this->m_model = model;
}

bool ClassifierPreProcess::DoPreProcess(const void *data, size_t inputSize)
{
    if (data == nullptr)
    {
        printf_err("Data pointer is null");
    }

    auto input = static_cast<const uint8_t *>(data);
    TfLiteTensor *inputTensor = this->m_model->GetInputTensor(0);

    memcpy(inputTensor->data.data, input, inputSize);
    debug("Input tensor populated \n");

    if (this->m_model->IsDataSigned())
    {
        image::ConvertImgToInt8(inputTensor->data.data, inputTensor->bytes);
    }

    return true;
}

void ClassifierPreProcess::MinMaxNormalize_Helium(std::vector<float> &input_output)
{

    // Initialize min and max values
    int32_t maxVal = std::numeric_limits<int32_t>::min();
    int32_t minVal = std::numeric_limits<int32_t>::max();

    // Compute min and max using Helium intrinsics
    for (size_t i = 0; i < input_output.size(); i += 4)
    {
        float32x4_t  vec = vld1q_f32(input_output.data() + i); // Load 4 elements

        int32x4_t int_vec = vcvtq_s32_f32(vec);

        maxVal = vmaxvq_s32(maxVal, int_vec);           // Vectorized max
        minVal = vminvq_s32(minVal, int_vec);           // Vectorized min
    }

    float range = static_cast<float>(maxVal - minVal);

    // Avoid division by zero
    if (range == 0)
    {
        // If all elements are the same, set normalized values to mid-scale
        std::fill(input_output.begin(), input_output.end(), minVal / 2.0);
        return;
    }

    // Normalize the data
    for (size_t i = 0; i < input_output.size(); i += 4)
    {
        float32x4_t vec = vld1q_f32(input_output.data() + i);          // Load 4 elements

        vec = vsubq_f32(vec, vdupq_n_f32(static_cast<float>(minVal))); // Subtract minVal

        vec = vmulq_f32(vec, vdupq_n_f32(1.0 / range));

        vst1q_f32(input_output.data() + i, vec);
    }
}

void ClassifierPreProcess::DoPreProcess_MaxminNor(std::vector<arm::app::face_landmark::KeypointResult> &results_KP,
                                                  std::vector<float> &normalized_numbers, uint32_t KP_size)
{
#if defined(__USE_ARMMVE_DSP)
    std::vector<float> selected_numbers_X;
    std::vector<float> selected_numbers_Y;
    selected_numbers_Y.reserve(KP_size);
    selected_numbers_X.reserve(KP_size);

    for (int i = 0; i < KP_size; i ++)
    {
        selected_numbers_X.push_back(static_cast<float>(results_KP[i].m_x));
        selected_numbers_Y.push_back(static_cast<float>(results_KP[i].m_y));
    }

    MinMaxNormalize_Helium(selected_numbers_X);
    MinMaxNormalize_Helium(selected_numbers_Y);

    for (size_t i = 0; i < KP_size; i++)
    {
        normalized_numbers[i * 2] = selected_numbers_X[i];
        normalized_numbers[i * 2 + 1] = selected_numbers_Y[i];
    }

#else
    std::vector<int> selected_numbers_X;
    std::vector<int> selected_numbers_Y;
    selected_numbers_Y.reserve(KP_size);
    selected_numbers_X.reserve(KP_size);

    for (int i = 0; i < KP_size; i ++)
    {
        selected_numbers_X.push_back(results_KP[i].m_x);
        selected_numbers_Y.push_back(results_KP[i].m_y);
    }

    //printf("results_KP[10].m_x: %d", results_KP[10].m_x);
    //printf("results_KP[10].m_y: %d", results_KP[10].m_y);

    int maxValue_X;
    uint32_t maxIndex;
    int minValue_X;
    uint32_t minIndex;

    int maxValue_Y;
    int minValue_Y;

#if defined(__USE_CMSIS_DSP)
    arm_max_q31(selected_numbers_X.data(), KP_size, &maxValue_X, &maxIndex);
    arm_min_q31(selected_numbers_X.data(), KP_size, &minValue_X, &minIndex);

    arm_max_q31(selected_numbers_Y.data(), KP_size, &maxValue_Y, &maxIndex);
    arm_min_q31(selected_numbers_Y.data(), KP_size, &minValue_Y, &minIndex);
#else
    auto maxIter_X = std::max_element(selected_numbers_X.begin(), selected_numbers_X.end());
    maxValue_X = *maxIter_X;
    auto minIter_X = std::min_element(selected_numbers_X.begin(), selected_numbers_X.end());
    minValue_X = *minIter_X;

    auto maxIter_Y = std::max_element(selected_numbers_Y.begin(), selected_numbers_Y.end());
    maxValue_Y = *maxIter_Y;
    auto minIter_Y = std::min_element(selected_numbers_Y.begin(), selected_numbers_Y.end());
    minValue_Y = *minIter_Y;
#endif

    auto normalize = [](float x, float min_value, float max_value) -> float
    {
        return (x - min_value) / (max_value - min_value);
    };

    for (size_t i = 0; i < KP_size; i++)
    {
        normalized_numbers[i * 2] = normalize(static_cast<float>(selected_numbers_X[i]),
                                              static_cast<float>(minValue_X), static_cast<float>(maxValue_X));
        normalized_numbers[i * 2 + 1] = normalize(static_cast<float>(selected_numbers_Y[i]),
                                                  static_cast<float>(minValue_Y), static_cast<float>(maxValue_Y));
    }

#endif


}

void ClassifierPreProcess::DoPreProcess_ForeheadNor(std::vector<arm::app::face_landmark::KeypointResult> &results_KP,
                                                    std::vector<float> &normalized_numbers, uint32_t KP_size)
{
    int forehead_x = results_KP[0].m_x;
    int forehead_y = results_KP[0].m_y;
    float img_scale_x = 192.0;
    float img_scale_y = 192.0;

    for (size_t idx = 0; idx < KP_size; ++idx)
    {
        // Normalize x-coordinate
        normalized_numbers[idx * 2] = ((results_KP[idx].m_x - forehead_x) / img_scale_x);
        // Normalize y-coordinate
        normalized_numbers[idx * 2 + 1] = ((results_KP[idx].m_y - forehead_y) / img_scale_y);
    }

}

ClassifierPostProcess::ClassifierPostProcess(Classifier &classifier, Model *model,
                                             const std::vector<std::string> &labels
                                            )
    : m_Classifier{classifier},
      m_labels{labels}
{
    this->m_model = model;
}

bool ClassifierPostProcess::DoPostProcess()
{
    return this->m_Classifier.GetClassificationResults(
               this->m_model->GetOutputTensor(0), this->m_results,
               this->m_labels, 1, false);
}

ClassificationResult ClassifierPostProcess::getResults()
{
    return this->m_results[0];
}

std::vector<ClassificationResult> ClassifierPostProcess::getTopkResults()
{
    return this->m_results;
}

} /* namespace app */
} /* namespace arm */