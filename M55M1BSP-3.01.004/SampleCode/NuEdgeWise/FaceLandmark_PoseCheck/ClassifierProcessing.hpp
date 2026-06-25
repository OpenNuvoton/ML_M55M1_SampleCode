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
#ifndef CLASS_PROCESSING_HPP
#define CLASS_PROCESSING_HPP

#include "BaseProcessing.hpp"
#include "Model.hpp"
#include "Classifier.hpp"
#include "KeypointResult.hpp"

namespace arm
{
namespace app
{

/**
 * @brief   Pre-processing class for Image Classification use case.
 *          Implements methods declared by BasePreProcess and anything else needed
 *          to populate input tensors ready for inference.
    *           Nuvoton adding:  DoPreProcess_MaxminNor and MinMaxNormalize_Helium.
 */
class ClassifierPreProcess : public BasePreProcess
{

public:
    explicit ClassifierPreProcess(Model *model);

    bool DoPreProcess(const void *input, size_t inputSize) override;
    void DoPreProcess_MaxminNor(std::vector<arm::app::face_landmark::KeypointResult> &results_KP, std::vector<float> &normalized_numbers, uint32_t KP_size);
    void DoPreProcess_ForeheadNor(std::vector<arm::app::face_landmark::KeypointResult> &results_KP, std::vector<float> &normalized_numbers, uint32_t KP_size);
    void MinMaxNormalize_Helium(std::vector<float> &input_output);
protected:
    Model *m_model = nullptr;
};

/**
 * @brief   Post-processing class for Image Classification use case.
 *          Implements methods declared by BasePostProcess and anything else needed
 *          to populate result vector.
 */
class ClassifierPostProcess : public BasePostProcess
{

private:
    Classifier &m_Classifier;
    const std::vector<std::string> &m_labels;
    std::vector<ClassificationResult> m_results;

public:
    ClassifierPostProcess(Classifier &classifier, Model *model,
                          const std::vector<std::string> &labels);

    bool DoPostProcess() override;
    ClassificationResult getResults();
    std::vector<ClassificationResult> getTopkResults();

protected:
    Model *m_model = nullptr;
};

} /* namespace app */
} /* namespace arm */

#endif /* CLASS_PROCESSING_HPP */