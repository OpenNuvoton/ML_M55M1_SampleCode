/**************************************************************************//**
 * @file     FaceRecognProcessing.cpp
 * @version  V0.10
 * @brief    Face recognition pre/post processing
 * * SPDX-License-Identifier: Apache-2.0
 * @copyright (C) 2022 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include "FaceRecognProcessing.hpp"
#include "log_macros.h"

static void ConvertImgToInt8(void *data, const size_t kMaxImageSize)
{
    auto *tmp_req_data = static_cast<uint8_t *>(data);
    auto *tmp_signed_req_data = static_cast<int8_t *>(data);

    for (size_t i = 0; i < kMaxImageSize; i++)
    {
        tmp_signed_req_data[i] = (int8_t)(
                                     (int32_t)(tmp_req_data[i]) - 128);
    }
}


namespace arm
{
namespace app
{

FaceRecognPreProcess::FaceRecognPreProcess(Model *model)
{
    this->m_model = model;
}

bool FaceRecognPreProcess::DoPreProcess(const void *data, size_t inputSize)
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
        ConvertImgToInt8(inputTensor->data.data, inputTensor->bytes);
    }

    return true;
}

FaceRecognPostProcess::FaceRecognPostProcess(Embedding &embedding, Model *model)
    : m_embedding{embedding}
{
    this->m_model = model;
}

bool FaceRecognPostProcess::DoPostProcess()
{
    return false;
}


bool FaceRecognPostProcess::RunPostProcess(
	std::vector<float> &result
)
{
    return this->m_embedding.GetEmbeddingResults(
               this->m_model->GetOutputTensor(0),
               result);
}


} /* namespace app */
} /* namespace arm */