/**************************************************************************//**
 * @file     Embedding.cpp
 * @version  V0.10
 * @brief    Recognizer function
 * * SPDX-License-Identifier: Apache-2.0
 * @copyright (C) 2022 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "Embedding.hpp"

#include "TensorFlowLiteMicro.hpp"
#include "log_macros.h"

#include <vector>
#include <string>
#include <set>
#include <cstdint>
#include <cinttypes>

namespace arm
{
namespace app
{
Embedding::Embedding()
{
}

bool Embedding::GetEmbeddingResults(
    TfLiteTensor *outputTensor,
     std::vector<float> &embedding)
{

    if (outputTensor == nullptr)
    {
        printf_err("Output vector is null pointer.\n");
        return false;
    }

    uint32_t embedDimension = 1;

    for (int inputDim = 0; inputDim < outputTensor->dims->size; inputDim++)
    {
        embedDimension *= outputTensor->dims->data[inputDim];
    }

    /* De-Quantize Output Tensor */
    QuantParams quantParams = GetTensorQuantParams(outputTensor);

    /* Floating point tensor data to be populated
     * NOTE: The assumption here is that the output tensor size isn't too
     * big and therefore, there's neglibible impact on heap usage. */
    std::vector<float> embedArray(embedDimension);
	embedding.clear();
	
    /* Populate the floating point buffer */
    switch (outputTensor->type)
    {
        case kTfLiteUInt8:
        {
            uint8_t *tensor_buffer = tflite::GetTensorData<uint8_t>(outputTensor);

            for (size_t i = 0; i < embedDimension; ++i)
            {
                embedArray[i] = quantParams.scale *
                                (static_cast<float>(tensor_buffer[i]) - quantParams.offset);
            }

            break;
        }

        case kTfLiteInt8:
        {
            int8_t *tensor_buffer = tflite::GetTensorData<int8_t>(outputTensor);

            for (size_t i = 0; i < embedDimension; ++i)
            {
                embedArray[i] = quantParams.scale *
                                (static_cast<float>(tensor_buffer[i]) - quantParams.offset);
            }

            break;
        }

        case kTfLiteFloat32:
        {
            float *tensor_buffer = tflite::GetTensorData<float>(outputTensor);

            for (size_t i = 0; i < embedDimension; ++i)
            {
                embedArray[i] = tensor_buffer[i];
            }

            break;
        }

        default:
            printf_err("Tensor type %s not supported by classifier\n",
                       TfLiteTypeGetName(outputTensor->type));
            return false;
    }

	for(size_t i = 0; i < embedArray.size(); i ++)
	{
		embedding.push_back(embedArray[i]);
	}	
	
    return true;
}


} /* namespace app */
} /* namespace arm */
