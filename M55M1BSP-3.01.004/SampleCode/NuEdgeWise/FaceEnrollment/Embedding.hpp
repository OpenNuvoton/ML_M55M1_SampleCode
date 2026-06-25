/**************************************************************************//**
 * @file     Embedding.hpp
 * @version  V0.10
 * @brief    Embedding header
 * * SPDX-License-Identifier: Apache-2.0
 * @copyright (C) 2022 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#ifndef RECOGNIZER_HPP
#define RECOGNIZER_HPP

#include "TensorFlowLiteMicro.hpp"

#include <vector>

namespace arm
{
namespace app
{

/**
 * @brief   embedding - a helper class to get face embedding data.
 **/
class Embedding
{
public:
    /** @brief Constructor. */
    Embedding();

    /**
     * @brief       Gets the recognition results from the
     *              output vector.
     * @param[in]   outputTensor   Inference output tensor from an NN model.
     * @param[out]  embedding     A vector of embedding results.
     **/

    virtual bool GetEmbeddingResults(
        TfLiteTensor *outputTensor,
        std::vector<float> &embedding);

};

} /* namespace app */
} /* namespace arm */

#endif /* CLASSIFIER_HPP */
