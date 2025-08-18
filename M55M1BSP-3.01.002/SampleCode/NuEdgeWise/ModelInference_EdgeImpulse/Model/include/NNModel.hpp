/**************************************************************************//**
 * @file     NNModel.hpp
 * @version  V1.00
 * @brief    NN model header file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef NN_MODEL_HPP
#define NN_MODEL_HPP

#include "Model.hpp"

namespace arm
{
namespace app
{

class NNModel : public Model
{

public:
    /* Indices for the expected model - based on input tensor shape */
    static constexpr uint32_t ms_inputRowsIdx     = 1;
    static constexpr uint32_t ms_inputColsIdx     = 2;
    static constexpr uint32_t ms_inputChannelsIdx = 3;

protected:
    /** @brief   Gets the reference to op resolver interface class. */
    const tflite::MicroOpResolver &GetOpResolver() override;

    /** @brief   Adds operations to the op resolver instance. */
    bool EnlistOperations() override;

private:
    /* Maximum number of individual operations that can be enlisted. */
    static constexpr int ms_maxOpCnt = 1;

    /* A mutable op resolver instance. */
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif /* HAND_LANDMARK_MODEL_HPP */