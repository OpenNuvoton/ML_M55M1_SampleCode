/**************************************************************************//**
 * @file     ImageSegModel.hpp
 * @version  V1.00
 * @brief    Image segmentation model header file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef IMAGE_SEG_MODEL_HPP
#define IMAGE_SEG_MODEL_HPP

#include "Model.hpp"

#define MODEL_INPUT_WIDTH	257
#define MODEL_INPUT_HEIGHT	257

#define MODEL_OUTPUT_WIDTH	MODEL_INPUT_WIDTH
#define MODEL_OUTPUT_HEIGHT	MODEL_INPUT_HEIGHT
#define MODEL_OUTPUT_CLASS	21

namespace arm
{
namespace app
{

class ImageSegModel : public Model
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
    static constexpr int ms_maxOpCnt = 2;

    /* A mutable op resolver instance. */
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif /* IMAGE_SEG_MODEL_HPP */
