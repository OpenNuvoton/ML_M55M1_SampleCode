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

#define MODEL_INPUT_RESOL  (192) //model input resolution

#define MODEL_OUTPUT_WIDTH (MODEL_INPUT_RESOL)
#define MODEL_OUTPUT_HEIGHT (MODEL_INPUT_RESOL)
#define MODEL_OUTPUT_CLASS (80) //coco dataset

#define MODEL_OUTPUT_OBJECT_TENSOR	(1)
#define MODEL_OUTPUT_MASK_TENSOR		(0)

//For 192 model, it would be 756
#define MODEL_OUTPUT_ANCHOR_BOXES		(756)

#define MODEL_OBJECT_BOX_CX_POS	(0)
#define MODEL_OBJECT_BOX_CY_POS	(1)
#define MODEL_OBJECT_BOX_W_POS	(2)
#define MODEL_OBJECT_BOX_H_POS	(3)
#define MODEL_OBJECT_BOX_LEN	(MODEL_OBJECT_BOX_H_POS - MODEL_OBJECT_BOX_CX_POS + 1)

#define MODEL_OBJECT_CONF_START_POS	(4)
#define MODEL_OBJECT_CONF_END_POS	(83)
#define MODEL_OBJECT_CONF_LEN	(MODEL_OBJECT_CONF_END_POS - MODEL_OBJECT_CONF_START_POS + 1)

#define MODEL_OBJECT_MASK_START_POS	(84)
#define MODEL_OBJECT_MASK_END_POS	(115)
#define MODEL_OBJECT_MASK_LEN	(MODEL_OBJECT_MASK_END_POS - MODEL_OBJECT_MASK_START_POS + 1)

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
    static constexpr int ms_maxOpCnt = 2;

    /* A mutable op resolver instance. */
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif /* NN_MODEL_HPP */