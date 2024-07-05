/**************************************************************************//**
 * @file     FaceLandmarkModel.hpp
 * @version  V1.00
 * @brief    Face landmark model header file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef FACE_LANDMARK_MODEL_HPP
#define FACE_LANDMARK_MODEL_HPP

#include "Model.hpp"

//For face landmark int8 model
#define FACE_LANDMARK_MESH_TENSOR_INDEX                         3               //[1,1,1,1404]
#define FACE_LANDMARK_LIPS_TENSOR_INDEX                         5               //[1,1,1,160]
#define FACE_LANDMARK_LEFT_EYE_TENSOR_INDEX                     0               //[1,1,1,142]
#define FACE_LANDMARK_RIGHT_EYE_TENSOR_INDEX            1               //[1,1,1,142]
#define FACE_LANDMARK_LEFT_IRIS_TENSOR_INDEX            4               //[1,1,1,10]
#define FACE_LANDMARK_RIGHT_IRIS_TENSOR_INDEX           6               //[1,1,1,10]
#define FACE_LANDMARK_FACE_FLAG_TENSOR_INDEX            2               //[1,1,1,1]

namespace arm
{
namespace app
{

class FaceLandmarkModel : public Model
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
    static constexpr int ms_maxOpCnt = 20;

    /* A mutable op resolver instance. */
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif /* FACE_LANDMARK_MODEL_HPP */
