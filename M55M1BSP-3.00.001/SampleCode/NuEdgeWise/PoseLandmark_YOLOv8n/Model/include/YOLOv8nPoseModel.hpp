/**************************************************************************//**
 * @file     YOLOv8nPoseModel.hpp
 * @version  V1.00
 * @brief    VOLOv8n Pose landmark model header file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef _YOLOV8N_POSE_MODEL_HPP_
#define _YOLOV8N_POSE_MODEL_HPP_

#include "Model.hpp"

#define YOLOV8_POSE_INPUT_TENSOR	(192)	//256,224,192

namespace arm
{
namespace app
{

class YOLOv8nPoseModel : public Model
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

#endif /* _YOLOV8N_POSE_MODEL_HPP_ */
