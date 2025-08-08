/**************************************************************************//**
 * @file     FaceLandmarkModel.cpp
 * @version  V1.00
 * @brief    Face landmark model source file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "FaceLandmarkModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver &arm::app::FaceLandmarkModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::FaceLandmarkModel::EnlistOperations()
{
    this->m_opResolver.AddPad();
    this->m_opResolver.AddSum();
    this->m_opResolver.AddGather();
    this->m_opResolver.AddFullyConnected();
    this->m_opResolver.AddReduceMin();

    this->m_opResolver.AddReduceMax();
    this->m_opResolver.AddDequantize();
    this->m_opResolver.AddFloor();
    this->m_opResolver.AddQuantize();
    this->m_opResolver.AddCast();

    this->m_opResolver.AddMaximum();
    this->m_opResolver.AddMinimum();
    this->m_opResolver.AddStridedSlice();
    this->m_opResolver.AddAdd();
    this->m_opResolver.AddConcatenation();

    this->m_opResolver.AddGatherNd();

#if defined(ARM_NPU)

    if (kTfLiteOk == this->m_opResolver.AddEthosU())
    {
        info("Added %s support to op resolver\n",
             tflite::GetString_ETHOSU());
    }
    else
    {
        printf_err("Failed to add Arm NPU support to op resolver.");
        return false;
    }

#endif /* ARM_NPU */
    return true;
}
