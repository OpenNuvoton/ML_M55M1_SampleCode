/**************************************************************************//**
 * @file     HandLandmarkModel.cpp
 * @version  V1.00
 * @brief    Hand landmark model source file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "PointHistoryClassifierModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver &arm::app::PointHistoryClassifierModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::PointHistoryClassifierModel::EnlistOperations()
{
    this->m_opResolver.AddPad();
	this->m_opResolver.AddDequantize();
	this->m_opResolver.AddQuantize();
	
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
