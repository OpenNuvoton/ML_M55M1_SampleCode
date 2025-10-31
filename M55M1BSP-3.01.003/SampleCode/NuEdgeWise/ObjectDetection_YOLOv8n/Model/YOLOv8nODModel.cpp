/**************************************************************************//**
 * @file     YOLOv8nODModel.cpp
 * @version  V1.00
 * @brief    YOLOv8n object detection model source file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "YOLOv8nODModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver &arm::app::YOLOv8nODModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::YOLOv8nODModel::EnlistOperations()
{
    this->m_opResolver.AddTranspose();

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
