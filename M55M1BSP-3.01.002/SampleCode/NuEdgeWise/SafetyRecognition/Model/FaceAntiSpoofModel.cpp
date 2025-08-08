/**************************************************************************//**
 * @file     FaceAntiSpoofModel.hpp
 * @version  V1.00
 * @brief    Face AntiSpoof model header file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "FaceAntiSpoofModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver &arm::app::FaceAntiSpoofModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::FaceAntiSpoofModel::EnlistOperations()
{
    this->m_opResolver.AddFullyConnected();

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
