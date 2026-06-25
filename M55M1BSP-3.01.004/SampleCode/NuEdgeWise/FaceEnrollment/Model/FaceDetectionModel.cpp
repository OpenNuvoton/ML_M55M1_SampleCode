/**************************************************************************//**
 * @file     NNModel.cpp
 * @version  V1.00
 * @brief    NN model source file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "FaceDetectionModel.hpp"
#include "log_macros.h"

const tflite::MicroOpResolver &arm::app::FaceDetectionModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::FaceDetectionModel::EnlistOperations()
{
/****************************************************************************
 * autogen section: Add Operators
 ****************************************************************************/
	this->m_opResolver.AddEthosU();

    return true;
}
