/**************************************************************************//**
 * @file     DetectionResult.hpp
 * @version  V1.00
 * @brief    DetectionResult header file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef DETECTION_RESULT_HPP
#define DETECTION_RESULT_HPP

#include <cstdio>
#include <vector>

struct S_DETECTION_BOX
{
    int x;
    int y;
    int w;
    int h;
    int cls;
    float normalisedVal;
};

namespace arm
{
namespace app
{
namespace yolov8n_od
{

/**
 * @brief   Class representing a single object result.
 */
class DetectionResult
{
public:
     DetectionResult(struct S_DETECTION_BOX detectBox) :
        m_detectBox(detectBox)
    {
    };

    DetectionResult() = default;
    ~DetectionResult() = default;
    struct S_DETECTION_BOX m_detectBox;
};

} /* namespace yolov8n_od */
} /* namespace app */
} /* namespace arm */

#endif /* POSE_RESULT_HPP */
