/**************************************************************************//**
 * @file     PoseResult.hpp
 * @version  V1.00
 * @brief    Keypoints of PoseResult header file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef POSE_RESULT_HPP
#define POSE_RESULT_HPP

#include <cstdio>
#include <vector>

struct S_KEY_POINT
{
    int x;
    int y;
    float visible;
};

struct S_POSE_BOX
{
    int x;
    int y;
    int w;
    int h;
};

namespace arm
{
namespace app
{
namespace yolov8n_pose
{

/**
 * @brief   Class representing a single pose result.
 */
class PoseResult
{
public:
     PoseResult(struct S_POSE_BOX poseBox, std::vector<struct S_KEY_POINT> keyPoints) :
        m_poseBox(poseBox),
        m_keyPoints(keyPoints)
    {
    };

    PoseResult() = default;
    ~PoseResult() = default;
	std::vector<struct S_KEY_POINT> m_keyPoints;
    struct S_POSE_BOX m_poseBox;
};

} /* namespace PoseResult */
} /* namespace app */
} /* namespace arm */

#endif /* POSE_RESULT_HPP */
