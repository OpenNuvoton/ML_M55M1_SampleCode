/**************************************************************************//**
 * @file     KeypointResult.hpp
 * @version  V1.00
 * @brief    Keypoints of hand landmark header file
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef KEYPOINT_RESULT_HPP
#define KEYPOINT_RESULT_HPP


namespace arm
{
namespace app
{
namespace face_landmark
{

/**
 * @brief   Class representing a single keypoint result.
 */
class KeypointResult
{
public:
    /**
     * @brief       Constructor
     * @param[in]   x    x point
     * @param[in]   y    y point
     * @param[in]   z    z point
     **/
    KeypointResult(int x, int y, int z) :
        m_x(x),
        m_y(y),
        m_z(z)
    {
    };

    KeypointResult() = default;
    ~KeypointResult() = default;

    int     m_x{0};
    int     m_y{0};
    int     m_z{0};
};

} /* namespace face_landmark */
} /* namespace app */
} /* namespace arm */

#endif /* KEYPOINT_RESULT_HPP */
