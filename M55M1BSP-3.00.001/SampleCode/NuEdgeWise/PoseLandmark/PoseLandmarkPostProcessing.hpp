#ifndef POSE_LANDMARK_POST_PROCESSING_HPP
#define POSE_LANDMARK_POST_PROCESSING_HPP

#include "KeypointResult.hpp"
#include "PoseLandmarkModel.hpp"

#include <forward_list>

namespace arm
{
namespace app
{
namespace pose_landmark
{

/**
 * @brief   Helper class to manage tensor post-processing for "pose landmark"
 *          output.
 */
class PoseLandmarkPostProcessing
{
public:
    /**
     * @brief       Constructor.
     * @param[in]   threshold     Post-processing threshold.
     **/
    explicit PoseLandmarkPostProcessing(float threshold = 0.5f);

    /**
     * @brief       Post processing part of hand landmark NN model.
     * @param[in]   imgNetRows      Number of rows in the network input image.
     * @param[in]   imgNetCols      Number of columns in the network input image.
     * @param[in]   imgSrcRows      Number of rows in the orignal input image.
     * @param[in]   imgSrcCols      Number of columns in the oringal input image.
     * @param[in]   screenLandmarkTensor  Screen landmark tensors after NN invoked.
     * @param[in]   presentTensor  Present tensor after NN invoked.
     * @param[out]  resultsOut   Vector of detected results.
     **/
    void RunPostProcessing(uint32_t imgNetRows,
                           uint32_t imgNetCols,
                           uint32_t imgSrcRows,
                           uint32_t imgSrcCols,
                           TfLiteTensor *screenLandmarkTensor,
                           TfLiteTensor *presentTensor,
                           std::vector<KeypointResult> &resultsOut);

private:
    float m_threshold;  /* Post-processing threshold */


};

} /* namespace hand_landmark */
} /* namespace app */
} /* namespace arm */

#endif /* HAND_LANDMARK_POST_PROCESSING_HPP */
