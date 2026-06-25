#ifndef YOLOV11N_OD_POST_PROCESSING_HPP
#define YOLOV11N_OD_POST_PROCESSING_HPP

#include "DetectionResult.hpp"
#include "YOLOv11nODModel.hpp"

#include <forward_list>

namespace arm
{
namespace app
{
namespace yolov11n_od
{

/**
 * Contains the x,y co-ordinates of a box centre along with the box width and height.
 */
struct Box {
    float cx;
    float cy;
    float w;
    float h;
};

struct Detection {
    Box bbox;
    int anchorIndex;
    int cls;
    std::vector<float> prob;
};

/**
 * @brief   Helper class to manage tensor post-processing for "yolov11n object detection" model. It takes the output tensor from the model and converts it into a more usable format for further processing or display. The class provides methods to run post-processing on the model's output and extract the detected objects along with their bounding boxes and associated probabilities.
 *          output.
 */
class YOLOv11nODPostProcessing
{
public:
    /**
     * @brief       Constructor.
     * @param[in]   threshold     Post-processing threshold.
     **/
    explicit YOLOv11nODPostProcessing(arm::app::YOLOv11nODModel *model, float threshold = 0.5f);

    /**
     * @brief       Post processing part of YOLOv11n object detection model.
     * @param[in]   imgNetRows      Number of rows in the network input image.
     * @param[in]   imgNetCols      Number of columns in the network input image.
     * @param[in]   imgSrcRows      Number of rows in the orignal input image.
     * @param[in]   imgSrcCols      Number of columns in the oringal input image.
     * @param[out]  resultsOut   Vector of detected results.
     **/
    void RunPostProcessing(uint32_t imgNetRows,
                           uint32_t imgNetCols,
                           uint32_t imgSrcRows,
                           uint32_t imgSrcCols,
                           std::vector<DetectionResult> &resultsOut);

private:
    arm::app::YOLOv11nODModel *m_model;
    float m_threshold;  /* Post-processing threshold */
};

} /* namespace yolov11n_od */
} /* namespace app */
} /* namespace arm */

#endif /* YOLOV11N_OD_POST_PROCESSING_HPP */
