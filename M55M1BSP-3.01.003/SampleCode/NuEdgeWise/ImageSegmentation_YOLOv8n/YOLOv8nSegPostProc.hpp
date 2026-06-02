#ifndef YOLOV8N_SEG_POST_PROC_HPP
#define YOLOV8N_SEG_POST_PROC_HPP

#include "YOLOv8nSegModel.hpp"
#include <forward_list>
#include "imlib.h"          /* Image processing */

namespace arm
{
namespace app
{
namespace yolov8n_seg
{

/**
 * Contains the x,y co-ordinates of a box centre along with the box width and height.
 */
struct Box {
    float x;
    float y;
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
 * @brief   Helper class to manage tensor post-processing for "image segmentation"
 *          output.
 */
class YOLOv8nSegPostProcessing
{
public:
    /**
     * @brief       Constructor.
     **/
    explicit YOLOv8nSegPostProcessing(arm::app::NNModel *model, float threshold);

    /**
     * @brief       Post processing part of YOLOv8n pose model.
     * @param[in]   colorMaps       color maps for each lable
     * @param[out]  segImg          segmentation image
     **/
    void RunPostProcessing(
        std::vector <uint16_t> &colorMaps,
        image_t &segImg
    );

private:
	arm::app::NNModel *m_model;
  float m_threshold;
};


} /* namespace yolov8n_seg */
} /* namespace app */
} /* namespace arm */


#endif /* YOLOV8N_SEG_POST_PROC_HPP */
