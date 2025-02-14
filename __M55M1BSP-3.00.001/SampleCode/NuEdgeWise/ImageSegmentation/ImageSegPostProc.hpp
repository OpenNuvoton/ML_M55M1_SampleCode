#ifndef IMAGE_SEG_POST_PROC_HPP
#define IMAGE_SEG_POST_PROC_HPP

#include "ImageSegModel.hpp"
#include <forward_list>
#include "imlib.h"          /* Image processing */

namespace arm
{
namespace app
{
namespace image_seg
{

/**
 * @brief   Helper class to manage tensor post-processing for "image segmentation"
 *          output.
 */
class ImageSegPostProcessing
{
public:
    /**
     * @brief       Constructor.
     **/
    explicit ImageSegPostProcessing(arm::app::ImageSegModel *model);

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
	arm::app::ImageSegModel *m_model;
};


} /* namespace image_seg */
} /* namespace app */
} /* namespace arm */

#endif /* IMAGE_SEG_POST_PROC_HPP */
