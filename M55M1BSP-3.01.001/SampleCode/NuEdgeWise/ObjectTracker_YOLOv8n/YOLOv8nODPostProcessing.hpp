#ifndef YOLOV8N_OD_POST_PROCESSING_HPP
#define YOLOV8N_OD_POST_PROCESSING_HPP

#include "DetectionResult.hpp"
#include "YOLOv8nODModel.hpp"

#include <forward_list>

namespace arm
{
namespace app
{
namespace yolov8n_od
{

#define YOLOV8N_OD_STRIDE_8	(8)
#define YOLOV8N_OD_STRIDE_16	(16)
#define YOLOV8N_OD_STRIDE_32	(32)

#define YOLOV8N_OD_STRIDE8_CONFIDENCE_TENSOR_INDEX	(2)		//[1, 1024, 80] for YOLOv8n_256
#define YOLOV8N_OD_STRIDE16_CONFIDENCE_TENSOR_INDEX	(4)		//[1, 256, 80] for YOLOv8n_256
#define YOLOV8N_OD_STRIDE32_CONFIDENCE_TENSOR_INDEX	(3)		//[1, 64, 80] for YOLOv8n_256

#define YOLOV8N_OD_STRIDE8_BOX_TENSOR_INDEX		(0)		//[1, 1024, 64] for YOLOv8n_256
#define YOLOV8N_OD_STRIDE16_BOX_TENSOR_INDEX	(1)		//[1, 256, 64] for YOLOv8n_256
#define YOLOV8N_OD_STRIDE32_BOX_TENSOR_INDEX	(5)		//[1, 64, 64] for YOLOv8n_256

	
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
	int strideIndex;
	int anchorIndex;
	int cls;
	std::vector<float> prob;
};

struct AnchorBox {
	float w;
	float h;
};
	
/**
 * @brief   Helper class to manage tensor post-processing for "yolov8n pose"
 *          output.
 */
class YOLOv8nODPostProcessing
{
public:
    /**
     * @brief       Constructor.
     * @param[in]   threshold     Post-processing threshold.
     **/
    explicit YOLOv8nODPostProcessing(arm::app::YOLOv8nODModel *model, float threshold = 0.5f);

    /**
     * @brief       Post processing part of YOLOv8n pose model.
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
	arm::app::YOLOv8nODModel *m_model;
    float m_threshold;  /* Post-processing threshold */
	int m_stride8_total_anchors;		//total anchors for stride 8
	int m_stride16_total_anchors;		//total anchors for stride 16
	int m_stride32_total_anchors;		//total anchors for stride 32

	std::vector<AnchorBox> m_stride8_anchros;
	std::vector<AnchorBox> m_stride16_anchros;
	std::vector<AnchorBox> m_stride32_anchros;

	void GetNetworkBoxes(std::forward_list<Detection>& detections);

};

} /* namespace yolov8n_od */
} /* namespace app */
} /* namespace arm */

#endif /* YOLOV8N_OD_POST_PROCESSING_HPP */
