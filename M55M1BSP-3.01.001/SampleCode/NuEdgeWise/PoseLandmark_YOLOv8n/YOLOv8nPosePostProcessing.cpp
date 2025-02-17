#include "YOLOv8nPosePostProcessing.hpp"
#include "PlatformMath.hpp"

#include <cmath>

using namespace arm::app::yolov8n_pose;


/************** YOLOv8n-pose  */ 

#if (YOLOV8_POSE_INPUT_TENSOR == 224)
#define YOLOV8_POSE_INPUT_TENSOR_WIDTH   224
#define YOLOV8_POSE_INPUT_TENSOR_HEIGHT  224
#elif (YOLOV8_POSE_INPUT_TENSOR == 256)
#define YOLOV8_POSE_INPUT_TENSOR_WIDTH   256
#define YOLOV8_POSE_INPUT_TENSOR_HEIGHT  256
#else		//192
#define YOLOV8_POSE_INPUT_TENSOR_WIDTH   192
#define YOLOV8_POSE_INPUT_TENSOR_HEIGHT  192
#endif


float Calculate1DOverlap(float x1Center, float width1, float x2Center, float width2)
{
	float left_1 = x1Center - width1/2;
	float left_2 = x2Center - width2/2;
	float leftest = left_1 > left_2 ? left_1 : left_2;

	float right_1 = x1Center + width1/2;
	float right_2 = x2Center + width2/2;
	float rightest = right_1 < right_2 ? right_1 : right_2;

	return rightest - leftest;
}

float CalculateBoxIntersect(Box& box1, Box& box2)
{
	float width = Calculate1DOverlap(box1.x, box1.w, box2.x, box2.w);
	if (width < 0) {
		return 0;
	}
	float height = Calculate1DOverlap(box1.y, box1.h, box2.y, box2.h);
	if (height < 0) {
		return 0;
	}

	float total_area = width*height;
	return total_area;
}

float CalculateBoxUnion(Box& box1, Box& box2)
{
	float boxes_intersection = CalculateBoxIntersect(box1, box2);
	float boxes_union = box1.w * box1.h + box2.w * box2.h - boxes_intersection;
	return boxes_union;
}

float CalculateBoxIOU(Box& box1, Box& box2)
{
	float boxes_intersection = CalculateBoxIntersect(box1, box2);
	if (boxes_intersection == 0) {
		return 0;
	}

	float boxes_union = CalculateBoxUnion(box1, box2);
	if (boxes_union == 0) {
		return 0;
	}

	return boxes_intersection / boxes_union;
}

void CalculateNMS(std::forward_list<Detection>& detections, int classes, float iouThreshold)
{
	int idxClass{0};
	auto CompareProbs = [idxClass](Detection& prob1, Detection& prob2) {
		return prob1.prob[idxClass] > prob2.prob[idxClass];
	};

	for (idxClass = 0; idxClass < classes; ++idxClass) {
		detections.sort(CompareProbs);

		for (auto it=detections.begin(); it != detections.end(); ++it) {
			if (it->prob[idxClass] == 0) continue;
			for (auto itc=std::next(it, 1); itc != detections.end(); ++itc) {
				if (itc->prob[idxClass] == 0) {
					continue;
				}
				if (CalculateBoxIOU(it->bbox, itc->bbox) > iouThreshold) {
					itc->prob[idxClass] = 0;
				}
			}
		}
	}
}

void AnchorMatrixConstruct(
	std::vector<AnchorBox> &vAnchorBoxs,
	int i32Stride,
	int i32StrideTotalAnchors
)
{
	int i;
	float fStartAnchorValue = 0.5;
	int iMaxAnchorValue = (YOLOV8_POSE_INPUT_TENSOR_WIDTH/i32Stride);
	float fAnchor0StepValue = 0.;
	float fAnchor1StepValue = -1.;

	for(int i = 0; i < i32StrideTotalAnchors; i++)
	{
		AnchorBox sAnchorBox;

		if((i % iMaxAnchorValue)==0)
		{
			fStartAnchorValue = 0.5;
			fAnchor0StepValue = 0.;
			fAnchor1StepValue++;
		}

		sAnchorBox.w = fStartAnchorValue + (fAnchor0StepValue++);
		sAnchorBox.h = fStartAnchorValue + fAnchor1StepValue;
		
		vAnchorBoxs.push_back(sAnchorBox);
	}

	//for(int i=0; i < vAnchorBoxs.size(); i++)
	//{
	//	printf("vAnchorBoxs[%d].w = %f \n", i, vAnchorBoxs[i].w);
	//	printf("vAnchorBoxs[%d].h = %f \n", i, vAnchorBoxs[i].h);		
	//}
}

void CalBoxXYWH(
	TfLiteTensor* psBoxOutputTensor,
	std::vector<AnchorBox> &vAnchorBoxs,
	int	i32AnchorIndex,
	int i32Stride,
	int i32StrideTotalAnchors,
	Detection &sDetection
)
{
	int i;
    float scaleBox;
    int zeroPointBox;
	int anchors;
	int boxDataSize;
    float  XYWHResult[4];
	
	int8_t *tensorOutputBox = psBoxOutputTensor->data.int8;
    scaleBox = ((TfLiteAffineQuantization *)(psBoxOutputTensor->quantization.params))->scale->data[0];
    zeroPointBox = ((TfLiteAffineQuantization *)(psBoxOutputTensor->quantization.params))->zero_point->data[0];

	anchors = psBoxOutputTensor->dims->data[1];
	boxDataSize = psBoxOutputTensor->dims->data[2];

	if(anchors != i32StrideTotalAnchors)
	{
		printf("CalBoxXYWH(): error tensor size not match \n");
		return;
	}

	//x:16 bytes, y:16 bytes, w:16 bytes, h:16 bytes
	if(boxDataSize != 64)
	{
		printf("CalBoxXYWH(): error tensor size not match \n");
		return;
	}

	tensorOutputBox = tensorOutputBox + (i32AnchorIndex * boxDataSize);
	
    for(int k = 0 ; k < 4 ; k++)
    {
		std::vector<float> XYWHSoftmaxTemp(16);
        float XYWHSoftmaxResult=0;

        for(int i = 0 ; i < 16 ; i++)
        {
			XYWHSoftmaxTemp[i] = scaleBox * (static_cast<float>(tensorOutputBox[k*16 + i]) - zeroPointBox);
		}

		arm::app::math::MathUtils::SoftmaxF32(XYWHSoftmaxTemp);
        for(int i = 0 ; i < 16 ; i++)
        {

            XYWHSoftmaxResult = XYWHSoftmaxResult + XYWHSoftmaxTemp[i]*i;
        }
        XYWHResult[k] = XYWHSoftmaxResult;
	}

    /* dist2bbox */
    float x1 = vAnchorBoxs[i32AnchorIndex].w -  XYWHResult[0];
    float y1 = vAnchorBoxs[i32AnchorIndex].h -  XYWHResult[1];
    float x2 = vAnchorBoxs[i32AnchorIndex].w +  XYWHResult[2];
    float y2 = vAnchorBoxs[i32AnchorIndex].h +  XYWHResult[3];
    
    float cx = (x1 + x2)/2.;
    float cy = (y1 + y2)/2.;
    float w = x2 - x1;
    float h = y2 - y1;

    XYWHResult[0] = cx * i32Stride;
    XYWHResult[1] = cy * i32Stride;
    XYWHResult[2] = w * i32Stride;
    XYWHResult[3] = h * i32Stride;

	sDetection.bbox.x = XYWHResult[0] - (0.5 * XYWHResult[2]);
    sDetection.bbox.y = XYWHResult[1] - (0.5 * XYWHResult[3]);
    sDetection.bbox.w = XYWHResult[2];
    sDetection.bbox.h = XYWHResult[3];
}


void CalKeyPoints(
	TfLiteTensor* psKeyPointsOutputTensor,
	std::vector<AnchorBox> &vAnchorBoxs,
	int	i32AnchorIndex,
	int i32Stride,
	int i32StartAnchorsIndex,
	std::vector<struct KeyPoint>&keyPoints
)
{
	int i;
    float scaleKeyPoint;
    int zeroPointKeyPoint;
	int8_t *tensorOutputKeyPoint = psKeyPointsOutputTensor->data.int8;
	float anchorW;
	float anchorH;
	
    scaleKeyPoint = ((TfLiteAffineQuantization *)(psKeyPointsOutputTensor->quantization.params))->scale->data[0];
    zeroPointKeyPoint = ((TfLiteAffineQuantization *)(psKeyPointsOutputTensor->quantization.params))->zero_point->data[0];
	keyPoints.clear();

	tensorOutputKeyPoint = tensorOutputKeyPoint + ((i32StartAnchorsIndex + i32AnchorIndex) * YOLOV8N_POSE_KEYPOINT_NUM * YOLOV8N_POSE_KEYPOINT_ELEM);
	anchorW = vAnchorBoxs[i32AnchorIndex].w;
	anchorH = vAnchorBoxs[i32AnchorIndex].h;
	
	for(i = 0; i < YOLOV8N_POSE_KEYPOINT_NUM; i ++)
	{
		struct KeyPoint keyPoint;
		
		keyPoint.x = scaleKeyPoint * (static_cast<float>(tensorOutputKeyPoint[0]) - zeroPointKeyPoint);
		keyPoint.y = scaleKeyPoint * (static_cast<float>(tensorOutputKeyPoint[1]) - zeroPointKeyPoint);
		keyPoint.visible = scaleKeyPoint * (static_cast<float>(tensorOutputKeyPoint[2]) - zeroPointKeyPoint);
		
		keyPoint.x = (keyPoint.x * 2.0 + (anchorW - 0.5))* i32Stride;
		keyPoint.y = (keyPoint.y * 2.0 + (anchorH - 0.5))* i32Stride;
		keyPoint.visible = arm::app::math::MathUtils::SigmoidF32(keyPoint.visible);
#if 0
		if(i == 0)
		{
			printf("keyPoint i32StartAnchorsIndex %d \n", i32StartAnchorsIndex);
			printf("keyPoint i32AnchorIndex %d \n", i32AnchorIndex);
			printf("keyPoint %d, keyPoint.x %f \n", i, keyPoint.x);
			printf("keyPoint %d, keyPoint.y %f \n", i, keyPoint.y);
			printf("keyPoint %d, keyPoint.visible %f \n", i, keyPoint.visible);
		}
#endif			
		tensorOutputKeyPoint += YOLOV8N_POSE_KEYPOINT_ELEM;
		keyPoints.push_back(keyPoint);
	}
	
	
}

void CalDetectionBox(
	TfLiteTensor* psConfidenceOutputTensor,
	TfLiteTensor* psBoxOutputTensor,
	std::vector<AnchorBox> &vAnchorBoxs,
	int i32Stride,
	int i32StrideTotalAnchors,
	float fThreshold,
	std::forward_list<Detection>&sDetections		
)
{
	int i;
    float scaleConf;
    int zeroPointConf;
    size_t tensorSizeConf;
	float score = 0;
	int8_t *tensorOutputConf = psConfidenceOutputTensor->data.int8;

    scaleConf = ((TfLiteAffineQuantization *)(psConfidenceOutputTensor->quantization.params))->scale->data[0];
    zeroPointConf = ((TfLiteAffineQuantization *)(psConfidenceOutputTensor->quantization.params))->zero_point->data[0];
    tensorSizeConf = psConfidenceOutputTensor->bytes;

	if(tensorSizeConf != i32StrideTotalAnchors)
	{
		printf("CalDetectionBox(): error tensor size not match \n");
		return;
	}

	//check confidence is over threshold or not
	for(i = 0 ; i <i32StrideTotalAnchors; i++)
	{
		score = arm::app::math::MathUtils::SigmoidF32(scaleConf * (static_cast<float>(tensorOutputConf[i]) - zeroPointConf));

		if(score >= fThreshold)
		{
			//inqueue detection list
			arm::app::yolov8n_pose::Detection det;
			det.strideIndex = i32Stride;
			det.anchorIndex = i;
			det.prob.emplace_back(score);
			
			//cal box xywh
			CalBoxXYWH(psBoxOutputTensor,
				vAnchorBoxs,
				i,
				i32Stride,
				i32StrideTotalAnchors,
				det);
            sDetections.emplace_front(det);
		}	
	}

}

/*****************************/
namespace arm
{
namespace app
{
namespace yolov8n_pose
{

YOLOv8nPosePostProcessing::YOLOv8nPosePostProcessing(
	arm::app::YOLOv8nPoseModel *model,
	const float threshold)
    :   m_threshold(threshold),
		m_model(model)
{
	int i;

	//For YOLOV8_POSE_INPUT_TENSOR == 256, it would be 1024
	//For YOLOV8_POSE_INPUT_TENSOR == 192, it would be 576	
	m_stride8_total_anchors = pow(( YOLOV8_POSE_INPUT_TENSOR_WIDTH / YOLOV8N_POSE_STRIDE_8),2);
	//For YOLOV8_POSE_INPUT_TENSOR == 256, it would be 256
	//For YOLOV8_POSE_INPUT_TENSOR == 192, it would be 144	
	m_stride16_total_anchors = pow(( YOLOV8_POSE_INPUT_TENSOR_WIDTH / YOLOV8N_POSE_STRIDE_16),2);
	//For YOLOV8_POSE_INPUT_TENSOR == 256, it would be 64
	//For YOLOV8_POSE_INPUT_TENSOR == 192, it would be 36	
	m_stride32_total_anchors = pow(( YOLOV8_POSE_INPUT_TENSOR_WIDTH / YOLOV8N_POSE_STRIDE_32),2);

	m_stride8_anchros.clear();
	m_stride16_anchros.clear();
	m_stride32_anchros.clear();
	
	//For YOLOV8_POSE_INPUT_TENSOR == 256
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[31.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride8[i]*8, [4x4], [12x4], ...,[252x4], [4x12], ...
	//For YOLOV8_POSE_INPUT_TENSOR == 192
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[23.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride8[i]*8, [4x4], [12x4], ...,[188x4], [4x12], ...
	AnchorMatrixConstruct(m_stride8_anchros, YOLOV8N_POSE_STRIDE_8, m_stride8_total_anchors);
	//For YOLOV8_POSE_INPUT_TENSOR == 256
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[15.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride16[i]*16, [8x8], [24x8], ...,[248x8], [8x24], ...
	//For YOLOV8_POSE_INPUT_TENSOR == 192
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[11.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride16[i]*16, [8x8], [24x8], ...,[184x8], [8x24], ...
	AnchorMatrixConstruct(m_stride16_anchros, YOLOV8N_POSE_STRIDE_16, m_stride16_total_anchors);
	//For YOLOV8_POSE_INPUT_TENSOR == 256
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[7.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride32[i]*32, [16x16], [48x16], ...,[240x16], [16x48], ...
	//For YOLOV8_POSE_INPUT_TENSOR == 192
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[5.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride32[i]*32, [16x16], [48x16], ...,[176x16], [16x48], ...
	AnchorMatrixConstruct(m_stride32_anchros, YOLOV8N_POSE_STRIDE_32, m_stride32_total_anchors);
	
}

void YOLOv8nPosePostProcessing::RunPostProcessing(
    uint32_t imgNetCols,
    uint32_t imgNetRows,
    uint32_t imgSrcCols,
    uint32_t imgSrcRows,
    std::vector<PoseResult> &resultsOut    /* init postprocessing */
)
{
    float fXScale = (float)imgSrcCols / (float)imgNetCols; 
    float fYScale = (float)imgSrcRows / (float)imgNetRows;
	int i;
	
	std::forward_list<Detection> sDetections;
	GetNetworkBoxes(sDetections);
	CalculateNMS(sDetections, 1, 0.45);

	resultsOut.clear();
	
	for (auto box=sDetections.begin(); box != sDetections.end(); ++box) {
		if(box->prob[0] != 0)
		{
			TfLiteTensor* psKeyPointsTensor;
			int startAnchorIndex;
			std::vector<struct KeyPoint>nnKeyPoints;
			
			psKeyPointsTensor = m_model->GetOutputTensor(YOLOV8N_POSE_KEYPOINT_TENSOR_INDEX);

			if(box->strideIndex == YOLOV8N_POSE_STRIDE_8){
				startAnchorIndex = 0;
				CalKeyPoints(psKeyPointsTensor, m_stride8_anchros, box->anchorIndex, box->strideIndex, startAnchorIndex, nnKeyPoints);
			}
			else if(box->strideIndex == YOLOV8N_POSE_STRIDE_16){
				startAnchorIndex = m_stride8_total_anchors;
				CalKeyPoints(psKeyPointsTensor, m_stride16_anchros, box->anchorIndex, box->strideIndex, startAnchorIndex, nnKeyPoints);
			}
			else if(box->strideIndex == YOLOV8N_POSE_STRIDE_32){
				startAnchorIndex = m_stride8_total_anchors + m_stride16_total_anchors;
				CalKeyPoints(psKeyPointsTensor, m_stride32_anchros, box->anchorIndex, box->strideIndex, startAnchorIndex, nnKeyPoints);
			}
				
			
			struct S_POSE_BOX poseBox;
			std::vector<struct S_KEY_POINT> keyPoints;
	
			poseBox.x = box->bbox.x * fXScale;
			poseBox.y = box->bbox.y * fYScale;
			poseBox.w = box->bbox.w * fXScale;
			poseBox.h = box->bbox.h * fYScale;

//			printf("bbox.x: %f \n", box->bbox.x);
//			printf("bbox.y: %f \n", box->bbox.y);	   
//			printf("bbox.w: %f \n", box->bbox.w);
//			printf("bbox.h: %f \n", box->bbox.h);	   
			
			poseBox.x = std::min(std::max(poseBox.x, 0), (int)imgSrcCols - 1);
			poseBox.y = std::min(std::max(poseBox.y, 0), (int)imgSrcRows - 1);
			
			poseBox.w = std::min(std::max(poseBox.w, 0), (int)imgSrcCols - 1);
			poseBox.h = std::min(std::max(poseBox.h, 0), (int)imgSrcRows - 1);
			
			struct KeyPoint nnKeyPoint;

			for( i = 0; i < nnKeyPoints.size(); i ++)
			{
				struct S_KEY_POINT keyPoint;
				nnKeyPoint = nnKeyPoints[i];
				
				keyPoint.x = nnKeyPoint.x * fXScale;
				keyPoint.y = nnKeyPoint.y * fYScale;
				keyPoint.visible = nnKeyPoint.visible;

				keyPoint.x = std::min(std::max(keyPoint.x, 0), (int)imgSrcCols - 1);
				keyPoint.y = std::min(std::max(keyPoint.y, 0), (int)imgSrcRows - 1);
				keyPoints.push_back(keyPoint);
			}

			PoseResult poseResult(poseBox, keyPoints);
			resultsOut.push_back(poseResult);
		}
	}
}

void YOLOv8nPosePostProcessing::GetNetworkBoxes(
        std::forward_list<Detection>& detections)
{
	TfLiteTensor* psConfidenceTensor;
	TfLiteTensor* psBoxTensor;
	
	psConfidenceTensor = m_model->GetOutputTensor(YOLOV8N_POSE_STRIDE8_CONFIDENCE_TENSOR_INDEX);
	psBoxTensor = m_model->GetOutputTensor(YOLOV8N_POSE_STRIDE8_BOX_TENSOR_INDEX);
	
	CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride8_anchros, YOLOV8N_POSE_STRIDE_8, m_stride8_total_anchors, m_threshold, detections); 

	psConfidenceTensor = m_model->GetOutputTensor(YOLOV8N_POSE_STRIDE16_CONFIDENCE_TENSOR_INDEX);
	psBoxTensor = m_model->GetOutputTensor(YOLOV8N_POSE_STRIDE16_BOX_TENSOR_INDEX);
	
	CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride16_anchros, YOLOV8N_POSE_STRIDE_16, m_stride16_total_anchors, m_threshold, detections); 

	psConfidenceTensor = m_model->GetOutputTensor(YOLOV8N_POSE_STRIDE32_CONFIDENCE_TENSOR_INDEX);
	psBoxTensor = m_model->GetOutputTensor(YOLOV8N_POSE_STRIDE32_BOX_TENSOR_INDEX);
	
	CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride32_anchros, YOLOV8N_POSE_STRIDE_32, m_stride32_total_anchors, m_threshold, detections); 

}


} /* namespace YOLOv8nPosePostProcessing */
} /* namespace app */
} /* namespace arm */
