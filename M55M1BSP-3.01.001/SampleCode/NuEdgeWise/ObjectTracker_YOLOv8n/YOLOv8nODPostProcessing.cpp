#include "YOLOv8nODPostProcessing.hpp"
#include "PlatformMath.hpp"

#include <cmath>

using namespace arm::app::yolov8n_od;

/************** YOLOv8n-pose  */ 

#if (YOLOV8_OD_INPUT_TENSOR == 224)
#define YOLOV8_OD_INPUT_TENSOR_WIDTH   224
#define YOLOV8_OD_INPUT_TENSOR_HEIGHT  224
#elif (YOLOV8_OD_INPUT_TENSOR == 256)
#define YOLOV8_OD_INPUT_TENSOR_WIDTH   256
#define YOLOV8_OD_INPUT_TENSOR_HEIGHT  256
#else		//192
#define YOLOV8_OD_INPUT_TENSOR_WIDTH   192
#define YOLOV8_OD_INPUT_TENSOR_HEIGHT  192
#endif

void AnchorMatrixConstruct(
	std::vector<AnchorBox> &vAnchorBoxs,
	int i32Stride,
	int i32StrideTotalAnchors
)
{
	int i;
	float fStartAnchorValue = 0.5;
	int iMaxAnchorValue = (YOLOV8_OD_INPUT_TENSOR_WIDTH/i32Stride);
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
	int i, j;
    float scaleConf;
    int zeroPointConf;
    size_t tensorSizeConf;
	float maxScore = 0.0;
	int maxConf;
	int cls= 0;
	int8_t *tensorOutputConf = psConfidenceOutputTensor->data.int8;

    scaleConf = ((TfLiteAffineQuantization *)(psConfidenceOutputTensor->quantization.params))->scale->data[0];
    zeroPointConf = ((TfLiteAffineQuantization *)(psConfidenceOutputTensor->quantization.params))->zero_point->data[0];
    tensorSizeConf = psConfidenceOutputTensor->bytes;

	if((tensorSizeConf / YOLOV8N_OD_CLASS) != i32StrideTotalAnchors)
	{
		printf("CalDetectionBox(): error tensor size not match \n");
		return;
	}

	//check confidence is over threshold or not
	for(i = 0 ; i <i32StrideTotalAnchors; i++)
	{
		maxScore = 0.0;
		cls = 0;
		maxConf = -128;

		for(j = 0; j < YOLOV8N_OD_CLASS; j ++)
		{
			int confTesorData;
			
			confTesorData  = tensorOutputConf[(i * YOLOV8N_OD_CLASS) + j];
			
			if(confTesorData > maxConf)
			{
				maxConf = confTesorData;
				cls = j;				
			}
		}		

		maxScore = arm::app::math::MathUtils::SigmoidF32(scaleConf * (static_cast<float>(maxConf - zeroPointConf)));		
		
		if(maxScore >= fThreshold)
		{
			//printf("max cls %d and score %f i32StrideTotalAnchors %d, index %d\n", cls, maxScore, i32StrideTotalAnchors, i);
			//inqueue detection list
			arm::app::yolov8n_od::Detection det;
			det.strideIndex = i32Stride;
			det.anchorIndex = i;
			det.cls = cls;

			for(j = 0; j < YOLOV8N_OD_CLASS; j ++){
				maxScore =  arm::app::math::MathUtils::SigmoidF32(scaleConf * (static_cast<float>(tensorOutputConf[(i * YOLOV8N_OD_CLASS) + j] - zeroPointConf)));
				det.prob.emplace_back(maxScore);
			}

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
namespace yolov8n_od
{

YOLOv8nODPostProcessing::YOLOv8nODPostProcessing(
	arm::app::YOLOv8nODModel *model,
	const float threshold)
    :   m_threshold(threshold),
		m_model(model)
{
	int i;

	//For YOLOV8_OD_INPUT_TENSOR == 256, it would be 1024
	//For YOLOV8_OD_INPUT_TENSOR == 192, it would be 576	
	m_stride8_total_anchors = pow(( YOLOV8_OD_INPUT_TENSOR_WIDTH / YOLOV8N_OD_STRIDE_8),2);
	//For YOLOV8_OD_INPUT_TENSOR == 256, it would be 256
	//For YOLOV8_OD_INPUT_TENSOR == 192, it would be 144	
	m_stride16_total_anchors = pow(( YOLOV8_OD_INPUT_TENSOR_WIDTH / YOLOV8N_OD_STRIDE_16),2);
	//For YOLOV8_OD_INPUT_TENSOR == 256, it would be 64
	//For YOLOV8_OD_INPUT_TENSOR == 192, it would be 36	
	m_stride32_total_anchors = pow(( YOLOV8_OD_INPUT_TENSOR_WIDTH / YOLOV8N_OD_STRIDE_32),2);

	m_stride8_anchros.clear();
	m_stride16_anchros.clear();
	m_stride32_anchros.clear();
	
	//For YOLOV8_POSE_INPUT_TENSOR == 256
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[31.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride8[i]*8, [4x4], [12x4], ...,[252x4], [4x12], ...
	//For YOLOV8_POSE_INPUT_TENSOR == 192
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[23.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride8[i]*8, [4x4], [12x4], ...,[188x4], [4x12], ...
	AnchorMatrixConstruct(m_stride8_anchros, YOLOV8N_OD_STRIDE_8, m_stride8_total_anchors);
	//For YOLOV8_POSE_INPUT_TENSOR == 256
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[15.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride16[i]*16, [8x8], [24x8], ...,[248x8], [8x24], ...
	//For YOLOV8_POSE_INPUT_TENSOR == 192
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[11.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride16[i]*16, [8x8], [24x8], ...,[184x8], [8x24], ...
	AnchorMatrixConstruct(m_stride16_anchros, YOLOV8N_OD_STRIDE_16, m_stride16_total_anchors);
	//For YOLOV8_POSE_INPUT_TENSOR == 256
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[7.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride32[i]*32, [16x16], [48x16], ...,[240x16], [16x48], ...
	//For YOLOV8_POSE_INPUT_TENSOR == 192
	//Anchor arrary would be [0.5,0.5], [1.5,0.5], ...[5.5, 0.5], [0.5,1.5], .....
	//So anchors box dimension will m_anchors_stride32[i]*32, [16x16], [48x16], ...,[176x16], [16x48], ...
	AnchorMatrixConstruct(m_stride32_anchros, YOLOV8N_OD_STRIDE_32, m_stride32_total_anchors);
	
}

void YOLOv8nODPostProcessing::RunPostProcessing(
    uint32_t imgNetCols,
    uint32_t imgNetRows,
    uint32_t imgSrcCols,
    uint32_t imgSrcRows,
    std::vector<DetectionResult> &resultsOut    /* init postprocessing */
)
{
    float fXScale = (float)imgSrcCols / (float)imgNetCols; 
    float fYScale = (float)imgSrcRows / (float)imgNetRows;
	int i;
	
	std::forward_list<Detection> sDetections;
	GetNetworkBoxes(sDetections);
	CalculateNMS(sDetections, YOLOV8N_OD_CLASS, 0.45);
	
	resultsOut.clear();

	float score = 0.0;
	int cls= 0;

	for (auto box=sDetections.begin(); box != sDetections.end(); ++box) {

		score = box->prob[box->cls]; 

		if(score > 0)
		{
			struct S_DETECTION_BOX detectBox;

			detectBox.x = box->bbox.x * fXScale;
			detectBox.y = box->bbox.y * fYScale;
			detectBox.w = box->bbox.w * fXScale;
			detectBox.h = box->bbox.h * fYScale;

	//			printf("bbox.x: %f \n", box->bbox.x);
	//			printf("bbox.y: %f \n", box->bbox.y);	   
	//			printf("bbox.w: %f \n", box->bbox.w);
	//			printf("bbox.h: %f \n", box->bbox.h);	   
			
			detectBox.x = std::min(std::max(detectBox.x, 0), (int)imgSrcCols - 1);
			detectBox.y = std::min(std::max(detectBox.y, 0), (int)imgSrcRows - 1);
			
			detectBox.w = std::min(std::max(detectBox.w, 0), (int)imgSrcCols - 1);
			detectBox.h = std::min(std::max(detectBox.h, 0), (int)imgSrcRows - 1);

			detectBox.cls = box->cls;
			detectBox.normalisedVal = score;

			DetectionResult detectResult(detectBox);
			resultsOut.push_back(detectResult);
		}
	}
}

void YOLOv8nODPostProcessing::GetNetworkBoxes(
        std::forward_list<Detection>& detections)
{
	TfLiteTensor* psConfidenceTensor;
	TfLiteTensor* psBoxTensor;
	
	psConfidenceTensor = m_model->GetOutputTensor(YOLOV8N_OD_STRIDE8_CONFIDENCE_TENSOR_INDEX);
	psBoxTensor = m_model->GetOutputTensor(YOLOV8N_OD_STRIDE8_BOX_TENSOR_INDEX);
	
	CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride8_anchros, YOLOV8N_OD_STRIDE_8, m_stride8_total_anchors, m_threshold, detections); 

	psConfidenceTensor = m_model->GetOutputTensor(YOLOV8N_OD_STRIDE16_CONFIDENCE_TENSOR_INDEX);
	psBoxTensor = m_model->GetOutputTensor(YOLOV8N_OD_STRIDE16_BOX_TENSOR_INDEX);
	
	CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride16_anchros, YOLOV8N_OD_STRIDE_16, m_stride16_total_anchors, m_threshold, detections); 

	psConfidenceTensor = m_model->GetOutputTensor(YOLOV8N_OD_STRIDE32_CONFIDENCE_TENSOR_INDEX);
	psBoxTensor = m_model->GetOutputTensor(YOLOV8N_OD_STRIDE32_BOX_TENSOR_INDEX);
	
	CalDetectionBox(psConfidenceTensor, psBoxTensor, m_stride32_anchros, YOLOV8N_OD_STRIDE_32, m_stride32_total_anchors, m_threshold, detections); 

}

} /* namespace YOLOv8nODPostProcessing */
} /* namespace app */
} /* namespace arm */
	
