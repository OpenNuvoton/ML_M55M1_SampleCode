#include "PoseLandmarkPostProcessing.hpp"
#include "PlatformMath.hpp"

#include <cmath>

inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

namespace arm
{
namespace app
{
namespace pose_landmark
{

enum
{
	eLANDMARK_TENSOR_INDEX_X,
	eLANDMARK_TENSOR_INDEX_Y,
	eLANDMARK_TENSOR_INDEX_Z,
	eLANDMARK_TENSOR_INDEX_VISIBILITY,
	eLANDMARK_TENSOR_INDEX_PRESENCE,
	eLANDMARK_TENSOR_INDEXS,
}E_LANDMARK_TENSOR_INDEX;

static float GetHandPresence(TfLiteTensor *modelOutput)
{
    float scale;
    int zeroPoint;
    size_t tensorSize;
	int8_t *tensorOutput = modelOutput->data.int8;

    scale = ((TfLiteAffineQuantization *)(modelOutput->quantization.params))->scale->data[0];
    zeroPoint = ((TfLiteAffineQuantization *)(modelOutput->quantization.params))->zero_point->data[0];
    tensorSize = modelOutput->bytes;

	std::vector<float>fTensorData(tensorSize);
	
	for(int i = 0; i < tensorSize; i++)
	{
		fTensorData[i] = scale * (static_cast<float>(tensorOutput[i]) - zeroPoint);
	}
	
	return fTensorData[0];
}	

static void GetPoseLandmark(
    float fXScale,
    float fYScale,
    float fZScale,
	TfLiteTensor *modelOutput,
	std::vector<KeypointResult> &resultsOut
)
{
    float scale;
    int zeroPoint;
    size_t tensorSize;
	int8_t *tensorOutput = modelOutput->data.int8;

    scale = ((TfLiteAffineQuantization *)(modelOutput->quantization.params))->scale->data[0];
    zeroPoint = ((TfLiteAffineQuantization *)(modelOutput->quantization.params))->zero_point->data[0];
    tensorSize = modelOutput->bytes;

	std::vector<float>fTensorData(tensorSize);

	for(int i = 0; i < tensorSize; i++)
	{
		fTensorData[i] = scale * (static_cast<float>(tensorOutput[i]) - zeroPoint);
	}

	for(int i = 0; i < tensorSize; i = i + eLANDMARK_TENSOR_INDEXS)
	{
		KeypointResult keypoint;
		
		keypoint.m_x = fTensorData[i + eLANDMARK_TENSOR_INDEX_X] * fXScale;
		keypoint.m_y = fTensorData[i + eLANDMARK_TENSOR_INDEX_Y] * fYScale;
		keypoint.m_z = fTensorData[i + eLANDMARK_TENSOR_INDEX_Z] * fZScale;
		keypoint.m_visibility = Sigmoid(fTensorData[i + eLANDMARK_TENSOR_INDEX_VISIBILITY]);
		keypoint.m_presence = Sigmoid(fTensorData[i + eLANDMARK_TENSOR_INDEX_PRESENCE]);
		
		resultsOut.push_back(keypoint);
	}
}

PoseLandmarkPostProcessing::PoseLandmarkPostProcessing(
    const float threshold)
    :   m_threshold(threshold)
{}

void PoseLandmarkPostProcessing::RunPostProcessing(
    uint32_t imgNetCols,
    uint32_t imgNetRows,
    uint32_t imgSrcCols,
    uint32_t imgSrcRows,
	TfLiteTensor *screenLandmarkTensor,
    TfLiteTensor *presenceTensor,
    std::vector<KeypointResult> &resultsOut    /* init postprocessing */
)
{
    float fXScale = (float)imgSrcCols / (float)imgNetCols; 
    float fYScale = (float)imgSrcRows / (float)imgNetRows;
    float fZScale = 1; //TODO: If have z-axis size

	//model tensor output 1 is for hand presence
	float fPosePresence = GetHandPresence(presenceTensor);

	resultsOut.clear();

	// If detect pose presence, start to get pose landmark
	if(fPosePresence >= m_threshold)
	{
		GetPoseLandmark(fXScale, fYScale, fZScale, screenLandmarkTensor, resultsOut);
	}
}

} /* namespace pose_landmark */
} /* namespace app */
} /* namespace arm */
