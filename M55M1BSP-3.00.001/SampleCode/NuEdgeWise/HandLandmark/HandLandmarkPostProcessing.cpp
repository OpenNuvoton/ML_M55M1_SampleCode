

#include "HandLandmarkPostProcessing.hpp"
#include "PlatformMath.hpp"

#include <cmath>

namespace arm
{
namespace app
{
namespace hand_landmark
{

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

static void GetHandLandmark(
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

	for(int i = 0; i < tensorSize; i = i + 3)
	{
		KeypointResult keypoint;
		
		keypoint.m_x = fTensorData[i] * fXScale;
		keypoint.m_y = fTensorData[i + 1] * fYScale;
		keypoint.m_z = fTensorData[i + 2] * fZScale;
		
		resultsOut.push_back(keypoint);
	}
}

HandLandmarkPostProcessing::HandLandmarkPostProcessing(
    const float threshold)
    :   m_threshold(threshold)
{}

void HandLandmarkPostProcessing::RunPostProcessing(
    uint32_t imgNetCols,
    uint32_t imgNetRows,
    uint32_t imgSrcCols,
    uint32_t imgSrcRows,
	TfLiteTensor *screenLandmarkTensor,
    TfLiteTensor *presentTensor,
    std::vector<KeypointResult> &resultsOut    /* init postprocessing */
)
{
    float fXScale = (float)imgSrcCols / (float)imgNetCols; 
    float fYScale = (float)imgSrcRows / (float)imgNetRows;
    float fZScale = 1; //TODO: If have z-axis size

	//model tensor output 1 is for hand presence
	float fHandPresence = GetHandPresence(presentTensor);

	resultsOut.clear();

	// If detect hand presence, start get hand landmark
	if(fHandPresence >= m_threshold)
	{
		GetHandLandmark(fXScale, fYScale, fZScale, screenLandmarkTensor, resultsOut);
	}
}


} /* namespace hand_landmark */
} /* namespace app */
} /* namespace arm */
