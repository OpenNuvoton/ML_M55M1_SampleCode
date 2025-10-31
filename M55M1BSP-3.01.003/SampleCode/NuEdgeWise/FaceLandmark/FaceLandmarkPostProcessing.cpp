#include "FaceLandmarkPostProcessing.hpp"
#include "PlatformMath.hpp"

#include <cmath>

inline float Sigmoid(float value) { return 1.0f / (1.0f + std::exp(-value)); }

namespace arm
{
namespace app
{
namespace face_landmark
{

enum
{
	eMESH_LANDMARK_DIM_INDEX_X,
	eMESH_LANDMARK_DIM_INDEX_Y,
	eMESH_LANDMARK_DIM_INDEX_Z,
	eMESH_LANDMARK_DIM_INDEXS,
}E_MESH_LANDMARK_DIM_INDEX;

enum
{
	eIRIS_LANDMARK_DIM_INDEX_X,
	eIRIS_LANDMARK_DIM_INDEX_Y,
	eIRIS_LANDMARK_DIM_INDEXS,
}E_IRIS_LANDMARK_DIM_INDEX;

#define IRIS_Z_AVERAGE_INDEXS   16

static int s_i32LeftIrisZIndexMappling[IRIS_Z_AVERAGE_INDEXS] = {
                  // Lower contour.
                  33, 7, 163, 144, 145, 153, 154, 155, 133,
                  // Upper contour (excluding corners).
                  246, 161, 160, 159, 158, 157, 173
};

static int s_i32RightIrisZIndexMappling[IRIS_Z_AVERAGE_INDEXS] = {
                  // Lower contour.
                  263, 249, 390, 373, 374, 380, 381, 382, 362,
                  // Upper contour (excluding corners).
                  466, 388, 387, 386, 385, 384, 398
};

static float GetFacePresence(TfLiteTensor *modelOutput)
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
	
	return Sigmoid(fTensorData[0]);
}	

static void GetMeshLandmark(
    float fXScale,
    float fYScale,
    float fZScale,
	TfLiteTensor *meshTensor,
	std::vector<KeypointResult> &resultsOut
)
{
    float scale;
    int zeroPoint;
    size_t tensorSize;
	int8_t *tensorOutput = meshTensor->data.int8;

    scale = ((TfLiteAffineQuantization *)(meshTensor->quantization.params))->scale->data[0];
    zeroPoint = ((TfLiteAffineQuantization *)(meshTensor->quantization.params))->zero_point->data[0];
    tensorSize = meshTensor->bytes;

	std::vector<float>fTensorData(tensorSize);

	for(int i = 0; i < tensorSize; i++)
	{
		fTensorData[i] = scale * (static_cast<float>(tensorOutput[i]) - zeroPoint);
	}

	for(int i = 0; i < tensorSize; i = i + eMESH_LANDMARK_DIM_INDEXS)
	{
		KeypointResult keypoint;
		
		keypoint.m_x = fTensorData[i + eMESH_LANDMARK_DIM_INDEX_X] * fXScale;
		keypoint.m_y = fTensorData[i + eMESH_LANDMARK_DIM_INDEX_Y] * fYScale;
		keypoint.m_z = fTensorData[i + eMESH_LANDMARK_DIM_INDEX_Z] * fZScale;
		
		resultsOut.push_back(keypoint);
	}
}

static void GetLeftIrisLandmark(
    float fXScale,
    float fYScale,
    float fZScale,
	TfLiteTensor *leftIrisTensor,
	std::vector<KeypointResult> &meshLandmark,
	std::vector<KeypointResult> &resultsOut
)
{
    float scale;
    int zeroPoint;
    size_t tensorSize;
	int8_t *tensorOutput = leftIrisTensor->data.int8;

    scale = ((TfLiteAffineQuantization *)(leftIrisTensor->quantization.params))->scale->data[0];
    zeroPoint = ((TfLiteAffineQuantization *)(leftIrisTensor->quantization.params))->zero_point->data[0];
    tensorSize = leftIrisTensor->bytes;

	std::vector<float>fTensorData(tensorSize);

	for(int i = 0; i < tensorSize; i++)
	{
		fTensorData[i] = scale * (static_cast<float>(tensorOutput[i]) - zeroPoint);
	}

	float fSum = 0.0;

	for (size_t k = 0; k < IRIS_Z_AVERAGE_INDEXS; k ++) {
			fSum += meshLandmark[s_i32LeftIrisZIndexMappling[k]].m_z;
	}

	for(int i = 0; i < tensorSize; i = i + eIRIS_LANDMARK_DIM_INDEXS)
	{
		KeypointResult keypoint;
		
		keypoint.m_x = fTensorData[i + eIRIS_LANDMARK_DIM_INDEX_X] * fXScale;
		keypoint.m_y = fTensorData[i + eIRIS_LANDMARK_DIM_INDEX_Y] * fYScale;
		keypoint.m_z = fSum / IRIS_Z_AVERAGE_INDEXS;
		
		resultsOut.push_back(keypoint);
	}
}

static void GetRightIrisLandmark(
    float fXScale,
    float fYScale,
    float fZScale,
	TfLiteTensor *rightIrisTensor,
	std::vector<KeypointResult> &meshLandmark,
	std::vector<KeypointResult> &resultsOut
)
{
    float scale;
    int zeroPoint;
    size_t tensorSize;
	int8_t *tensorOutput = rightIrisTensor->data.int8;

    scale = ((TfLiteAffineQuantization *)(rightIrisTensor->quantization.params))->scale->data[0];
    zeroPoint = ((TfLiteAffineQuantization *)(rightIrisTensor->quantization.params))->zero_point->data[0];
    tensorSize = rightIrisTensor->bytes;

	std::vector<float>fTensorData(tensorSize);

	for(int i = 0; i < tensorSize; i++)
	{
		fTensorData[i] = scale * (static_cast<float>(tensorOutput[i]) - zeroPoint);
	}

	float fSum = 0.0;

	for (size_t k = 0; k < IRIS_Z_AVERAGE_INDEXS; k ++) {
			fSum += meshLandmark[s_i32RightIrisZIndexMappling[k]].m_z;
	}

	for(int i = 0; i < tensorSize; i = i + eIRIS_LANDMARK_DIM_INDEXS)
	{
		KeypointResult keypoint;
		
		keypoint.m_x = fTensorData[i + eIRIS_LANDMARK_DIM_INDEX_X] * fXScale;
		keypoint.m_y = fTensorData[i + eIRIS_LANDMARK_DIM_INDEX_Y] * fYScale;
		keypoint.m_z = fSum / IRIS_Z_AVERAGE_INDEXS;
		
		resultsOut.push_back(keypoint);
	}
}



FaceLandmarkPostProcessing::FaceLandmarkPostProcessing(
    const float threshold)
    :   m_threshold(threshold)
{}

void FaceLandmarkPostProcessing::RunPostProcessing(uint32_t imgNetRows,
   uint32_t imgNetCols,
   uint32_t imgSrcRows,
   uint32_t imgSrcCols,
   TfLiteTensor *meshTensor,
   TfLiteTensor *leftIrisTensor,
   TfLiteTensor *rightIrisTensor,
   TfLiteTensor *presenceTensor,
   std::vector<KeypointResult> &resultsOut)
{
    float fXScale = (float)imgSrcCols / (float)imgNetCols; 
    float fYScale = (float)imgSrcRows / (float)imgNetRows;
    float fZScale = 1; //TODO: If have z-axis size

	//model tensor output 1 is for hand presence
	float fFacePresence = GetFacePresence(presenceTensor);
	std::vector<KeypointResult> leftIrisResult;
	std::vector<KeypointResult> rightIrisResult;

	resultsOut.clear();
	leftIrisResult.clear();
	rightIrisResult.clear();

	// If detect pose presence, start to get pose landmark
	if(fFacePresence >= m_threshold)
	{
		GetMeshLandmark(fXScale, fYScale, fZScale, meshTensor, resultsOut);

		//Iris landmark was used by face landmark with attention model
		if(leftIrisTensor)
			GetLeftIrisLandmark(fXScale, fYScale, fZScale, leftIrisTensor, resultsOut, leftIrisResult);

		if(rightIrisTensor)
			GetRightIrisLandmark(fXScale, fYScale, fZScale, rightIrisTensor, resultsOut, rightIrisResult);

		for(size_t i = 0; i < leftIrisResult.size(); i ++)
		{
			resultsOut.push_back(leftIrisResult[i]);
		}

		for(size_t j = 0; j < rightIrisResult.size(); j ++)
		{
			resultsOut.push_back(rightIrisResult[j]);
		}
	}
}



} /* namespace face_landmark */
} /* namespace app */
} /* namespace arm */
