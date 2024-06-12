/**************************************************************************//**
 * @file     PointHistoryProcessing.hpp
 * @version  V1.00
 * @brief    Point history model pre/post processing function
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <cstdio>
#include <iostream>

#include "PointHistoryProcessing.hpp"
#include "log_macros.h"

#define HAND_POINT_DIMENSION 	2

template<typename T>
void pop_front(std::vector<T> &v)
{
    if (v.size() > 0) {
        v.erase(v.begin());
    }
}
namespace arm
{
namespace app
{

PointHistoryPreProcess::PointHistoryPreProcess(Model *model)
{
    TfLiteTensor *inputTensor = model->GetInputTensor(0);

	m_pointHistoryStep = inputTensor->bytes / HAND_POINT_DIMENSION;
	m_pointHistory.resize(m_pointHistoryStep);
	m_pointHistoryNorm.resize(m_pointHistoryStep * HAND_POINT_DIMENSION);
    this->m_model = model;
	m_pointCoordIndex = 0;
}

bool PointHistoryPreProcess::CollectPoint(S_POINT_COORD *psPoint, int cols, int rows)
{
    if (psPoint == nullptr)
    {
        printf_err("Data pointer is null");
    }
	
	if(m_pointCoordIndex >= m_pointHistoryStep)
	{		
		//replace oldest point
		pop_front(m_pointHistory);
		m_pointHistory.push_back(*psPoint);		
	}
	else
	{
		m_pointHistory[m_pointCoordIndex] = *psPoint;
		m_pointCoordIndex ++;
	}

	if(m_pointCoordIndex >= m_pointHistoryStep)
	{
		//Calcute point normal value
		m_pointHistoryNorm[0] = 0.0;
		m_pointHistoryNorm[1] = 0.0;

		int j = 2;
		float fTemp;
		S_POINT_COORD sPointCoord;
		S_POINT_COORD sPointCoord_Prev = m_pointHistory[0];	
		
		for(int i = 1; i < m_pointHistoryStep; i ++)
		{
			sPointCoord =  m_pointHistory[i];
			fTemp = (float)(sPointCoord.i32X - sPointCoord_Prev.i32X) / cols;
			m_pointHistoryNorm[j] = fTemp;
			j ++;
			
			fTemp = (float)(sPointCoord.i32Y - sPointCoord_Prev.i32Y) / rows;
			m_pointHistoryNorm[j] = fTemp;
			j ++;
			
			sPointCoord_Prev = sPointCoord;
		}

		return true;
	}
	
	return false;
}

void PointHistoryPreProcess::ResetPointHistory(void)
{
	m_pointCoordIndex = 0;
}

bool PointHistoryPreProcess::DoPreProcess(const void *data, size_t inputSize)
{
#if 0 // test data, the result should be "Clockwise"
        m_pointHistoryNorm[0] = 0.0;
        m_pointHistoryNorm[1] = 0.0;
        m_pointHistoryNorm[2] = 0.011458333;
        m_pointHistoryNorm[3] = 0.018518519;
        m_pointHistoryNorm[4] = 0.017708333;
        m_pointHistoryNorm[5] = 0.040740741;
        m_pointHistoryNorm[6] = 0.01875;
        m_pointHistoryNorm[7] = 0.061111111;
        m_pointHistoryNorm[8] = 0.009375;
        m_pointHistoryNorm[9] = 0.090740741;
        
        m_pointHistoryNorm[10] = -0.001041667;
        m_pointHistoryNorm[11] = 0.111111111;
        m_pointHistoryNorm[12] = -0.016666667;
        m_pointHistoryNorm[13] = 0.125925926;
        m_pointHistoryNorm[14] = -0.032291667;
        m_pointHistoryNorm[15] = 0.133333333;
        m_pointHistoryNorm[16] = -0.05;
        m_pointHistoryNorm[17] = 0.135185185;
        m_pointHistoryNorm[18] = -0.047916667;
        m_pointHistoryNorm[19] = 0.133333333;
        m_pointHistoryNorm[20] = -0.071875;
        
        m_pointHistoryNorm[21] = 0.109259259;
        m_pointHistoryNorm[22] = -0.076041667;
        m_pointHistoryNorm[23] = 0.094444444;
        m_pointHistoryNorm[24] = -0.075;
        m_pointHistoryNorm[25] = 0.075925926;
        m_pointHistoryNorm[26] = -0.070833333;
        m_pointHistoryNorm[27] = 0.055555556;
        m_pointHistoryNorm[28] = -0.063541667;
        m_pointHistoryNorm[29] = 0.037037037;

        m_pointHistoryNorm[30] = -0.05625;
        m_pointHistoryNorm[31] = 0.033333333;
#endif

    TfLiteTensor *inputTensor = this->m_model->GetInputTensor(0);

    arm::app::QuantParams inQuantParams = arm::app::GetTensorQuantParams(inputTensor);

	//Quantize input tensor data
	auto *signed_req_data = static_cast<int8_t *>(inputTensor->data.data);

	for (size_t i = 0; i < inputTensor->bytes; i++)
	{
		auto i_data_int8 = static_cast<int8_t>(((m_pointHistoryNorm[i]) / inQuantParams.scale) + inQuantParams.offset);
		signed_req_data[i] = std::min<int8_t>(INT8_MAX, std::max<int8_t>(i_data_int8, INT8_MIN));
	}

	return true;
}

PointHistoryPostProcess::PointHistoryPostProcess(Classifier &classifier, Model *model,
                                         const std::vector<std::string> &labels,
                                         std::vector<ClassificationResult> &results)
    : m_Classifier{classifier},
      m_labels{labels},
      m_results{results}
{
    this->m_model = model;
}

bool PointHistoryPostProcess::DoPostProcess()
{
    return this->m_Classifier.GetClassificationResults(
               this->m_model->GetOutputTensor(0), this->m_results,
               this->m_labels, 3, false);
}

} /* namespace app */
} /* namespace arm */