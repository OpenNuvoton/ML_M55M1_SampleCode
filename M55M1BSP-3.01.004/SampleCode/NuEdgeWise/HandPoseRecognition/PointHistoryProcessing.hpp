/**************************************************************************//**
 * @file     PointHistoryProcessing.hpp
 * @version  V1.00
 * @brief    Point history model pre/post processing function
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef POINT_HISTORY_PROCESSING_HPP
#define POINT_HISTORY_PROCESSING_HPP

#include "BaseProcessing.hpp"
#include "Model.hpp"
#include "Classifier.hpp"

namespace arm
{
namespace app
{

/**
 * @brief   Pre-processing class for point histroy use case.
 *          Implements methods declared by BasePreProcess and anything else needed
 *          to populate input tensors ready for inference.
 */
class PointHistoryPreProcess : public BasePreProcess
{
public:
	typedef struct {
		int32_t i32X;
		int32_t i32Y;		
	}S_POINT_COORD;

	explicit PointHistoryPreProcess(Model *model);
	bool CollectPoint(S_POINT_COORD *psPoint, int cols, int rows);
    bool DoPreProcess(const void *input, size_t inputSize) override;
	void ResetPointHistory(void);
protected:
    Model *m_model = nullptr;
	//record each step point coordinate 
	std::vector<S_POINT_COORD> m_pointHistory;
	//record each step point normailize float value
	std::vector<float> m_pointHistoryNorm;
	int m_pointHistoryStep;
	int m_pointCoordIndex;
};

/**
 * @brief   Post-processing class for Image Classification use case.
 *          Implements methods declared by BasePostProcess and anything else needed
 *          to populate result vector.
 */
class PointHistoryPostProcess : public BasePostProcess
{

private:
    Classifier &m_Classifier;
    const std::vector<std::string> &m_labels;
    std::vector<ClassificationResult> &m_results;

public:
    PointHistoryPostProcess(Classifier &classifier, Model *model,
                        const std::vector<std::string> &labels,
                        std::vector<ClassificationResult> &results);

    bool DoPostProcess() override;
protected:
    Model *m_model = nullptr;
};

} /* namespace app */
} /* namespace arm */



#endif /* POINT_HISTORY_PROCESSING_HPP */
