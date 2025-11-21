/*
 * Copyright (c) 2022 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "DetectorPostProcessing.hpp"
#include "PlatformMath.hpp"
#include "log_macros.h"      /* Logging macros (optional) */

#include <cmath>

namespace arm
{
namespace app
{
namespace object_detection
{

DetectorPostprocessing::DetectorPostprocessing(
    const float threshold,
    const float nms,
    int numClasses,
    int topN)
    :   m_threshold(threshold),
        m_nms(nms),
        m_numClasses(numClasses),
        m_topN(topN)
{}

void DetectorPostprocessing::RunPostProcessing(
    uint32_t imgNetRows,
    uint32_t imgNetCols,
    uint32_t imgSrcRows,
    uint32_t imgSrcCols,
    TfLiteTensor *modelOutput0,
    std::vector<DetectionResult> &resultsOut    /* init postprocessing */
)
{
#if defined (__ICCARM__)
    Network net;
    net.inputWidth = static_cast<int>(imgNetCols);
    net.inputHeight = static_cast<int>(imgNetRows);
    net.numClasses = m_numClasses;


    Branch branches0;
    branches0.resolution = modelOutput0->dims->data[1];
    branches0.modelOutput = modelOutput0->data.int8;
    branches0.scale = ((TfLiteAffineQuantization *)(modelOutput0->quantization.params))->scale->data[0];
    branches0.zeroPoint = ((TfLiteAffineQuantization *)(modelOutput0->quantization.params))->zero_point->data[0];
    branches0.size = modelOutput0->bytes;

    net.branches.push_back(branches0);

    net.topN = m_topN;
#else
    Network net
    {
        .inputWidth = static_cast<int>(imgNetCols),
        .inputHeight = static_cast<int>(imgNetRows),
        .numClasses = m_numClasses,
        .branches = {
            Branch {
                .resolution = modelOutput0->dims->data[1],
                .modelOutput = modelOutput0->data.int8,
                .scale = ((TfLiteAffineQuantization *)(modelOutput0->quantization.params))->scale->data[0],
                .zeroPoint = ((TfLiteAffineQuantization *)(modelOutput0->quantization.params))->zero_point->data[0],
                .size = modelOutput0->bytes
            }
        },
        .topN = m_topN
    };
#endif
    /* End init */

    /* Start postprocessing */
    int originalImageWidth = imgSrcCols;
    int originalImageHeight = imgSrcRows;

    std::forward_list<image::Detection> detections;
    GetNetworkBoxes(net, originalImageWidth, originalImageHeight, m_threshold, detections);

    /* Do nms */
    CalculateNMS(detections, net.numClasses, m_nms);

    for (auto &it : detections)
    {

        // transfer to original img size (base on resize ratio)
        it.bbox.x = (it.bbox.x * originalImageWidth) / net.inputWidth;
        it.bbox.y = (it.bbox.y * originalImageHeight) / net.inputHeight;
        it.bbox.w = (it.bbox.w * originalImageWidth) / net.inputWidth;
        it.bbox.h = (it.bbox.h * originalImageHeight) / net.inputHeight;

        float xMin = it.bbox.x - it.bbox.w / 2.0f;
        float xMax = it.bbox.x + it.bbox.w / 2.0f;
        float yMin = it.bbox.y - it.bbox.h / 2.0f;
        float yMax = it.bbox.y + it.bbox.h / 2.0f;

        if (xMin < 0)
        {
            xMin = 0;
        }

        if (yMin < 0)
        {
            yMin = 0;
        }

        if (xMax > originalImageWidth)
        {
            xMax = originalImageWidth;
        }

        if (yMax > originalImageHeight)
        {
            yMax = originalImageHeight;
        }

        float boxX = xMin;
        float boxY = yMin;
        float boxWidth = xMax - xMin;
        float boxHeight = yMax - yMin;

        for (int j = 0; j < net.numClasses; ++j)
        {
            if (it.prob[j] > 0)
            {

                DetectionResult tmpResult = {};
                tmpResult.m_normalisedVal = it.prob[j];
                tmpResult.m_x0 = (int)boxX;
                tmpResult.m_y0 = (int)boxY;
                tmpResult.m_w = (int)boxWidth;
                tmpResult.m_h = (int)boxHeight;
                tmpResult.m_cls = j;

                resultsOut.push_back(tmpResult);
            }
        }
    }

#if defined (__ICCARM__)
    net.branches.erase(net.branches.begin(), net.branches.end());
#endif
}


void DetectorPostprocessing::InsertTopNDetections(std::forward_list<image::Detection> &detections, image::Detection &det)
{
    std::forward_list<image::Detection>::iterator it;
    std::forward_list<image::Detection>::iterator last_it;

    for (it = detections.begin(); it != detections.end(); ++it)
    {
        if (it->objectness > det.objectness)
            break;

        last_it = it;
    }

    if (it != detections.begin())
    {
        detections.emplace_after(last_it, det);
        detections.pop_front();
    }
}

void DetectorPostprocessing::GetNetworkBoxes(Network &net, int imageWidth, int imageHeight, float threshold, std::forward_list<image::Detection> &detections)
{
    auto det_objectness_comparator = [](image::Detection & pa, image::Detection & pb)
    {
        return pa.objectness < pb.objectness;
    };

    const float strides[] = {8, 16, 32};
    std::vector<int> hsizes;
    std::vector<int> wsizes;

    for (int stride : strides)
    {
        hsizes.push_back(net.inputHeight / stride); //[40, 20 ,8]
        wsizes.push_back(net.inputWidth / stride);
    }

    int section0_1d = hsizes[0] * wsizes[0]; // 40*40
    int section1_1d = hsizes[0] * wsizes[0] + hsizes[1] * wsizes[1]; // 40*40 + 20*20


    int numClasses = net.numClasses;
    int num = 0;

    for (int i = 0; i < net.branches[0].resolution; ++i)
    {
        int bbox_obj_offset = i * (numClasses + 5) + 4;
        int bbox_scr_offset = i * (numClasses + 5) + 5;
        float objectness = (static_cast<float>(net.branches[0].modelOutput[bbox_obj_offset]) - net.branches[0].zeroPoint) * net.branches[0].scale;

        if (objectness > threshold) // if obj < threshold, there is no way score*obj > threshold. because score, obj: [0, 1)
        {
            image::Detection det;
            int bbox_x_offset = i * (numClasses + 5);
            int bbox_y_offset = bbox_x_offset + 1;
            int bbox_w_offset = bbox_x_offset + 2;
            int bbox_h_offset = bbox_x_offset + 3;

            det.bbox.x = (static_cast<float>(net.branches[0].modelOutput[bbox_x_offset]) - net.branches[0].zeroPoint) * net.branches[0].scale;
            det.bbox.y = (static_cast<float>(net.branches[0].modelOutput[bbox_y_offset]) - net.branches[0].zeroPoint) * net.branches[0].scale;
            det.bbox.w = (static_cast<float>(net.branches[0].modelOutput[bbox_w_offset]) - net.branches[0].zeroPoint) * net.branches[0].scale;
            det.bbox.h = (static_cast<float>(net.branches[0].modelOutput[bbox_h_offset]) - net.branches[0].zeroPoint) * net.branches[0].scale;

            // yolox-n
            if (i < section0_1d)
            {
                det.bbox.x = (det.bbox.x + (i % wsizes[0])) * strides[0];
                det.bbox.y = (det.bbox.y + (i / hsizes[0])) * strides[0];
                det.bbox.w = std::exp(det.bbox.w) * strides[0];
                det.bbox.h = std::exp(det.bbox.h) * strides[0];
                //info("x: %d y: %d\n", (i%wsizes[0]), (i/hsizes[0]));
            }
            else if (i < section1_1d)
            {
                int section_i = i - section0_1d;
                //info("i: %d y: %lf i/hsizes[1]: %d stride: %lf\n", i, det.bbox.y, (section_i/hsizes[1]), strides[1]);
                det.bbox.x = (det.bbox.x + (section_i % wsizes[1])) * strides[1];
                det.bbox.y = (det.bbox.y + (section_i / hsizes[1])) * strides[1];
                det.bbox.w = std::exp(det.bbox.w) * strides[1];
                det.bbox.h = std::exp(det.bbox.h) * strides[1];
            }
            else
            {
                int section_i = i - section1_1d;
                //info("i: %d y: %lf i/hsizes[2]: %d stride: %lf\n", i, det.bbox.y, (section_i/hsizes[2]), strides[2]);
                det.bbox.x = (det.bbox.x + (section_i % wsizes[2])) * strides[2];
                det.bbox.y = (det.bbox.y + (section_i / hsizes[2])) * strides[2];
                det.bbox.w = std::exp(det.bbox.w) * strides[2];
                det.bbox.h = std::exp(det.bbox.h) * strides[2];
            }

            for (int j = 0; j < (numClasses); ++j)
            {
                float sig = (static_cast<float>(net.branches[0].modelOutput[bbox_scr_offset + j]) - net.branches[0].zeroPoint) * net.branches[0].scale * objectness;
                det.prob.emplace_back((sig > threshold) ? sig : 0);
            }

            //info("x: %lf y: %lf w: %lf h: %lf\n", det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);

            if (num < net.topN || net.topN <= 0)
            {
                detections.emplace_front(det);
                num += 1;
            }
            else if (num == net.topN)
            {
                detections.sort(det_objectness_comparator);
                InsertTopNDetections(detections, det);
                num += 1;
            }
            else
            {
                InsertTopNDetections(detections, det);
            }

        } // if
    } // for net.branches[0].resolution


    if (num > net.topN)
        num -= 1;
}

} /* namespace object_detection */
} /* namespace app */
} /* namespace arm */
