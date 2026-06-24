#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "YOLOv11nODPostProcessing.hpp"
#include "PlatformMath.hpp"

using namespace arm::app::yolov11n_od;
using MatrixXint8 = Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>;

static int32_t find_realiable_box(
    Eigen::Ref<Eigen::Map<MatrixXint8>> psObjectMat,
    float fQScale,
    int i32QZeroPoint,
    float fThreadhold,
    std::forward_list<Detection>& sDetections
)
{
    Eigen::Ref<MatrixXint8> tConfMat = psObjectMat.block<MODEL_OUTPUT_ANCHOR_BOXES, MODEL_OUTPUT_CLASS>(0, 4);
    //std::cout << "conf matrix 1st rows: " << tConfMat.row(0) << std::endl;

    float fMaxConf = 0;
    float fScore = 0;
    int i32MaxConf;
    int i32Cls = 0;

    for( int r = 0; r < tConfMat.rows(); r++)
    {
        fMaxConf = 0;
        i32MaxConf = -128;
        i32Cls = 0;

        // check each class confidence and find the max one as the box confidence and class
        for( int c = 0; c < tConfMat.cols(); c++)
        {
            if(tConfMat(r, c) > i32MaxConf)
            {
                i32MaxConf = tConfMat(r, c);
                i32Cls = c;
            }
        }

        fMaxConf = fQScale * (static_cast<float>(i32MaxConf - i32QZeroPoint));

        if(fMaxConf > fThreadhold)
        {
            //std::cout << "conf matrix 1st rows: " << psObjectMat->row(r) << std::endl;
            MatrixXint8 tBoxMat = psObjectMat.block<1, MODEL_OBJECT_BOX_LEN>(r, MODEL_OBJECT_BOX_CX_POS);
            //std::cout << "box matrix: " << tBoxMat << std::endl;

            //printf("The row %d max conf %f cls %d \n", r, fMaxConf, i32Cls);

            Detection det;
            det.anchorIndex = r;
            det.cls = i32Cls;

            // store all class confidence for latter NMS and mask processing
            for(int i = 0; i < MODEL_OBJECT_CONF_LEN; i ++) {
                fMaxConf = fQScale * (static_cast<float>(tConfMat(r, i) - i32QZeroPoint));
                det.prob.emplace_back(fMaxConf);
            }

            // Calculate Box X,Y,W,H
            float cx = fQScale * (static_cast<float>(tBoxMat(0, MODEL_OBJECT_BOX_CX_POS) - i32QZeroPoint));
            float cy = fQScale * (static_cast<float>(tBoxMat(0, MODEL_OBJECT_BOX_CY_POS) - i32QZeroPoint));
            float w = fQScale * (static_cast<float>(tBoxMat(0, MODEL_OBJECT_BOX_W_POS) - i32QZeroPoint));
            float h = fQScale * (static_cast<float>(tBoxMat(0, MODEL_OBJECT_BOX_H_POS) - i32QZeroPoint));

            det.bbox.cx = MODEL_OUTPUT_WIDTH * cx;
            det.bbox.cy = MODEL_OUTPUT_HEIGHT * cy;
            det.bbox.w = MODEL_OUTPUT_WIDTH * w;
            det.bbox.h = MODEL_OUTPUT_HEIGHT * h;

            float halfW = det.bbox.w / 2;
            float halfH = det.bbox.h / 2;

            //clip the box within the image boundary
            if(det.bbox.cx - halfW < 0)
            {
                det.bbox.w = det.bbox.cx * 2;
                halfW = det.bbox.cx;
            }

            if(det.bbox.cy - halfH < 0)
            {
                det.bbox.h = det.bbox.cy * 2;
                halfH = det.bbox.cy;
            }

            if(det.bbox.cx + halfW >  MODEL_OUTPUT_WIDTH)
                det.bbox.w = (MODEL_OUTPUT_WIDTH - det.bbox.cx) * 2;

            if(det.bbox.cy + halfH >  MODEL_OUTPUT_HEIGHT)
                det.bbox.h = (MODEL_OUTPUT_HEIGHT - det.bbox.cy) * 2;

            //printf("The bbox cls: %d, x:%f, y: %f, w:%f, h: %f\n", det.cls, det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);
            sDetections.emplace_front(det);
        }
    }

    return 0;
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
    float width = Calculate1DOverlap(box1.cx, box1.w, box2.cx, box2.w);
    if (width < 0) {
        return 0;
    }
    float height = Calculate1DOverlap(box1.cy, box1.h, box2.cy, box2.h);
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



/*****************************/
namespace arm
{
namespace app
{
namespace yolov11n_od
{

YOLOv11nODPostProcessing::YOLOv11nODPostProcessing(
    arm::app::YOLOv11nODModel *model,
    const float threshold)
    :   m_threshold(threshold),
        m_model(model)
{

}

void YOLOv11nODPostProcessing::RunPostProcessing(
    uint32_t imgNetCols,
    uint32_t imgNetRows,
    uint32_t imgSrcCols,
    uint32_t imgSrcRows,
    std::vector<DetectionResult> &resultsOut    /* init postprocessing */
)
{
    float fXScale = (float)imgSrcCols / (float)imgNetCols;
    float fYScale = (float)imgSrcRows / (float)imgNetRows;

    TfLiteTensor* psObjectTensor;

    size_t tensorSizeSegMap;
    float fObjectQScale;
    int i32ObjectQZeroPoint;
    float fMaskQScale;
    int i32MaskQZeroPoint;
    std::forward_list<Detection> sDetections;

    psObjectTensor = m_model->GetOutputTensor(MODEL_OUTPUT_OBJECT_TENSOR);

    fObjectQScale = ((TfLiteAffineQuantization *)(psObjectTensor->quantization.params))->scale->data[0];
    i32ObjectQZeroPoint = ((TfLiteAffineQuantization *)(psObjectTensor->quantization.params))->zero_point->data[0];

    TfLiteIntArray *psObjectShape = m_model->GetOutputShape(MODEL_OUTPUT_OBJECT_TENSOR);

    if(psObjectShape->data[2] != MODEL_OUTPUT_ANCHOR_BOXES)
    {
        printf("RunPostProcessing(): error on output tensor shape \n");
        return;
    }

    //Eigen matix map default is Colmnt-Major. But C/C++ array is Row-Major
    // For 192 model. it would be C++[84, 756] to Eigen[756 ,84]
    Eigen::Map<MatrixXint8> tObjectMatInt8(psObjectTensor->data.int8, psObjectShape->data[2], psObjectShape->data[1]);

    // find reliable boxes according to the confidence score and threshold, and store them in sDetections for latter NMS and mask processing
    find_realiable_box(tObjectMatInt8, fObjectQScale, i32ObjectQZeroPoint, m_threshold, sDetections);

    CalculateNMS(sDetections, MODEL_OUTPUT_CLASS, 0.45);

    resultsOut.clear();

    float score = 0.0;
    int cls= 0;

    for (auto box=sDetections.begin(); box != sDetections.end(); ++box) {

        score = box->prob[box->cls];

        if(score > 0)
        {
            struct S_DETECTION_BOX detectBox;

            detectBox.x = (box->bbox.cx - (box->bbox.w / 2.0f)) * fXScale;
            detectBox.y = (box->bbox.cy - (box->bbox.h / 2.0f)) * fYScale;
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

} /* namespace YOLOv11nODPostProcessing */
} /* namespace app */
} /* namespace arm */

