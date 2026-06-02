#include <iostream>
#include <cmath>
#include <Eigen/Dense>

#include "YOLOv8nSegPostProc.hpp"
#include "PlatformMath.hpp"

using namespace arm::app::yolov8n_seg;
using MatrixXint8 = Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>;

static int32_t find_realiable_box(
    Eigen::Map<MatrixXint8> *psObjectMat,
    float fQScale,
    int i32QZeroPoint,
    float fThreadhold,
    std::forward_list<Detection>& sDetections
)
{
    MatrixXint8 tConfMat = psObjectMat->block<MODEL_OUTPUT_ANCHOR_BOXES, MODEL_OUTPUT_CLASS>(0, 4);
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
            MatrixXint8 tBoxMat = psObjectMat->block<1, MODEL_OBJECT_BOX_LEN>(r, MODEL_OBJECT_BOX_CX_POS);
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

            det.bbox.x = MODEL_OUTPUT_WIDTH * (cx - (0.5 * w));
            det.bbox.y = MODEL_OUTPUT_HEIGHT * (cy - (0.5 * h));
            det.bbox.w = MODEL_OUTPUT_WIDTH * (w);
            det.bbox.h = MODEL_OUTPUT_HEIGHT * (h);

            //clip the box within the image boundary
            if(det.bbox.x < 0)
                det.bbox.x = 0;

            if(det.bbox.x >= MODEL_OUTPUT_WIDTH)
                det.bbox.x = MODEL_OUTPUT_WIDTH - 1;

            if(det.bbox.y < 0)
                det.bbox.y = 0;

            if(det.bbox.y >= MODEL_OUTPUT_HEIGHT)
                det.bbox.y = MODEL_OUTPUT_HEIGHT - 1;

            if(det.bbox.x + det.bbox.w >  MODEL_OUTPUT_WIDTH)
                det.bbox.w = MODEL_OUTPUT_WIDTH - det.bbox.x;

            if(det.bbox.y + det.bbox.h >  MODEL_OUTPUT_HEIGHT)
                det.bbox.h = MODEL_OUTPUT_HEIGHT - det.bbox.y;

            //printf("The bbox x:%f, y: %f, w:%f, h: %f\n", det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);
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

static float CalculateBoxIntersect(Box& box1, Box& box2)
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

static float CalculateBoxUnion(Box& box1, Box& box2)
{
    float boxes_intersection = CalculateBoxIntersect(box1, box2);
    float boxes_union = box1.w * box1.h + box2.w * box2.h - boxes_intersection;
    return boxes_union;
}


static float CalculateBoxIOU(Box& box1, Box& box2)
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


static void CalculateNMS(std::forward_list<Detection>& detections, int classes, float iouThreshold)
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

static void paint_segment_image(
    image_t &tSegImg,
    std::vector <uint16_t> &tColorMaps,
    Detection *pBox,
    Eigen::MatrixXf &tMaskMat
)
{
    int32_t i32RowScale;
    int32_t i32ColScale;

    int32_t i32BoxX = pBox->bbox.x;
    int32_t i32BoxY = pBox->bbox.y;
    int32_t i32BoxW = pBox->bbox.w;
    int32_t i32BoxH = pBox->bbox.h;
    uint16_t *pu16SegImgData = (uint16_t *)tSegImg.data;

    // calculate the scale between the box size and mask matrix size, which is used to map the mask pixel to the segmentation image pixel
    i32ColScale = tSegImg.w / tMaskMat.cols();
    i32RowScale = tSegImg.h / tMaskMat.rows();

    int y_offset;

    // paint the segmentation image with the mask, box location and class info
    for(int y = i32BoxY; y < (i32BoxY + i32BoxH); y++)
    {
        y_offset = y * tSegImg.w;

        for(int x = i32BoxX; x < (i32BoxX + i32BoxW); x++)
        {
            if(tMaskMat( (y / i32RowScale), (x / i32ColScale)) == 1.0)
            {
                pu16SegImgData[y_offset + x] = tColorMaps[pBox->cls + 1];
            }
        }
    }
}

/*****************************/
namespace arm
{
namespace app
{
namespace yolov8n_seg
{

YOLOv8nSegPostProcessing::YOLOv8nSegPostProcessing(
    arm::app::NNModel *model, float threshold)
    : m_model(model)
    , m_threshold(threshold)
{

}

using MatrixXint8 = Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>;

void YOLOv8nSegPostProcessing::RunPostProcessing(
    std::vector <uint16_t> &colorMaps,
    image_t &segImg
)
{
    TfLiteTensor* psObjectTensor;
    TfLiteTensor* psMaskTensor;

    size_t tensorSizeSegMap;
    float fObjectQScale;
    int i32ObjectQZeroPoint;
    float fMaskQScale;
    int i32MaskQZeroPoint;
    std::forward_list<Detection> sDetections;

    psObjectTensor = m_model->GetOutputTensor(MODEL_OUTPUT_OBJECT_TENSOR);
    psMaskTensor = m_model->GetOutputTensor(MODEL_OUTPUT_MASK_TENSOR);

    fObjectQScale = ((TfLiteAffineQuantization *)(psObjectTensor->quantization.params))->scale->data[0];
    i32ObjectQZeroPoint = ((TfLiteAffineQuantization *)(psObjectTensor->quantization.params))->zero_point->data[0];

    fMaskQScale = ((TfLiteAffineQuantization *)(psMaskTensor->quantization.params))->scale->data[0];
    i32MaskQZeroPoint = ((TfLiteAffineQuantization *)(psMaskTensor->quantization.params))->zero_point->data[0];

    TfLiteIntArray *psObjectShape = m_model->GetOutputShape(MODEL_OUTPUT_OBJECT_TENSOR);
    TfLiteIntArray *psMaskShape = m_model->GetOutputShape(MODEL_OUTPUT_MASK_TENSOR);

    if(psObjectShape->data[2] != MODEL_OUTPUT_ANCHOR_BOXES)
    {
        printf("RunPostProcessing(): error on output tensor shape \n");
        return;
    }

    // clear the segmentation image before painting, set the default background color with 0 in the color map, which is black in this sample code
    memset(segImg.data, 0x0, segImg.size);

    //Eigen matix map default is Colmnt-Major. But C/C++ array is Row-Major
    // For 192 model. it would be C++[116, 756] to Eigen[756 ,116]
    Eigen::Map<MatrixXint8> tObjectMatInt8(psObjectTensor->data.int8, psObjectShape->data[2], psObjectShape->data[1]);

    // For 192 model. it would be C++[48, 48, 32] to Eigen[32 ,2304]
    Eigen::Map<MatrixXint8> tProtoMaskMatInt8(psMaskTensor->data.int8, psMaskShape->data[3], (psMaskShape->data[1]* psMaskShape->data[2]));
    Eigen::MatrixXf tProtoMaskMatFloat = (tProtoMaskMatInt8.cast<float>().array() - i32MaskQZeroPoint) * fMaskQScale;
    //std::cout << "Proto mask matrix col 0: " << tProtoMaskMatFloat.col(0) << std::endl;

    // find reliable boxes according to the confidence score and threshold, and store them in sDetections for latter NMS and mask processing
    find_realiable_box(&tObjectMatInt8, fObjectQScale, i32ObjectQZeroPoint, m_threshold, sDetections);

    CalculateNMS(sDetections, MODEL_OUTPUT_CLASS, 0.45);

    float score = 0.0;
    for (auto box=sDetections.begin(); box != sDetections.end(); ++box) {
        score = box->prob[box->cls];
        if(score > 0)
        {
            // generate the mask for each box according to the mask coefficients and proto mask, and then paint the segmentation image with different color for different class

            // For 192 model. the mask coefficient matrix is 1*32, which is from the box anchor index row in the object output tensor and mask coefficient column
            MatrixXint8 tMaskCoefMat = tObjectMatInt8.block<1, MODEL_OBJECT_MASK_LEN>(box->anchorIndex, MODEL_OBJECT_MASK_START_POS);
            //std::cout << tMaskMat.row(0) << std::endl;
            // dequantization for the mask coefficient matrix
            Eigen::MatrixXf tMaskCoefMatFloat = (tMaskCoefMat.cast<float>().array() - i32ObjectQZeroPoint) * fObjectQScale;
            //std::cout << tMaskMatFloat.row(0) << std::endl;
            // matrix multiplication between the mask coefficient and proto mask to get the mask matrix for each box
            Eigen::MatrixXf tMaskMatFloat = tMaskCoefMatFloat * tProtoMaskMatFloat;

            // apply sigmoid function and binarization on the mask matrix
            for(int i = 0; i < tMaskMatFloat.cols(); i ++)
            {
                //binarization mask matrix
                float fTemp = arm::app::math::MathUtils::SigmoidF32(tMaskMatFloat(0, i));
                if( fTemp > 0.5)
                    tMaskMatFloat(0, i) = 1.0;
                else
                    tMaskMatFloat(0, i) = 0.0;
            }

            // reshape the mask matrix to 48*48, and paint the segmentation image
            Eigen::MatrixXf tMaskMatReshapeFloat = tMaskMatFloat.reshaped<Eigen::RowMajor>(psMaskShape->data[1], psMaskShape->data[2]);

            Detection tBox = { *box };
            // paint the segmentation image with the mask, box location and class info
            paint_segment_image(segImg, colorMaps, &tBox, tMaskMatReshapeFloat);
        }
    }
}


} /* namespace yolov8n_seg */
} /* namespace app */
} /* namespace arm */
