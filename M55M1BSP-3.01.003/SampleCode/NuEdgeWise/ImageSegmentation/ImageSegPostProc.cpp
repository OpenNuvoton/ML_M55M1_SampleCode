#include "ImageSegPostProc.hpp"

#include <cmath>

using namespace arm::app::image_seg;

/*****************************/
namespace arm
{
namespace app
{
namespace image_seg
{

ImageSegPostProcessing::ImageSegPostProcessing(
	arm::app::ImageSegModel *model)
    :   m_model(model)
{

}

void ImageSegPostProcessing::RunPostProcessing(
	std::vector <uint16_t> &colorMaps,
	image_t &segImg
)
{
	TfLiteTensor* psSegMapTensor;
	size_t tensorSizeSegMap;
	uint16_t *pu16SegImgData = (uint16_t *)segImg.data;
	float qScale;
    int qZeroPoint;

 	
	psSegMapTensor = m_model->GetOutputTensor(0);	
	tensorSizeSegMap = psSegMapTensor->bytes;
	qScale = ((TfLiteAffineQuantization *)(psSegMapTensor->quantization.params))->scale->data[0];
    qZeroPoint = ((TfLiteAffineQuantization *)(psSegMapTensor->quantization.params))->zero_point->data[0];
	
	if(tensorSizeSegMap != MODEL_OUTPUT_WIDTH * MODEL_OUTPUT_HEIGHT * MODEL_OUTPUT_CLASS)
	{
		printf("RunPostProcessing(): error tensor size not match \n");
		return;
	}

	int x,y,z;
	int pos;
	int y_offset;	
	float fMaxConf;
	float fConf;
	int cls;
	int8_t *tensorOutputData = psSegMapTensor->data.int8;

	for(y = 0; y < MODEL_OUTPUT_HEIGHT; y++)
	{
		y_offset = y * MODEL_OUTPUT_WIDTH;

		for(x = 0; x < MODEL_OUTPUT_WIDTH; x++)
		{
			fMaxConf = 0;
			cls = 0;
			pos = (y_offset + x) * MODEL_OUTPUT_CLASS;
			
			for(z = 0; z < MODEL_OUTPUT_CLASS; z++)
			{
				fConf = qScale * (static_cast<float>(tensorOutputData[pos  + z]) - qZeroPoint);

				if((z == 0) || (fConf > fMaxConf))
				{
					fMaxConf = fConf;
					cls = z;
				}
			}

			pu16SegImgData[y_offset + x] = colorMaps[cls];
		}
	}
}

	
} /* namespace image_seg */
} /* namespace app */
} /* namespace arm */
