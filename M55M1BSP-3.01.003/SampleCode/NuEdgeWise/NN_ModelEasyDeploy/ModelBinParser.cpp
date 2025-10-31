/**************************************************************************//**
 * @file     ModelBinParser.cpp
 * @version  V1.00
 * @brief    Parser model binary function
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <stdio.h>
#include <cinttypes>
#include <cstring>
#include <vector>
#include <string>

#include "cJSON.h"
#include "ModelBinParser.hpp"

static cJSON *s_psModelJsonObj = NULL;

int ModelBinParser_Init(char *pu8ModelBin)
{
	char *strJson;

    uint32_t u32JsonStrLen = std::strlen(pu8ModelBin);

    s_psModelJsonObj = cJSON_Parse(pu8ModelBin);
	if(s_psModelJsonObj == NULL)
		return -1;
	
	return 0;
}

int ModelBinParser_GetTfliteOffset(void)
{
	if(s_psModelJsonObj == NULL)
		return -1;

    const cJSON *psTfliteOffset = cJSON_GetObjectItemCaseSensitive(s_psModelJsonObj, "model_offset");
	if(psTfliteOffset)
	{
		if(!cJSON_IsNumber(psTfliteOffset))
			return -2;

		return (int)psTfliteOffset->valuedouble; 
	}
	return -3;
}

int ModelBinParser_GetTfliteSize(void)
{
	if(s_psModelJsonObj == NULL)
		return -1;

    const cJSON *psTfliteSize = cJSON_GetObjectItemCaseSensitive(s_psModelJsonObj, "model_size");
	if(psTfliteSize)
	{
		if(!cJSON_IsNumber(psTfliteSize))
			return -2;

		return (int)psTfliteSize->valuedouble; 
	}
	return -3;
}

int ModelBinParser_GetLabels(std::vector<std::string> &labels)
{
	if(s_psModelJsonObj == NULL)
		return -1;

	const cJSON *psLabelsObj = cJSON_GetObjectItemCaseSensitive(s_psModelJsonObj, "class");
	if(!psLabelsObj)
	{
		return -2;
	}
	
	int i32ArraySize = cJSON_GetArraySize(psLabelsObj);
	
	int i;
    labels.clear();
	
	for(i = 0; i < i32ArraySize; i ++)
	{
		const cJSON *psLabelsItemObj= cJSON_GetArrayItem(psLabelsObj, i);
		printf("Add label item %s \n", cJSON_GetStringValue(psLabelsItemObj));
        labels.emplace_back(cJSON_GetStringValue(psLabelsItemObj));
	}
	
	return i32ArraySize;
}