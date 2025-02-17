/**************************************************************************//**
 * @file     LabelFileReader.c
 * @version  V1.00
 * @brief    Write label file from SD card function
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "NuMicro.h"
#include "LabelFileReader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ff.h"

static FIL	labelFileObject;

#define LABEL_INFO_LINE_BUF_LEN	(4096)
__attribute__((section(".bss.sram.data"), aligned(32))) static char labelInfoLineBuf[LABEL_INFO_LINE_BUF_LEN];


//----------------------------------------------------------------------------
// Public functions
//----------------------------------------------------------------------------
BOOL
LabelFileReader_Initialize(
    PCSTR               pszOutFileName
)
{
    FRESULT res;

    res = f_open(&labelFileObject, (const TCHAR *)pszOutFileName, FA_OPEN_EXISTING | FA_READ);

    if (res != FR_OK)
    {
        printf("Open file error!\n");
        return FALSE;
    }

	return TRUE;
}

BOOL
LabelFileReader_Finish(VOID)
{
    if (0 != f_close(&labelFileObject))
        return FALSE;

    return TRUE;
}

INT32
LabelFileReader_ReadData(
    BYTE               *pbyData,
    INT32              i32DataSize
)
{
    FRESULT res;
	size_t  ReturnSize;

	if (f_eof(&labelFileObject))
		return -1;

	res = f_read(&labelFileObject, pbyData, i32DataSize, &ReturnSize);
	if(res != FR_OK)
	{
		return -1;
	}

	return ReturnSize;
}

char *LabelFileReader_ReadLine(
    INT32              *pi32DataSize
)
{
    BYTE chData;
    INT32 i32Read;
    INT32 i32LineLen = 0;

    while(i32LineLen < LABEL_INFO_LINE_BUF_LEN)
    {
        i32Read = LabelFileReader_ReadData(&chData, 1);

        if(i32Read <= 0)
            break;

        if(chData == '\n')
        {
            break;
        }
        else
        {
            labelInfoLineBuf[i32LineLen] = chData;
        }

        i32LineLen ++;
    }

    if(i32LineLen >= LABEL_INFO_LINE_BUF_LEN)
        printf("Error: Label line buffer is too small \n");

    labelInfoLineBuf[i32LineLen] = '\0';
    *pi32DataSize = i32LineLen;

    return labelInfoLineBuf;
}


INT32
LabelFileReader_FileSize(VOID)
{
	return f_size(&labelFileObject);
}

VOID
LabelFileReader_Rewind(VOID)
{
	f_rewind(&labelFileObject);
}
