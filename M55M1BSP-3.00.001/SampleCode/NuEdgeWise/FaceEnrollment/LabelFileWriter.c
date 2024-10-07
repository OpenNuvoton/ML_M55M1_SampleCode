/**************************************************************************//**
 * @file     LabelFileWriter.c
 * @version  V1.00
 * @brief    Write label file from SD card function
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "NuMicro.h"
#include "LabelFileWriter.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ff.h"

static FIL	labelFileObject;

//----------------------------------------------------------------------------
// Public functions
//----------------------------------------------------------------------------
BOOL
LabelFileWriter_Initialize(
    PCSTR               pszOutFileName
)
{
    FRESULT res;

    res = f_open(&labelFileObject, (const TCHAR *)pszOutFileName, FA_OPEN_APPEND | FA_WRITE);

    if (res != FR_OK)
    {
        printf("Open file error!\n");
        return FALSE;
    }

	return TRUE;
}

BOOL
LabelFileWriter_Finish(VOID)
{
    if (0 != f_close(&labelFileObject))
        return FALSE;

    return TRUE;
}

INT32
LabelFileWriter_WriteData(
    BYTE               *pbyData,
    INT32              i32DataSize
)
{
    FRESULT res;
	size_t  ReturnSize;

	res = f_write(&labelFileObject, pbyData, i32DataSize, &ReturnSize);
	if(res != FR_OK)
	{
		return -1;
	}

	return ReturnSize;
}

INT32
LabelFileWriter_FileSize(VOID)
{
	return f_size(&labelFileObject);
}

VOID
LabelFileWriter_Rewind(VOID)
{
	f_rewind(&labelFileObject);
}
