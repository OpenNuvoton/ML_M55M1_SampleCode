/**************************************************************************//**
 * @file     ModelFileReader.c
 * @version  V1.00
 * @brief    Read model file from SD card function
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "NuMicro.h"
#include "ModelFileReader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ff.h"

static FIL	modelFileObject;

//----------------------------------------------------------------------------
// Public functions
//----------------------------------------------------------------------------
BOOL
ModelFileReader_Initialize(
    PCSTR               pszOutFileName
)
{
    FRESULT res;

    res = f_open(&modelFileObject, (const TCHAR *)pszOutFileName, FA_OPEN_EXISTING | FA_READ);      //USBH:0 , SD0: 1

    if (res != FR_OK)
    {
        printf("Open file error!\n");
        return FALSE;
    }

	return TRUE;
}

BOOL
ModelFileReader_Finish(VOID)
{
    if (0 != f_close(&modelFileObject))
        return FALSE;

    return TRUE;
}

INT32
ModelFileReader_ReadData(
    BYTE               *pbyData,
    INT32              i32DataSize
)
{
    FRESULT res;
	size_t  ReturnSize;

	if (f_eof(&modelFileObject))
		return -1;
		
	res = f_read(&modelFileObject, pbyData, i32DataSize, &ReturnSize);
	if(res != FR_OK)
	{
		return -2;
	}

	return ReturnSize;
}

INT32
ModelFileReader_FileSize(VOID)
{
	return f_size(&modelFileObject);
}

VOID
ModelFileReader_Rewind(VOID)
{
	f_rewind(&modelFileObject);
}
