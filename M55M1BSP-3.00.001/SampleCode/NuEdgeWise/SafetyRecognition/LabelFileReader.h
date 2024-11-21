/**************************************************************************//**
 * @file     LabelFileReader.h
 * @version  V1.00
 * @brief    Write label file from SD card function
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef __LABEL_FILE_READER_H__
#define __LABEL_FILE_READER_H__

#ifdef  __cplusplus
extern "C"
{
#endif

//-----------------------------------------------------------------------------
// Type declaration
//-----------------------------------------------------------------------------
typedef signed int          BOOL;
typedef unsigned char       BYTE;
typedef signed int          INT32;
typedef const char         *PCSTR;
typedef unsigned char      *PUINT8;
typedef unsigned short     *PUINT16;
typedef unsigned int       *PUINT32;
typedef unsigned short      UINT16;
typedef unsigned int        UINT32;
typedef void                VOID;

// Read file functions
//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOL
LabelFileReader_Initialize(
    PCSTR               pszOutFileName
);

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
INT32
LabelFileReader_WriteData(
    BYTE               *pbyData,
    INT32              i32DataSize
);

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOL
LabelFileReader_Finish(VOID);

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
INT32
LabelFileReader_FileSize(VOID);

VOID
LabelFileReader_Rewind(VOID);

char *LabelFileReader_ReadLine(
    INT32              *pi32DataSize
);

#ifdef  __cplusplus
}
#endif

#endif
