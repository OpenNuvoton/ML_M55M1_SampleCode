/**************************************************************************//**
 * @file     ModelFileReader.h
 * @version  V1.00
 * @brief    Read model file from SD card function
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef __MODEL_FILE_READER_H__
#define __MODEL_FILE_READER_H__

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
ModelFileReader_Initialize(
    PCSTR               pszOutFileName
);

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
INT32
ModelFileReader_ReadData(
    BYTE               *pbyData,
    INT32              i32DataSize
);

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
BOOL
ModelFileReader_Finish(VOID);

//-----------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------
INT32
ModelFileReader_FileSize(VOID);

VOID
ModelFileReader_Rewind(VOID);


#ifdef  __cplusplus
}
#endif

#endif
