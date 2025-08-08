/**************************************************************************//**
 * @file     Codec.h
 * @version  V1.00
 * @brief    Audio codec device driver
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 *****************************************************************************/
#ifndef __CODEC_H__
#define __CODEC_H__

#include "NuMicro.h"

#include "AudioCodec.h"

#define Codec_Delay_MilliSec   AudioCodec_Delay

//Init codec devcie
typedef int32_t (*PFN_INIT_CODEC_FUNC)(uint32_t u32Param);
//Config codec sample rate and channel
typedef int32_t (*PFN_CONFIG_CODEC_SR_FUNC)(uint32_t u32SampleRate, uint32_t u32Channel);

typedef struct s_code_info
{
    char        m_strName[16];
    PFN_INIT_CODEC_FUNC    pfnInitCodec;
    PFN_CONFIG_CODEC_SR_FUNC pfnCofigCodecSR;
} S_CODEC_INFO;

extern S_CODEC_INFO g_sCodecNAU8822;

#endif
