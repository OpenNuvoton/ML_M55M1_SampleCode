/**************************************************************************//**
 * @file     AudioBufCtrl.h
 * @version  V1.00
 * @brief    Audio buffer data contorl function
 *
 * @copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef __AUDIO_BUF_CTRL_H__
#define __AUDIO_BUF_CTRL_H__

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int32_t i32ReadSampleIndex;
    int32_t i32WriteSampleIndex;
    int32_t i32TotalSamples;
    uint32_t u32Channels;
    int16_t *pi16AudioInBuf;
    uint32_t u32PDMABlockSamples;
} S_AUDIO_BUF_CTRL;

// Create audio buffer control object
S_AUDIO_BUF_CTRL *AudioBufCtrl_Create(uint32_t u32Channels,  uint32_t u32PDMABlockSamples, uint32_t u32BlockCounts);
// Push audio data into audio ring buffer
int AudioBufCtrl_Push(S_AUDIO_BUF_CTRL *psBufCtrl, int16_t *pi16Data);
// Read audio data from audio ring buffer only. Sample read index not update
int AudioBufCtrl_Read(S_AUDIO_BUF_CTRL *psBufCtrl, int16_t *pi16Data, int32_t i32Samples);
// Pop audio data from ring buffer. Only update sample read index
int AudioBufCtrl_Pop(S_AUDIO_BUF_CTRL *psBufCtrl, int32_t i32Samples);
// Get available audio data on ring buffer
int AudioBufCtrl_AvailSamples(S_AUDIO_BUF_CTRL *psBufCtrl);
// Release audio buffer control object
void AudioBufCtrl_Release(S_AUDIO_BUF_CTRL *psAudioBufCtrl);

#ifdef __cplusplus
}
#endif

#endif
