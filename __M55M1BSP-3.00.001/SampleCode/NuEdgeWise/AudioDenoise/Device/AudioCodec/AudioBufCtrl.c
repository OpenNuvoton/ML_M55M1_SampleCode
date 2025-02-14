/**************************************************************************//**
 * @file     AudioBufCtrl.c
 * @version  V1.00
 * @brief    Audio buffer data contorl function
 *
 * @copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "NuMicro.h"
#include "AudioBufCtrl.h"

// Create audio buffer control object
S_AUDIO_BUF_CTRL *AudioBufCtrl_Create(uint32_t u32Channels,  uint32_t u32PDMABlockSamples, uint32_t u32BlockCounts)
{
    int16_t *pi16AudioInBuf = NULL;
    S_AUDIO_BUF_CTRL *psAudioBufCtrl = NULL;

    pi16AudioInBuf = malloc(u32PDMABlockSamples * u32BlockCounts * u32Channels * sizeof(int16_t));

    if(pi16AudioInBuf == NULL)
        return NULL;

    psAudioBufCtrl = malloc(sizeof(S_AUDIO_BUF_CTRL));

    if(psAudioBufCtrl == NULL)
    {
        free(pi16AudioInBuf);
        return NULL;
    }

    psAudioBufCtrl->i32ReadSampleIndex = 0;
    psAudioBufCtrl->i32WriteSampleIndex = 0;
    psAudioBufCtrl->i32TotalSamples = u32PDMABlockSamples * u32BlockCounts;
    psAudioBufCtrl->u32Channels = u32Channels;
    psAudioBufCtrl->pi16AudioInBuf = pi16AudioInBuf;
    psAudioBufCtrl->u32PDMABlockSamples = u32PDMABlockSamples;

    return psAudioBufCtrl;
}

// Push audio data to ring buffer
int AudioBufCtrl_Push(S_AUDIO_BUF_CTRL *psBufCtrl, int16_t *pi16Data)
{
    int32_t i32NextWriteIndex;
    int32_t i32ReadIndex = psBufCtrl->i32ReadSampleIndex;
    int32_t i32WriteIndex = psBufCtrl->i32WriteSampleIndex;
    int32_t i32TotalSamples = psBufCtrl->i32TotalSamples;
    int32_t i32PDMASamples = psBufCtrl->u32PDMABlockSamples;
    uint32_t u32Channels = psBufCtrl->u32Channels;
    int16_t *pi16AudioInBuf = psBufCtrl->pi16AudioInBuf;
    int32_t i32FreeSampleSpace;

    //buffer full, reserved a PDMA block samples to avoid overflow
    if (((i32WriteIndex + i32PDMASamples) % i32TotalSamples) == i32ReadIndex)
        return -1;

    if (i32WriteIndex >= i32ReadIndex)
    {
        i32FreeSampleSpace = i32TotalSamples - (i32WriteIndex - i32ReadIndex);
    }
    else
    {
        i32FreeSampleSpace = i32ReadIndex - i32WriteIndex;
    }

    if (i32FreeSampleSpace < i32PDMASamples)
        return -2;

    i32NextWriteIndex = i32WriteIndex + i32PDMASamples;

#if 0 //defined(NVT_DCACHE_ON)    
    SCB_InvalidateDCache_by_Addr(pi16Data, i32PDMASamples * u32Channels * sizeof(int16_t));
#endif

    if (i32NextWriteIndex >= i32TotalSamples)
    {
        int32_t i32CopySamples = i32TotalSamples - i32WriteIndex;

        memcpy(&pi16AudioInBuf[i32WriteIndex * u32Channels], pi16Data, i32CopySamples * u32Channels * sizeof(int16_t));

        i32NextWriteIndex = i32NextWriteIndex - i32TotalSamples;

        if (i32NextWriteIndex)
        {
            memcpy(&pi16AudioInBuf[0], &pi16Data[i32CopySamples * u32Channels], i32NextWriteIndex * u32Channels * sizeof(int16_t));
        }
    }
    else
    {
        memcpy(&pi16AudioInBuf[i32WriteIndex * u32Channels], pi16Data, i32PDMASamples * u32Channels * sizeof(int16_t));
    }

    psBufCtrl->i32WriteSampleIndex = i32NextWriteIndex;

    return 0;
}

// Read audio data from ring buffer
int AudioBufCtrl_Read(S_AUDIO_BUF_CTRL *psBufCtrl, int16_t *pi16Data, int32_t i32Samples)
{
    int32_t i32NextReadIndex;
    int32_t i32ReadIndex = psBufCtrl->i32ReadSampleIndex;
    int32_t i32WriteIndex = psBufCtrl->i32WriteSampleIndex;
    int32_t i32TotalSamples = psBufCtrl->i32TotalSamples;
    uint32_t u32Channels = psBufCtrl->u32Channels;
    int16_t *pi16AudioInBuf = psBufCtrl->pi16AudioInBuf;
    int32_t i32AvailSampleSpace;

    //empty
    if (i32ReadIndex == i32WriteIndex)
        return -1;

    if (i32WriteIndex > i32ReadIndex)
    {
        i32AvailSampleSpace = i32WriteIndex - i32ReadIndex;
    }
    else
    {
        i32AvailSampleSpace = i32TotalSamples - (i32ReadIndex - i32WriteIndex);
    }

    if (i32AvailSampleSpace < i32Samples)
        return -2;

    i32NextReadIndex = i32ReadIndex + i32Samples;

#if 0 //defined(NVT_DCACHE_ON)    
    SCB_CleanDCache_by_Addr(pi16Data, i32Samples * u32Channels * sizeof(int16_t));
#endif

    if (i32NextReadIndex >= i32TotalSamples)
    {
        int32_t i32CopySamples = i32TotalSamples - i32ReadIndex;

        memcpy(pi16Data, &pi16AudioInBuf[i32ReadIndex * u32Channels],  i32CopySamples * u32Channels * sizeof(int16_t));

        i32NextReadIndex = i32NextReadIndex - i32TotalSamples;

        if (i32NextReadIndex)
        {
            memcpy(&pi16Data[i32CopySamples * u32Channels], &pi16AudioInBuf[0],  i32NextReadIndex * u32Channels * sizeof(int16_t));
        }
    }
    else
    {
        memcpy(pi16Data, &pi16AudioInBuf[i32ReadIndex * u32Channels], i32Samples * u32Channels * sizeof(int16_t));
    }

    return 0;
}

// Pop audio data from ring buffer
int AudioBufCtrl_Pop(S_AUDIO_BUF_CTRL *psBufCtrl, int32_t i32Samples)
{
    int32_t i32NextReadIndex;
    int32_t i32ReadIndex = psBufCtrl->i32ReadSampleIndex;
    int32_t i32WriteIndex = psBufCtrl->i32WriteSampleIndex;
    int32_t i32TotalSamples = psBufCtrl->i32TotalSamples;
    int32_t i32AvailSampleSpace;

    //empty
    if (i32ReadIndex == i32WriteIndex)
        return -1;

    if (i32WriteIndex > i32ReadIndex)
    {
        i32AvailSampleSpace = i32WriteIndex - i32ReadIndex;
    }
    else
    {
        i32AvailSampleSpace = i32TotalSamples - (i32ReadIndex - i32WriteIndex);
    }

    if (i32AvailSampleSpace < i32Samples)
        return -2;

    i32NextReadIndex = i32ReadIndex + i32Samples;

    if (i32NextReadIndex >= i32TotalSamples)
    {
        i32NextReadIndex = i32NextReadIndex - i32TotalSamples;
    }

    psBufCtrl->i32ReadSampleIndex = i32NextReadIndex;

    return 0;
}

// Get available audio data size
int AudioBufCtrl_AvailSamples(S_AUDIO_BUF_CTRL *psBufCtrl)
{
    int32_t i32ReadIndex = psBufCtrl->i32ReadSampleIndex;
    int32_t i32WriteIndex = psBufCtrl->i32WriteSampleIndex;
    int32_t i32TotalSamples = psBufCtrl->i32TotalSamples;
    int32_t i32AvailSampleSpace;

    //empty
    if (i32ReadIndex == i32WriteIndex)
        return 0;

    if (i32WriteIndex > i32ReadIndex)
    {
        i32AvailSampleSpace = i32WriteIndex - i32ReadIndex;
    }
    else
    {
        i32AvailSampleSpace = i32TotalSamples - (i32ReadIndex - i32WriteIndex);
    }

    return i32AvailSampleSpace;
}

// Release audio buffer control object
void AudioBufCtrl_Release(S_AUDIO_BUF_CTRL *psAudioBufCtrl)
{
	if(psAudioBufCtrl == NULL)
		return;

	if(psAudioBufCtrl->pi16AudioInBuf)
		free(psAudioBufCtrl->pi16AudioInBuf);
		
	free(psAudioBufCtrl);
	return;
}