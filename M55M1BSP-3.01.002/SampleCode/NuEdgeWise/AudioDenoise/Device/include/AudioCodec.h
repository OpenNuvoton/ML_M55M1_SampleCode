/**************************************************************************//**
 * @file     AudioCodec.h
 * @version  V1.00
 * @brief    Audio codec device function
 *
 * @copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#ifndef __AUDIO_CODEC_H__
#define __AUDIO_CODEC_H__

#include <inttypes.h>
#include "AudioBufCtrl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    eAUDIOCODEC_CHANNEL_MONO = 1,
    eAUDIOCODEC_CHANNEL_STEREO = 2,
}E_AUDIOCODEC_CHANNEL;

typedef enum
{
    eAUDIOCODEC_SAMPLERATE_16000 = 16000,
    eAUDIOCODEC_SAMPLERATE_44100 = 44100,
    eAUDIOCODEC_SAMPLERATE_48000 = 48000,
    eAUDIOCODEC_SAMPLERATE_96000 = 96000,
}E_AUDIOCODEC_SAMPLERATE;

typedef struct
{
    E_AUDIOCODEC_SAMPLERATE eSampleRate;				//audio sample rate
    E_AUDIOCODEC_CHANNEL eChannel;		//audio channels
	S_AUDIO_BUF_CTRL *psRecBufCtrl;		//reocrd path buffer control
	S_AUDIO_BUF_CTRL *psPlayBufCtrl;	//play path buffer control
	int16_t *pi16PDMAPingPongTXBuf[2];	//PDMA TX buffer
	uint32_t u32PDMAPingPongTXBuf_Aligned[2];	//alignment address of PDMA TX buffer
	int16_t *pi16PDMAPingPongRXBuf[2];	//PDMA RX buffer
	uint32_t u32PDMAPingPongRXBuf_Aligned[2];	//alignment address of PDMA RX buffer
}S_AUDIOCODEC_RES;

/**
  * @function AudioCodec_Init
  * @brief Initiate audio codec resource
  * @param[in] u32SampleRate: sample rate
  * @param[in] eChannel: channels
  * @param[in] u32BlockSamples: samples of each PDMA block transfer
  * @param[in] u32BlockCounts: blocks of audio play/record buffer
  * @param[in] bEnableRec: enable record path
  * @param[in] bEnablePlay: enable play path
  * @return S_AUDIOCODEC_RES pointer 
  * \hideinitializer
  */

S_AUDIOCODEC_RES *AudioCodec_Init(
    E_AUDIOCODEC_SAMPLERATE eSampleRate,
    E_AUDIOCODEC_CHANNEL eChannel,
    uint32_t u32BlockSamples,
    uint32_t u32BlockCounts,
	bool bEnableRec,
	bool bEnablePlay		
);

/**
  * @function AudioCodec_SendPCMBlockData
  * @brief Send PCM block data to audio codec
  * @param[in] psAudioCodecRes: audio codec resource
  * @param[in] pi16PCMBlockData: source PCM data address
  * @return Send success samples
  * \hideinitializer
  */
	
int32_t AudioCodec_SendPCMBlockData(
	S_AUDIOCODEC_RES *psAudioCodecRes,
	int16_t *pi16PCMBlockData
);

/**
  * @function AudioCodec_RecvPCMBlockData
  * @brief Receive PCM block data from audio codec
  * @param[in] psAudioCodecRes: audio codec resource
  * @param[out] pi16PCMBlockData: destination PCM data address
  * @return Receive success samples
  * \hideinitializer
  */

int32_t AudioCodec_RecvPCMBlockData(
	S_AUDIOCODEC_RES *psAudioCodecRes,
	int16_t *pi16PCMBlockData
);

/**
  * @function AudioCodec_UnInit
  * @brief Finalize and release audio codec resource
  * @param[in] psAudioCodecRes: audio codec resource
  * @return None
  * \hideinitializer
  */
	
void AudioCodec_UnInit(S_AUDIOCODEC_RES *psAudioCodecRes);

/**
  * @function AudioCodec_Delay
  * @brief delay function
  * @param[in] u32MilliSec: milli seconds
  * @return None
  * \hideinitializer
  */

void AudioCodec_Delay(uint32_t u32MilliSec);

#ifdef __cplusplus
}
#endif

#endif


