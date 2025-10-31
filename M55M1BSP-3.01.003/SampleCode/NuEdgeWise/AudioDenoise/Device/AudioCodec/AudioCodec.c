/**************************************************************************//**
 * @file     AudioCodec.c
 * @version  V1.00
 * @brief    AudioCodec functions
 *
 * @copyright (C) 2024 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include "NuMicro.h"
#include "AudioCodec.h"
#include "Codec.h"

#include "pmu_counter.h"

#define PDMA_BUF_ALIGN 	    	(32)

typedef struct dma_desc_t
{
    uint32_t ctl;
    uint32_t src;
    uint32_t dest;
    uint32_t offset;
} DMA_DESC_T;

NVT_DTCM static DMA_DESC_T PDMA_TXDESC[2], PDMA_RXDESC[2];
static S_AUDIOCODEC_RES *s_psAudioCodecRes = NULL;

void PDMA0_IRQHandler(void)
{
    uint32_t u32Status = PDMA_GET_INT_STATUS(PDMA0);

    if (u32Status & PDMA_INTSTS_TDIF_Msk)	//Transfer done interrupt
    {
		uint32_t u32TDStatus = PDMA_GET_TD_STS(PDMA0);

        if (u32TDStatus & PDMA_TDSTS_TDIF1_Msk)             /* channel 1 done for PDMA TX*/
        {
			S_AUDIO_BUF_CTRL *psPlayBufCtrl = NULL;

			if((s_psAudioCodecRes) && (s_psAudioCodecRes->psPlayBufCtrl))
				psPlayBufCtrl = s_psAudioCodecRes->psPlayBufCtrl;

			if(psPlayBufCtrl)
			{
				uint32_t u32BlockSamplesBytes = psPlayBufCtrl->u32PDMABlockSamples * psPlayBufCtrl->u32Channels * sizeof(int16_t);
				//Copy play buffer data to PDMA buffer
				if (PDMA0->CURSCAT[1] == (uint32_t)&PDMA_TXDESC[0])
				{
					AudioBufCtrl_Read(psPlayBufCtrl, (int16_t *)s_psAudioCodecRes->u32PDMAPingPongTXBuf_Aligned[0], psPlayBufCtrl->u32PDMABlockSamples);
					AudioBufCtrl_Pop(psPlayBufCtrl, psPlayBufCtrl->u32PDMABlockSamples);
#if (NVT_DCACHE_ON == 1)
					SCB_CleanDCache_by_Addr((void *)(s_psAudioCodecRes->u32PDMAPingPongTXBuf_Aligned[0]), u32BlockSamplesBytes);
#endif
				}
				else
				{
					AudioBufCtrl_Read(psPlayBufCtrl, (int16_t *)s_psAudioCodecRes->u32PDMAPingPongTXBuf_Aligned[1], psPlayBufCtrl->u32PDMABlockSamples);
					AudioBufCtrl_Pop(psPlayBufCtrl, psPlayBufCtrl->u32PDMABlockSamples);
#if (NVT_DCACHE_ON == 1)
					SCB_CleanDCache_by_Addr((void *)(s_psAudioCodecRes->u32PDMAPingPongTXBuf_Aligned[1]), u32BlockSamplesBytes);
#endif
				}
			}
            PDMA_CLR_TD_FLAG(PDMA0, PDMA_TDSTS_TDIF1_Msk);
		}

        if ( u32TDStatus & PDMA_TDSTS_TDIF2_Msk)             /* channel 2 done for PDMA RX*/
        {
			S_AUDIO_BUF_CTRL *psRecBufCtrl = NULL;

			if((s_psAudioCodecRes) && (s_psAudioCodecRes->psRecBufCtrl))
				psRecBufCtrl = s_psAudioCodecRes->psRecBufCtrl;

			if(psRecBufCtrl)
			{
				uint32_t u32BlockSamplesBytes = psRecBufCtrl->u32PDMABlockSamples * psRecBufCtrl->u32Channels * sizeof(int16_t);

				if (PDMA0->CURSCAT[2] == (uint32_t)&PDMA_RXDESC[0])
				{
#if (NVT_DCACHE_ON == 1)
					SCB_InvalidateDCache_by_Addr((void *)(s_psAudioCodecRes->u32PDMAPingPongRXBuf_Aligned[0]), u32BlockSamplesBytes);
#endif
					AudioBufCtrl_Push(psRecBufCtrl, (int16_t *)s_psAudioCodecRes->u32PDMAPingPongRXBuf_Aligned[0]);
				}
				else
				{
#if (NVT_DCACHE_ON == 1)
					SCB_InvalidateDCache_by_Addr((void *)(s_psAudioCodecRes->u32PDMAPingPongRXBuf_Aligned[1]), u32BlockSamplesBytes);
#endif
					AudioBufCtrl_Push(psRecBufCtrl, (int16_t *)s_psAudioCodecRes->u32PDMAPingPongRXBuf_Aligned[1]);
				}
			}
            PDMA_CLR_TD_FLAG(PDMA0, PDMA_TDSTS_TDIF2_Msk);
		}

	}
	else
    {
		printf("unknown interrupt, status=0x%x!!\n", u32Status);
	}
}

/* Configure PDMA to Scatter Gather mode */
static void PDMA_Init(
    uint32_t u32BlockSamples,
    E_AUDIOCODEC_CHANNEL eChannel,
	bool bEnableRec,
	bool bEnablePlay,
	uint32_t u32PDMARXAddr0,
	uint32_t u32PDMARXAddr1,
	uint32_t u32PDMATXAddr0,
	uint32_t u32PDMATXAddr1	
)
{
	uint32_t u32TransCnt = (u32BlockSamples * eChannel * sizeof(int16_t)) / sizeof(uint32_t);  //PDMA transfer bytes / PDMA transfer width
	
	if(bEnablePlay)
	{
		/* Tx description */
		PDMA_TXDESC[0].ctl = ((u32TransCnt - 1) << PDMA_DSCT_CTL_TXCNT_Pos) | PDMA_WIDTH_32 | PDMA_SAR_INC | PDMA_DAR_FIX | PDMA_REQ_SINGLE | PDMA_OP_SCATTER;
		PDMA_TXDESC[0].src =  u32PDMATXAddr0;
		PDMA_TXDESC[0].dest = (uint32_t)&I2S0->TXFIFO;
		PDMA_TXDESC[0].offset = (uint32_t)&PDMA_TXDESC[1];

		PDMA_TXDESC[1].ctl = ((u32TransCnt - 1) << PDMA_DSCT_CTL_TXCNT_Pos) | PDMA_WIDTH_32 | PDMA_SAR_INC | PDMA_DAR_FIX | PDMA_REQ_SINGLE | PDMA_OP_SCATTER;
		PDMA_TXDESC[1].src = u32PDMATXAddr1;
		PDMA_TXDESC[1].dest = (uint32_t)&I2S0->TXFIFO;
		PDMA_TXDESC[1].offset = (uint32_t)&PDMA_TXDESC[0];   /* Link to first description */
	}
		
	if(bEnableRec)
	{
		/* Rx description */
		PDMA_RXDESC[0].ctl = ((u32TransCnt - 1) << PDMA_DSCT_CTL_TXCNT_Pos) | PDMA_WIDTH_32 | PDMA_SAR_FIX | PDMA_DAR_INC | PDMA_REQ_SINGLE | PDMA_OP_SCATTER;
		PDMA_RXDESC[0].src = (uint32_t)&I2S0->RXFIFO;
		PDMA_RXDESC[0].dest = u32PDMARXAddr0;
		PDMA_RXDESC[0].offset = (uint32_t)&PDMA_RXDESC[1];

		PDMA_RXDESC[1].ctl = ((u32TransCnt - 1) << PDMA_DSCT_CTL_TXCNT_Pos) | PDMA_WIDTH_32 | PDMA_SAR_FIX | PDMA_DAR_INC | PDMA_REQ_SINGLE | PDMA_OP_SCATTER;
		PDMA_RXDESC[1].src = (uint32_t)&I2S0->RXFIFO;
		PDMA_RXDESC[1].dest = u32PDMARXAddr1;
		PDMA_RXDESC[1].offset = (uint32_t)&PDMA_RXDESC[0];   /* Link to first description */
	}

	uint32_t u32CHMask = 0;
	
	//Channel 1 for TX
	if(bEnablePlay)
		u32CHMask |= BIT1;

	//Channel 2 for RX
	if(bEnableRec)
		u32CHMask |= BIT2;
	
    /* Open PDMA channel 1 for I2S TX and channel 2 for I2S RX */
    PDMA_Open(PDMA0, u32CHMask);

    /* Configure PDMA transfer mode */
	if(bEnablePlay)
	{
		PDMA_SetTransferMode(PDMA0, 1, PDMA_I2S0_TX, 1, (uint32_t)&PDMA_TXDESC[0]);
		/* Enable PDMA channel 1 interrupt */
		PDMA_EnableInt(PDMA0, 1, PDMA_INT_TRANS_DONE);
	}

	if(bEnableRec)
	{
		PDMA_SetTransferMode(PDMA0, 2, PDMA_I2S0_RX, 1, (uint32_t)&PDMA_RXDESC[0]);
		/* Enable PDMA channel 2 interrupt */
		PDMA_EnableInt(PDMA0, 2, PDMA_INT_TRANS_DONE);
	}

    NVIC_EnableIRQ(PDMA0_IRQn);
}

void AudioCodec_Delay(uint32_t u32MilliSec)
{
    uint64_t u64WaiteCycles = pmu_get_systick_Count();

    u64WaiteCycles += (SystemCoreClock / 1000) * u32MilliSec;

    while (pmu_get_systick_Count() <= u64WaiteCycles)
    {
        __NOP();
    }
}

S_AUDIOCODEC_RES *AudioCodec_Init(
    E_AUDIOCODEC_SAMPLERATE eSampleRate,
    E_AUDIOCODEC_CHANNEL eChannel,
    uint32_t u32BlockSamples,
    uint32_t u32BlockCounts,
	bool bEnableRec,
	bool bEnablePlay		
)
{
    int i32Ret = 0;
	
	S_AUDIOCODEC_RES *psAudioCodecRes = NULL;
	S_AUDIO_BUF_CTRL *psRecBufCtrl = NULL;
	S_AUDIO_BUF_CTRL *psPlayBufCtrl = NULL;
	S_CODEC_INFO *psCodecInfo = NULL;	
    int16_t *pi16PDMAPingPongTXBuf0 = NULL;
    int16_t *pi16PDMAPingPongTXBuf1 = NULL;
    int16_t *pi16PDMAPingPongRXBuf0 = NULL;
    int16_t *pi16PDMAPingPongRXBuf1 = NULL;
    uint32_t u32PDMAPingPongTXBuf0_Aligned = 0;
    uint32_t u32PDMAPingPongTXBuf1_Aligned = 0;
    uint32_t u32PDMAPingPongRXBuf0_Aligned = 0;
    uint32_t u32PDMAPingPongRXBuf1_Aligned = 0;
	
    /* Unlock protected registers */
    SYS_UnlockReg();

    /* Select source from HXT(24MHz) */
//    CLK_SetModuleClock(I2S0_MODULE, CLK_I2SSEL_I2S0SEL_HXT, CLK_I2SDIV_I2S0DIV(1));
    CLK_SetModuleClock(I2S0_MODULE, CLK_I2SSEL_I2S0SEL_HXT, 0);

    /* Enable I2S0 module clock */
    CLK_EnableModuleClock(I2S0_MODULE);

    /* Enable PDMA0 module clock */
    CLK_EnableModuleClock(PDMA0_MODULE);

    /* Init codec device */
    psCodecInfo = &g_sCodecNAU8822;
    if(psCodecInfo->pfnInitCodec(0) != 0)
    {
        printf("Init audio codec device failed \n");
        goto AudioCodec_Init_Done;
    }

    if(psCodecInfo->pfnCofigCodecSR(eSampleRate, eChannel) != 0)
    {
        printf("Unable configure audio codec device \n");
        goto AudioCodec_Init_Done;
    }
	
    /* Unlock protected registers */
    SYS_LockReg();

    /* Set multi-function pins for I2S0 */
    SET_I2S0_BCLK_PI6();
    SET_I2S0_MCLK_PI7();
    SET_I2S0_DI_PI8();
    SET_I2S0_DO_PI9();
    SET_I2S0_LRCK_PI10();

    /* Enable I2S0 clock pin (PI6) schmitt trigger */
    PI->SMTEN |= GPIO_SMTEN_SMTEN6_Msk;

    /* Set JK-EN low to enable phone jack on NuMaker board. */
    SET_GPIO_PD1();
    GPIO_SetMode(PD, BIT1, GPIO_MODE_OUTPUT);
    PD1 = 0;

    if(eChannel == eAUDIOCODEC_CHANNEL_MONO)
    {
        /* Open I2S0 interface and set to slave mode, mono channel, I2S format */
        I2S_Open(I2S0, I2S_MODE_SLAVE, eSampleRate, I2S_DATABIT_16, I2S_MONO, I2S_FORMAT_I2S);
        /* NAU8822 will store data in left channel */
        I2S_SET_MONO_RX_CHANNEL(I2S0, I2S_MONO_LEFT);

    }
    else
    {
        /* Open I2S0 interface and set to slave mode, stereo channel, I2S format */
        I2S_Open(I2S0, I2S_MODE_SLAVE, eSampleRate, I2S_DATABIT_16, I2S_STEREO, I2S_FORMAT_I2S);
    }

    /* Set MCLK and enable MCLK */
    I2S_EnableMCLK(I2S0, 12000000);

	//allocate PDMA transfer buffer
	if(bEnableRec)
	{

		pi16PDMAPingPongRXBuf0 =  malloc((u32BlockSamples * eChannel * sizeof(int16_t)) + (PDMA_BUF_ALIGN - 1));
		pi16PDMAPingPongRXBuf1 =  malloc((u32BlockSamples * eChannel * sizeof(int16_t)) + (PDMA_BUF_ALIGN - 1));

		if((pi16PDMAPingPongRXBuf0 == NULL) || (pi16PDMAPingPongRXBuf1 == NULL))
		{
			printf("Unable allocate audio PDMA buffer for record \n");
			goto AudioCodec_Init_Done;
		}

		u32PDMAPingPongRXBuf0_Aligned = (uint32_t)pi16PDMAPingPongRXBuf0 + (PDMA_BUF_ALIGN - 1);
		u32PDMAPingPongRXBuf0_Aligned &= ~(PDMA_BUF_ALIGN - 1);
		u32PDMAPingPongRXBuf1_Aligned = (uint32_t)pi16PDMAPingPongRXBuf1 + (PDMA_BUF_ALIGN - 1);
		u32PDMAPingPongRXBuf1_Aligned &= ~(PDMA_BUF_ALIGN - 1);
	}

	
	if(bEnablePlay)
	{
		pi16PDMAPingPongTXBuf0 =  malloc((u32BlockSamples * eChannel * sizeof(int16_t)) + (PDMA_BUF_ALIGN - 1));
		pi16PDMAPingPongTXBuf1 =  malloc((u32BlockSamples * eChannel * sizeof(int16_t)) + (PDMA_BUF_ALIGN - 1));

		if((pi16PDMAPingPongTXBuf0 == NULL) || (pi16PDMAPingPongTXBuf1 == NULL))
		{
			printf("Unable allocate audio PDMA buffer for play \n");
			goto AudioCodec_Init_Done;
		}

		u32PDMAPingPongTXBuf0_Aligned = (uint32_t)pi16PDMAPingPongTXBuf0 + (PDMA_BUF_ALIGN - 1);
		u32PDMAPingPongTXBuf0_Aligned &= ~(PDMA_BUF_ALIGN - 1);
		u32PDMAPingPongTXBuf1_Aligned = (uint32_t)pi16PDMAPingPongTXBuf1 + (PDMA_BUF_ALIGN - 1);
		u32PDMAPingPongTXBuf1_Aligned &= ~(PDMA_BUF_ALIGN - 1);
	}

	//Allocate audio buffer ctrl
	if(bEnableRec)
	{
		psRecBufCtrl = AudioBufCtrl_Create(eChannel, u32BlockSamples, u32BlockCounts);
		if(psRecBufCtrl == NULL){
			printf("Unable allocate audio buffer ctrl for record \n");
			goto AudioCodec_Init_Done;
		}
	}

	if(bEnablePlay)
	{
		psPlayBufCtrl = AudioBufCtrl_Create(eChannel, u32BlockSamples, u32BlockCounts);
		if(psPlayBufCtrl == NULL){
			printf("Unable allocate audio buffer ctrl for play \n");
			goto AudioCodec_Init_Done;
		}
	}

	psAudioCodecRes = malloc(sizeof(S_AUDIOCODEC_RES));
	if(psAudioCodecRes)
	{
		psAudioCodecRes->eSampleRate = eSampleRate;
		psAudioCodecRes->eChannel = eChannel;
		psAudioCodecRes->psRecBufCtrl = psRecBufCtrl;
		psAudioCodecRes->psPlayBufCtrl = psPlayBufCtrl;
		psAudioCodecRes->pi16PDMAPingPongRXBuf[0] = pi16PDMAPingPongRXBuf0;
		psAudioCodecRes->pi16PDMAPingPongRXBuf[1] = pi16PDMAPingPongRXBuf1;
		psAudioCodecRes->pi16PDMAPingPongTXBuf[0] = pi16PDMAPingPongTXBuf0;
		psAudioCodecRes->pi16PDMAPingPongTXBuf[1] = pi16PDMAPingPongTXBuf1;
		psAudioCodecRes->u32PDMAPingPongRXBuf_Aligned[0] = u32PDMAPingPongRXBuf0_Aligned;
		psAudioCodecRes->u32PDMAPingPongRXBuf_Aligned[1] = u32PDMAPingPongRXBuf1_Aligned;
		psAudioCodecRes->u32PDMAPingPongTXBuf_Aligned[0] = u32PDMAPingPongTXBuf0_Aligned;
		psAudioCodecRes->u32PDMAPingPongTXBuf_Aligned[1] = u32PDMAPingPongTXBuf1_Aligned;
	}

	s_psAudioCodecRes = psAudioCodecRes;

	//Init PDMA
	PDMA_Init(
		u32BlockSamples,
		eChannel,
		bEnableRec,
		bEnablePlay,
		u32PDMAPingPongRXBuf0_Aligned,
		u32PDMAPingPongRXBuf1_Aligned,
		u32PDMAPingPongTXBuf0_Aligned,
		u32PDMAPingPongTXBuf1_Aligned	
	);

	if(bEnableRec)
	{
		/* Enable I2S Rx function */
		I2S_ENABLE_RXDMA(I2S0);
		I2S_ENABLE_RX(I2S0);
	}
		
	if(bEnablePlay)
	{
		/* Enable I2S Tx function */
		I2S_ENABLE_TXDMA(I2S0);
		I2S_ENABLE_TX(I2S0);
	}
	
	
AudioCodec_Init_Done:

	if(psAudioCodecRes == NULL)
	{
		if(pi16PDMAPingPongTXBuf0)
			free(pi16PDMAPingPongTXBuf0);
		if(pi16PDMAPingPongTXBuf1)
			free(pi16PDMAPingPongTXBuf1);
		if(pi16PDMAPingPongRXBuf0)
			free(pi16PDMAPingPongRXBuf0);
		if(pi16PDMAPingPongRXBuf1)
			free(pi16PDMAPingPongRXBuf1);

		if(psRecBufCtrl)
			AudioBufCtrl_Release(psRecBufCtrl);
		if(psPlayBufCtrl)
			AudioBufCtrl_Release(psPlayBufCtrl);
	}

	return psAudioCodecRes;

}

void AudioCodec_UnInit(S_AUDIOCODEC_RES *psAudioCodecRes)
{
	if(psAudioCodecRes == NULL)
		return;
	
	NVIC_DisableIRQ(PDMA0_IRQn);
	I2S_Close(I2S0);
    PDMA_Close(PDMA0);

	s_psAudioCodecRes = NULL;
	
	if(psAudioCodecRes->pi16PDMAPingPongTXBuf[0])
		free(psAudioCodecRes->pi16PDMAPingPongTXBuf[0]);
	if(psAudioCodecRes->pi16PDMAPingPongTXBuf[1])
		free(psAudioCodecRes->pi16PDMAPingPongTXBuf[1]);
	if(psAudioCodecRes->pi16PDMAPingPongRXBuf[0])
		free(psAudioCodecRes->pi16PDMAPingPongRXBuf[0]);
	if(psAudioCodecRes->pi16PDMAPingPongRXBuf[1])
		free(psAudioCodecRes->pi16PDMAPingPongRXBuf[1]);

	if(psAudioCodecRes->psRecBufCtrl)
		AudioBufCtrl_Release(psAudioCodecRes->psRecBufCtrl);
		
	if(psAudioCodecRes->psPlayBufCtrl)
		AudioBufCtrl_Release(psAudioCodecRes->psPlayBufCtrl);

	free(psAudioCodecRes);
	return;
}

int32_t AudioCodec_SendPCMBlockData(
	S_AUDIOCODEC_RES *psAudioCodecRes,
	int16_t *pi16PCMBlockData
)
{
	int32_t i32Ret;
	S_AUDIO_BUF_CTRL *psPlayBufCtrl = NULL;

	if(psAudioCodecRes == NULL)
		return 0;

	psPlayBufCtrl = psAudioCodecRes->psPlayBufCtrl;
	
	if(psPlayBufCtrl == NULL)
		return 0;

	i32Ret = AudioBufCtrl_Push(psPlayBufCtrl, pi16PCMBlockData);

	if(i32Ret != 0)
		return 0;

	return psPlayBufCtrl->u32PDMABlockSamples;
}

int32_t AudioCodec_RecvPCMBlockData(
	S_AUDIOCODEC_RES *psAudioCodecRes,
	int16_t *pi16PCMBlockData
)
{
	int32_t i32Ret;
	S_AUDIO_BUF_CTRL *psRecBufCtrl = NULL;

	if(psAudioCodecRes == NULL)
		return 0;

	psRecBufCtrl = psAudioCodecRes->psRecBufCtrl;
	
	if(psRecBufCtrl == NULL)
		return 0;

	i32Ret = AudioBufCtrl_Read(psRecBufCtrl, pi16PCMBlockData, psRecBufCtrl->u32PDMABlockSamples);

	if(i32Ret != 0)
		return 0;

	AudioBufCtrl_Pop(psRecBufCtrl, psRecBufCtrl->u32PDMABlockSamples);
	return psRecBufCtrl->u32PDMABlockSamples;
}


