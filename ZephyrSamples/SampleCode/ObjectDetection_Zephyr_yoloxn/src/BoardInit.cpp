/**************************************************************************//**
 * @file     BoardInit.cpp
 * @version  V1.00
 * @brief    Target board initiate function
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <cstdio>

#include "NuMicro.h"
#include "hyperram_code.h"

#define DESIGN_NAME "M55M1"
#define HYPERRAM_SPIM_PORT SPIM0        //For NuMaker-M55M1 board

#if defined(__LOAD_MODEL_FROM_SD__)
static void SDCard0_PinConfig(void)
{
	/* Set multi-function pin for SDH */
    SET_SD0_nCD_PD13();
    SET_SD0_CLK_PE6();
    SET_SD0_CMD_PE7();
    SET_SD0_DAT0_PE2();
    SET_SD0_DAT1_PE3();
    SET_SD0_DAT2_PE4();
    SET_SD0_DAT3_PE5();
}

static void SDCard1_PinConfig(void)
{
	/* Set multi-function pin for SDH */
    SET_SD1_nCD_PA6();
    SET_SD1_CLK_PB6();
    SET_SD1_CMD_PA5();
    SET_SD1_DAT0_PA8();
    SET_SD1_DAT1_PA9();
    SET_SD1_DAT2_PA10();
    SET_SD1_DAT3_PA11();
}
#endif

static void SYS_Init(void)
{
    /* Enable APLL1 clock */
    CLK_EnableAPLL(CLK_APLLCTL_APLLSRC_HXT, FREQ_220MHZ, CLK_APLL1_SELECT);

   /* Update System Core Clock */
    /* User can use SystemCoreClockUpdate() to calculate SystemCoreClock. */
    SystemCoreClockUpdate();

    /* Enable GPIO module clock */
    CLK_EnableModuleClock(GPIOA_MODULE);
    CLK_EnableModuleClock(GPIOB_MODULE);
    CLK_EnableModuleClock(GPIOC_MODULE);
    CLK_EnableModuleClock(GPIOD_MODULE);
    CLK_EnableModuleClock(GPIOE_MODULE);
    CLK_EnableModuleClock(GPIOF_MODULE);
    CLK_EnableModuleClock(GPIOG_MODULE);
    CLK_EnableModuleClock(GPIOH_MODULE);
    CLK_EnableModuleClock(GPIOI_MODULE);
    CLK_EnableModuleClock(GPIOJ_MODULE);

    /* Enable FMC0 module clock to keep FMC clock when CPU idle but NPU running*/
    CLK_EnableModuleClock(FMC0_MODULE);

    /* Enable NPU module clock */
    CLK_EnableModuleClock(NPU0_MODULE);

#if defined(__LOAD_MODEL_FROM_SD__)
#if defined(__NUMAKER_M55M1__)
/* Enable SDH0 module clock source as HCLK and SDH0 module clock divider as 4 */
    CLK_SetModuleClock(SDH0_MODULE, CLK_SDHSEL_SDH0SEL_APLL1_DIV2, CLK_SDHDIV_SDH0DIV(5));
    CLK_EnableModuleClock(SDH0_MODULE);
#endif

#if defined(__NUGESTUREAI_M55M1__)
/* Enable SDH1 module clock source as HCLK and SDH1 module clock divider as 4 */
    CLK_SetModuleClock(SDH1_MODULE, CLK_SDHSEL_SDH1SEL_APLL1_DIV2, CLK_SDHDIV_SDH1DIV(5));
    CLK_EnableModuleClock(SDH1_MODULE);
#endif
#endif

    /* Enable CCAP0 module clock */
    CLK_EnableModuleClock(CCAP0_MODULE);

#if defined(__USE_HYPERRAM__)
    HyperRAM_PinConfig(HYPERRAM_SPIM_PORT);
#endif

#if defined(__LOAD_MODEL_FROM_SD__)
#if defined(__NUMAKER_M55M1__)
    SDCard0_PinConfig();
#endif
#if defined(__NUGESTUREAI_M55M1__)
    SDCard1_PinConfig();
#endif
#endif
}

/**
  * @brief Initiate the hardware resources of board
  * @return 0: Success, <0: Fail
  * @details Initiate clock, UART, NPU, hyperflash/hyperRAM
  * \hideinitializer
  */
int BoardInit(void)
{
    /* Unlock protected registers */
    SYS_UnlockReg();

    SYS_Init();
    SYS_LockReg();                   /* Unlock register lock protect */

#if defined(__USE_HYPERRAM__)
    HyperRAM_Init(HYPERRAM_SPIM_PORT);
    /* Enter direct-mapped mode to run new applications */
    SPIM_HYPER_EnterDirectMapMode(HYPERRAM_SPIM_PORT);
#endif

#if defined(__LOAD_MODEL_FROM_SD__)
    /* SDH open SD card*/
#if defined(__NUMAKER_M55M1__)
    SDH_Open_Disk(SDH0, CardDetect_From_GPIO);
#endif

#if defined(__NUGESTUREAI_M55M1__)
    SDH_Open_Disk(SDH1, CardDetect_From_GPIO);
#endif

#endif

    printf("%s: complete\n", __FUNCTION__);

    /* Print target design info */
    printf("Target system: %s\n", DESIGN_NAME);

    return 0;
}


