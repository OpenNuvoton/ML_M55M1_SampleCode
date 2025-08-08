/**************************************************************************//**
 * @file        hyperflash_code.h
 * @version     V3.00
 * @brief       HyperFlash device driver
 *
 * SPDX-License-Identifier: Apache-2.0
 * @copyright (C) 2019 Nuvoton Technology Corp. All rights reserved.
*****************************************************************************/

#ifndef __HYPER_RAM_CODE_H__
#define __HYPER_RAM_CODE_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    union
    {
        uint32_t u32REG;
        struct
        {
            uint32_t u4Manufacturer         : 4;
            uint32_t u4ColumnAddressBitCount: 4;
            uint32_t u4RowAddressBitCount   : 4;
            uint32_t u3Reserved             : 3;
            uint32_t u2MCPDieAddr           : 2;
            uint32_t : 16;
        };
    } ID0;

    union
    {
        uint32_t u32REG;
        struct
        {
            uint32_t u4DeviceType           : 4;
            uint32_t u12Reserved            : 12;
            uint32_t : 16;
        };
    } ID1;

    union
    {
        uint32_t u32REG;
        struct
        {
            uint32_t u2BurstLength          : 2;
            uint32_t u1HybridBurstEnable    : 1;
            uint32_t u1FixedLatencyEanble   : 1;
            uint32_t u4InitialLatency       : 4;
            uint32_t u3Reserved             : 4;
            uint32_t u3DriveStrength        : 3;
            uint32_t u1DeepPowerDownEnable  : 1;
            uint32_t : 16;
        };
    } CONFIG0;

    union
    {
        uint32_t u32REG;
        struct
        {
            uint32_t u2DistributedRefreshInerval     : 2;
            uint32_t u3PartialArrayRefresh           : 3;
            uint32_t u1HybirdSleep                   : 1;
            uint32_t u1MasterClockType               : 1;
            uint32_t u8Reserved                      : 8;
            uint32_t : 16;
        };
    } CONFIG1;

} HRAM_REG_T;

//------------------------------------------------------------------------------
#define BUFF_SIZE                   0x200

//------------------------------------------------------------------------------
void HyperRAM_Erase(SPIM_T *spim, uint32_t u32StartAddr, uint32_t u32EraseSize);
void HyperRAM_Init(SPIM_T *spim);
void HyperRAM_PinConfig(SPIM_T *spim);

#ifdef __cplusplus
}
#endif

#endif  /* __HYPER_RAM_CODE_H__ */

/*** (C) COPYRIGHT 2019 Nuvoton Technology Corp. ***/
