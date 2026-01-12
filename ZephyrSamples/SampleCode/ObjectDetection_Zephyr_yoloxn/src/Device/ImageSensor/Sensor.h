/**************************************************************************//**
 * @file     sensor.h
 * @version  V1.00
 * @brief    Sensor driver
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 *****************************************************************************/
#ifndef __SENSOR_H__
#define __SENSOR_H__

#include "NuMicro.h"

//TODO: define by project's kconfig
#if defined(__NUMAKER_M55M1__)

#define CONFIG_CMOS_SW_I2C_SCL_GPIO    SET_GPIO_PH2
#define CONFIG_CMOS_SW_I2C_SDA_GPIO    SET_GPIO_PH3

#define CONFIG_CMOS_RESET_GPIO         SET_GPIO_PG11
#define CONFIG_CMOS_RESET_PORT_PIN     PG11
#define CONFIG_CMOS_RESET_PORT         PG
#define CONFIG_CMOS_RESET_PIN          BIT11

#define CONFIG_CMOS_PWDN_GPIO          SET_GPIO_PD12
#define CONFIG_CMOS_PWDN_PORT_PIN      PD12
#define CONFIG_CMOS_PWDN_PORT          PD
#define CONFIG_CMOS_PWDN_PIN           BIT12

#define CONFIG_CMOS_I2C_SCL_PORT        eDRVGPIO_GPIOH
#define CONFIG_CMOS_I2C_SCL_PIN         eDRVGPIO_PIN2
#define CONFIG_CMOS_I2C_SDA_PORT        eDRVGPIO_GPIOH
#define CONFIG_CMOS_I2C_SDA_PIN         eDRVGPIO_PIN3

#endif

#if defined(__NUGESTUREAI_M55M1__)

#define CONFIG_CMOS_SW_I2C_SCL_GPIO    SET_GPIO_PC1
#define CONFIG_CMOS_SW_I2C_SDA_GPIO    SET_GPIO_PC0

#define CONFIG_CMOS_RESET_GPIO         SET_GPIO_PB11
#define CONFIG_CMOS_RESET_PORT_PIN     PB11
#define CONFIG_CMOS_RESET_PORT         PB
#define CONFIG_CMOS_RESET_PIN          BIT11

#define CONFIG_CMOS_PWDN_GPIO          SET_GPIO_PF6
#define CONFIG_CMOS_PWDN_PORT_PIN      PF6
#define CONFIG_CMOS_PWDN_PORT          PF
#define CONFIG_CMOS_PWDN_PIN           BIT6

#define CONFIG_CMOS_I2C_SCL_PORT       eDRVGPIO_GPIOC
#define CONFIG_CMOS_I2C_SCL_PIN        eDRVGPIO_PIN1
#define CONFIG_CMOS_I2C_SDA_PORT       eDRVGPIO_GPIOC
#define CONFIG_CMOS_I2C_SDA_PIN        eDRVGPIO_PIN0

#endif



typedef int32_t (*PFN_INIT_SENSOR_FUNC)(uint32_t u32Param);

typedef struct s_sensor_info
{
    char        m_strName[16];
    uint32_t    m_u32Polarity;
    uint32_t    m_u32InputFormat;
    uint16_t    m_u16Width;
    uint16_t    m_u16Height;
    PFN_INIT_SENSOR_FUNC    pfnInitSensor;
} S_SENSOR_INFO;

extern S_SENSOR_INFO g_sSensorHM1055_VGA_YUV422;
extern S_SENSOR_INFO g_sSensorHM1055_QVGA_YUV422;
extern S_SENSOR_INFO g_sSensorHM1055_320_320_YUV422;
extern S_SENSOR_INFO g_sSensorGC0308_VGA_YUV422;

#endif  // __SENSOR_H__
