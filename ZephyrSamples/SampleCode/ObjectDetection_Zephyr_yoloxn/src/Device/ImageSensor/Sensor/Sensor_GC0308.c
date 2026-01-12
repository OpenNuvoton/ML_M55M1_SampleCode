/**************************************************************************//**
 * @file     Sensor_HM1055.c
 * @version  V1.00
 * @brief    HM1055 sensor driver
 *
 * @copyright (C) 2022 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include <stdio.h>
#include "NuMicro.h"
#include "../Sensor.h"
#include "SWI2C.h"

#if __has_include("board_config.h")
    #include "board_config.h"
#endif

#if defined(__CMOS_GC0308_SENSOR__)

int32_t InitGC0308_VGA_YUV422(uint32_t u32Param);

S_SENSOR_INFO g_sSensorGC0308_VGA_YUV422 =
{
    .m_strName        = "GC0308",
    .m_u32Polarity    = (CCAP_PAR_VSP_LOW | CCAP_PAR_HSP_LOW | CCAP_PAR_PCLKP_HIGH),
    .m_u32InputFormat = (CCAP_PAR_INFMT_YUV422 | CCAP_PAR_INDATORD_YUYV),
    .m_u16Width       = 640,
    .m_u16Height      = 480,
    .pfnInitSensor    = InitGC0308_VGA_YUV422
};

struct NT_RegValue
{
    uint8_t    u16RegAddr;         /* Sensor Register Address */
    uint8_t     u8Value;            /* Sensor Register Data */
};

static struct NT_RegValue s_sGC0308_VGA_YUV422[] =
{
    // Initail Sequence Write In.

    /*init registers code.*/
    {0xfe , 0x80},  	

    {0xfe , 0x00},        // set page0


    {0xd2 , 0x10},  // close AEC
    {0x22 , 0x55},  // close AWB

    {0x5a , 0x56},
    {0x5b , 0x40},
    {0x5c , 0x4a},			

    {0x22 , 0x57}, // Open AWB

    {0x01 , 0xfa},
    {0x02 , 0x70},
    {0x0f , 0x01},

    {0x03 , 0x01},
    {0x04 , 0x2c},

    {0xe2 , 0x00},	//anti-flicker step [11:8]
    {0xe3 , 0x64},  //anti-flicker step [7:0]

    {0xe4 , 0x02},  //exp level 1  16.67fps
    {0xe5 , 0x58},
    {0xe6 , 0x03},  //exp level 2  12.5fps
    {0xe7 , 0x20},
    {0xe8 , 0x04},  //exp level 3  8.33fps
    {0xe9 , 0xb0},
    {0xea , 0x09},  //exp level 4  4.00fps
    {0xeb , 0xc4},

    {0x05 , 0x00},
    {0x06 , 0x00},
    {0x07 , 0x00},
    {0x08 , 0x00},
    {0x09 , 0x01},
    {0x0a , 0xe8},
    {0x0b , 0x02},
    {0x0c , 0x88},
    {0x0d , 0x02},
    {0x0e , 0x02},
    {0x10 , 0x22},
    {0x11 , 0xfd},
    {0x12 , 0x2a},
    {0x13 , 0x00},

    {0x14 , 0x10},// change direction  10:normal , 11:H SWITCH,12: V SWITCH, 13:H&V SWITCH   JAMES ADDED

    {0x15 , 0x0a},
    {0x16 , 0x05},
    {0x17 , 0x01},
    {0x18 , 0x44},
    {0x19 , 0x44},
    {0x1a , 0x1e},
    {0x1b , 0x00},
    {0x1c , 0xc1},
    {0x1d , 0x08},
    {0x1e , 0x60},
    {0x1f  , 0x16}, //pad drv ,00 03 13 1f 3f james remarked


    {0x20 , 0xff},
    {0x21 , 0xf8},
    {0x22 , 0x57},
    {0x24 , 0xa2},
    {0x25 , 0x0f},
         
    //output sync_mode       
    {0x26 , 0x03},//vsync  maybe need changed, value is 0x02
    {0x2f , 0x01},
    {0x30 , 0xf7},
    {0x31 , 0x50},
    {0x32 , 0x00},
    {0x39 , 0x04},
    {0x3a , 0x18},
    {0x3b , 0x20},
    {0x3c , 0x00},
    {0x3d , 0x00},
    {0x3e , 0x00},
    {0x3f , 0x00},
    {0x50 , 0x10},
    {0x53 , 0x82},
    {0x54 , 0x80},
    {0x55 , 0x80},
    {0x56 , 0x82},
    {0x8b , 0x40},
    {0x8c , 0x40},
    {0x8d , 0x40},
    {0x8e , 0x2e},
    {0x8f , 0x2e},
    {0x90 , 0x2e},
    {0x91 , 0x3c},
    {0x92 , 0x50},
    {0x5d , 0x12},
    {0x5e , 0x1a},
    {0x5f , 0x24},
    {0x60 , 0x07},
    {0x61 , 0x15},
    {0x62 , 0x08},
    {0x64 , 0x03},
    {0x66 , 0xe8},
    {0x67 , 0x86},
    {0x68 , 0xa2},
    {0x69 , 0x18},
    {0x6a , 0x0f},
    {0x6b , 0x00},
    {0x6c , 0x5f},
    {0x6d , 0x8f},
    {0x6e , 0x55},
    {0x6f , 0x38},
    {0x70 , 0x15},
    {0x71 , 0x33},
    {0x72 , 0xdc},
    {0x73 , 0x80},
    {0x74 , 0x02},
    {0x75 , 0x3f},
    {0x76 , 0x02},
    {0x77 , 0x36},
    {0x78 , 0x88},
    {0x79 , 0x81},
    {0x7a , 0x81},
    {0x7b , 0x22},
    {0x7c , 0xff},
    {0x93 , 0x48},
    {0x94 , 0x00},
    {0x95 , 0x05},
    {0x96 , 0xe8},
    {0x97 , 0x40},
    {0x98 , 0xf0},
    {0xb1 , 0x38},
    {0xb2 , 0x38},
    {0xbd , 0x38},
    {0xbe , 0x36},
    {0xd0 , 0xc9},
    {0xd1 , 0x10},
    //{0xd2 , 0x90},
    {0xd3 , 0x80},
    {0xd5 , 0xf2},
    {0xd6 , 0x16},
    {0xdb , 0x92},
    {0xdc , 0xa5},
    {0xdf , 0x23},
    {0xd9 , 0x00},
    {0xda , 0x00},
    {0xe0 , 0x09},
    {0xec , 0x20},
    {0xed , 0x04},
    {0xee , 0xa0},
    {0xef , 0x40},
    {0x80 , 0x03},
    {0x80 , 0x03},
    {0x9F , 0x10},
    {0xA0 , 0x20},
    {0xA1 , 0x38},
    {0xA2 , 0x4E},
    {0xA3 , 0x63},
    {0xA4 , 0x76},
    {0xA5 , 0x87},
    {0xA6 , 0xA2},
    {0xA7 , 0xB8},
    {0xA8 , 0xCA},
    {0xA9 , 0xD8},
    {0xAA , 0xE3},
    {0xAB , 0xEB},
    {0xAC , 0xF0},
    {0xAD , 0xF8},
    {0xAE , 0xFD},
    {0xAF , 0xFF},
    {0xc0 , 0x00},
    {0xc1 , 0x10},
    {0xc2 , 0x1C},
    {0xc3 , 0x30},
    {0xc4 , 0x43},
    {0xc5 , 0x54},
    {0xc6 , 0x65},
    {0xc7 , 0x75},
    {0xc8 , 0x93},
    {0xc9 , 0xB0},
    {0xca , 0xCB},
    {0xcb , 0xE6},
    {0xcc , 0xFF},
    {0xf0 , 0x02},
    {0xf1 , 0x01},
    {0xf2 , 0x01},
    {0xf3 , 0x30},
    {0xf9 , 0x9f},
    {0xfa , 0x78},

    //---------------------------------------------------------------
    {0xfe , 0x01},  //set page 1

    {0x00 , 0xf5},
    {0x02 , 0x1a},
    {0x0a , 0xa0},
    {0x0b , 0x60},
    {0x0c , 0x08},
    {0x0e , 0x4c},
    {0x0f , 0x39},
    {0x11 , 0x3f},
    {0x12 , 0x72},
    {0x13 , 0x13},
    {0x14 , 0x42},
    {0x15 , 0x43},
    {0x16 , 0xc2},
    {0x17 , 0xa8},
    {0x18 , 0x18},
    {0x19 , 0x40},
    {0x1a , 0xd0},
    {0x1b , 0xf5},
    {0x70 , 0x40},
    {0x71 , 0x58},
    {0x72 , 0x30},
    {0x73 , 0x48},
    {0x74 , 0x20},
    {0x75 , 0x60},
    {0x77 , 0x20},
    {0x78 , 0x32},
    {0x30 , 0x03},
    {0x31 , 0x40},
    {0x32 , 0xe0},
    {0x33 , 0xe0},
    {0x34 , 0xe0},
    {0x35 , 0xb0},
    {0x36 , 0xc0},
    {0x37 , 0xc0},
    {0x38 , 0x04},
    {0x39 , 0x09},
    {0x3a , 0x12},
    {0x3b , 0x1C},
    {0x3c , 0x28},
    {0x3d , 0x31},
    {0x3e , 0x44},
    {0x3f , 0x57},
    {0x40 , 0x6C},
    {0x41 , 0x81},
    {0x42 , 0x94},
    {0x43 , 0xA7},
    {0x44 , 0xB8},
    {0x45 , 0xD6},
    {0x46 , 0xEE},
    {0x47 , 0x0d},

    {0xfe , 0x00},  

    {0xd2 , 0x90}, // Open AEC at last.  


    /////////////////////////////////////////////////////////
    //-----------Update the registers 2010/07/07-------------//
    ///////////////////////////////////////////////////////////

    {0xfe , 0x00},//set Page0

    {0x10 , 0x26},                               
    {0x11 , 0x0d},// fd,modified by mormo 2010/07/06                               
    {0x1a , 0x2a},// 1e,modified by mormo 2010/07/06                                  

    {0x1c , 0x49}, // c1,modified by mormo 2010/07/06                                 
    {0x1d , 0x9a}, // 08,modified by mormo 2010/07/06                                 
    {0x1e , 0x61}, // 60,modified by mormo 2010/07/06                                 

    {0x3a , 0x20},

    {0x50 , 0x14},// 10,modified by mormo 2010/07/06                               
    {0x53 , 0x80},                                
    {0x56 , 0x80},

    {0x8b , 0x20}, //LSC                                 
    {0x8c , 0x20},                                
    {0x8d , 0x20},                                
    {0x8e , 0x14},                                
    {0x8f , 0x10},                                
    {0x90 , 0x14},                                

    {0x94 , 0x02},                                
    {0x95 , 0x07},                                
    {0x96 , 0xe0},                                

    {0xb1 , 0x40}, // YCPT                                 
    {0xb2 , 0x40},                                
    {0xb3 , 0x40},
    {0xb6 , 0xe0},

    {0xd0 , 0xcb}, // AECT  c9,modifed by mormo 2010/07/06                                
    {0xd3 , 0x48}, // 80,modified by mormor 2010/07/06                           

    {0xf2 , 0x02},                                
    {0xf7 , 0x12},
    {0xf8 , 0x0a},



    {0xfe , 0x01},//set  Page1

    {0x02 , 0x20},
    {0x04 , 0x10},
    {0x05 , 0x08},
    {0x06 , 0x20},
    {0x08 , 0x0a},

    {0x0e , 0x44},                                
    {0x0f , 0x32},
    {0x10 , 0x41},                                
    {0x11 , 0x37},                                
    {0x12 , 0x22},                                
    {0x13 , 0x19},                                
    {0x14 , 0x44},                                
    {0x15 , 0x44},

    {0x19 , 0x50},                                
    {0x1a , 0xd8}, 

    {0x32 , 0x10}, 

    {0x35 , 0x00},                                
    {0x36 , 0x80},                                
    {0x37 , 0x00}, 
    //-----------Update the registers end---------//

    {0xfe , 0x00},// set back for page1
    /*Customer can adjust GAMMA, MIRROR & UPSIDEDOWN here!*/
    //GC0308_GAMMA_Select(3}
    //GC0308_H_V_Switch(1}

};

static void Delay(uint32_t nCount)
{
    volatile uint32_t i;

    for (; nCount != 0; nCount--)
        for (i = 0; i < 100; i++);
}

int32_t InitGC0308_VGA_YUV422(uint32_t u32Param)
{
    uint32_t i;
    const uint8_t u8DeviceID = 0x42;

    /* ----------------------------------------------------------------------
     * 1) 設定需要用到的腳位為 GPIO
     *    - PC1: SW I2C SCL
     *    - PC0: SW I2C SDA
     *    - PF6: CMOS_PWDN
     *    - PB11: Sensor nRESET 
     * ---------------------------------------------------------------------- */
    CONFIG_CMOS_SW_I2C_SCL_GPIO();        /* PH2 for GPIO to act as SCL */
    CONFIG_CMOS_SW_I2C_SDA_GPIO();        /* PH3 for GPIO to act as SDA */

    CONFIG_CMOS_RESET_GPIO();
    CONFIG_CMOS_PWDN_GPIO();

    /* ----------------------------------------------------------------------
     * 2) PWDN / RESET 時序
     *    常見：PWDN=0 解除省電；nRESET 低脈衝後拉高
     * ---------------------------------------------------------------------- */
    GPIO_SetMode(CONFIG_CMOS_PWDN_PORT, CONFIG_CMOS_PWDN_PIN, GPIO_MODE_OUTPUT);
    CONFIG_CMOS_PWDN_PORT_PIN = 0;
    Delay(2000);

    GPIO_SetMode(CONFIG_CMOS_RESET_PORT, CONFIG_CMOS_RESET_PIN, GPIO_MODE_OUTPUT);
    CONFIG_CMOS_RESET_PORT_PIN = 0;
    Delay(1000);
    CONFIG_CMOS_RESET_PORT_PIN = 1;
    Delay(2000);

    /* ----------------------------------------------------------------------
     * 3) 開 SW I2C
     * ---------------------------------------------------------------------- */
     SWI2C_Open(CONFIG_CMOS_I2C_SCL_PORT, CONFIG_CMOS_I2C_SCL_PIN,
               CONFIG_CMOS_I2C_SDA_PORT, CONFIG_CMOS_I2C_SDA_PIN, Delay);
    /* ----------------------------------------------------------------------
     * 4) Probe：先確認 sensor 會 ACK（避免後面默默全失敗）
     * ---------------------------------------------------------------------- */
    if (SWI2C_Write_8bitSlaveAddr_8bitReg_8bitData(u8DeviceID, 0xFE, 0x00) == FALSE)
    {
        printf("GC0308 no ACK at 0x%02X\n", u8DeviceID);
        return 0;
    }

    /* ----------------------------------------------------------------------
     * 5) 寫入初始化表（寫失敗就直接回傳 0）
     * ---------------------------------------------------------------------- */
    for (i = 0; i < (sizeof(s_sGC0308_VGA_YUV422) / sizeof(s_sGC0308_VGA_YUV422[0])); i++)
    {
        if (SWI2C_Write_8bitSlaveAddr_8bitReg_8bitData(
                u8DeviceID,
                s_sGC0308_VGA_YUV422[i].u16RegAddr,
                s_sGC0308_VGA_YUV422[i].u8Value) == FALSE)
        {
            printf("GC0308 I2C write fail reg=0x%02X val=0x%02X\n",
                   s_sGC0308_VGA_YUV422[i].u16RegAddr,
                   s_sGC0308_VGA_YUV422[i].u8Value);
            return 0;
        }
    }

    return 1;
}

#endif // __CMOS_GC0308_SENSOR__
