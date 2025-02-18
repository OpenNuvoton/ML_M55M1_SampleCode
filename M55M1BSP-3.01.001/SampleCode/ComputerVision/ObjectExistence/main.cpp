/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    ObjectExistence sample. Demonstrate detecting objects existence or not.
 *           Use imageprocessing methods, including absDifference, binary, dilate &
 *           erode and finding blobs.
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2023 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "BoardInit.hpp"      /* Board initialisation */
#include "log_macros.h"      /* Logging macros (optional) */

#include "BufAttributes.hpp" /* Buffer attributes to be applied */
#include "InputFiles.hpp"             /* Baked-in input (not needed for live data) */
#include "Labels.hpp"


#include "imlib.h"          /* Image processing */
#include "framebuffer.h"

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

//#define __PROFILE__
#define __USE_CCAP__
#define __USE_DISPLAY__
//#define __USE_UVC__

#include "Profiler.hpp"

#if defined (__USE_CCAP__)
    #include "ImageSensor.h"
#endif
#if defined (__USE_DISPLAY__)
    #include "Display.h"
#endif

#if defined (__USE_UVC__)
    #include "UVC.h"
#endif

#define IMAGE_DISP_UPSCALE_FACTOR 1
#if defined(LT7381_LCD_PANEL)
#define FONT_DISP_UPSCALE_FACTOR 2
#else
#define FONT_DISP_UPSCALE_FACTOR 1
#endif

/* openMV orb.c */
#define KDESC_SIZE  (32) // 32 bytes
#define MAX_KP_DIST (KDESC_SIZE*8)
#define __BGDI_EDGE__

namespace arm
{
namespace app
{
/* Tensor arena buffer */
//static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

/* Optional getter function for the model pointer and its size. */
namespace mobilenet
{
extern uint8_t *GetModelPointer();
extern size_t GetModelLen();
} /* namespace mobilenet */
} /* namespace app */
} /* namespace arm */

/* Image processing initiate function */
//Used by omv library
#if defined(__USE_UVC__)
    //UVC only support QVGA, QQVGA
    #define GLCD_WIDTH  320
    #define GLCD_HEIGHT 240
#else
    #define GLCD_WIDTH 320
    #define GLCD_HEIGHT 240
#endif

//RGB565
#define IMAGE_FB_SIZE   (GLCD_WIDTH * GLCD_HEIGHT * 2)

#undef OMV_FB_SIZE
#define OMV_FB_SIZE (IMAGE_FB_SIZE + 1024)

#undef OMV_FB_CALC_SIZE
#define OMV_FB_CALC_SIZE (IMAGE_FB_SIZE + 1024)

__attribute__((section(".bss.vram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.vram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

char *_fb_base = NULL;
char *_fb_end = NULL;
char *_jpeg_buf = NULL;
char *_fballoc = NULL;

__attribute__((section(".bss.sram.data"), aligned(32))) static char pu8ImgFB_bg_rgb[OMV_FB_CALC_SIZE];
//__attribute__((section(".bss.sram.data"), aligned(32))) static char pu8ImgFB_rgb[OMV_FB_CALC_SIZE];

static void omv_init()
{
    image_t frameBuffer;

    frameBuffer.w = GLCD_WIDTH;
    frameBuffer.h = GLCD_HEIGHT;
    frameBuffer.size = GLCD_WIDTH * GLCD_HEIGHT * 2;
    frameBuffer.pixfmt = PIXFORMAT_RGB565;

    _fb_base = fb_array;
    _fb_end =  fb_array + OMV_FB_SIZE - 1;
    _fballoc = _fb_base + OMV_FB_SIZE + OMV_FB_ALLOC_SIZE;
    _jpeg_buf = jpeg_array;

    fb_alloc_init0();

    framebuffer_init0();
    framebuffer_init_from_image(&frameBuffer);
}

/* openMV orb.c */
static inline uint32_t popcount(uint32_t i)
{
    i = i - ((i >> 1) & 0x55555555);
    i = ((i & 0xAAAAAAAA) >> 1) | (i & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

/* openMV orb.c */
static kp_t *find_best_match(kp_t *kp1, array_t *kpts, int *dist_out1, int *dist_out2, int *index)
{
    kp_t *min_kp = NULL;
    int min_dist1 = MAX_KP_DIST;
    int min_dist2 = MAX_KP_DIST;
    int kpts_size = array_length(kpts);

    for (int i = 0; i < kpts_size; i++)
    {
        int dist = 0;
        kp_t *kp2 = (kp_t *)array_at(kpts, i);

        if (kp2->matched == 0)
        {
            for (int m = 0; m < (KDESC_SIZE / 4); m++)
            {
                dist += popcount(((uint32_t *)(kp1->desc))[m] ^ ((uint32_t *)(kp2->desc))[m]);
            }

            if (dist < min_dist1)
            {
                *index = i;
                min_kp = kp2;
                min_dist2 = min_dist1;
                min_dist1 = dist;
            }
        }
    }

    *dist_out1 = min_dist1;
    *dist_out2 = min_dist2;
    return min_kp;
}

int main()
{

    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();


    /* Setup cache poicy of tensor arean buffer */
    info("Set tesnor arena cache policy to WTRA \n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)fb_array),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)fb_array) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
        {
            // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
            ARM_MPU_RBAR(((unsigned int)pu8ImgFB_bg_rgb),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)pu8ImgFB_bg_rgb) + OMV_FB_SIZE - 1),        // Limit
                         eMPU_ATTR_NON_CACHEABLE) // NonCache
        },
        //{
        //    // Image data from CCAP DMA, so must set frame buffer to Non-cache attribute
        //    ARM_MPU_RBAR(((unsigned int)pu8ImgFB_rgb),        // Base
        //                 ARM_MPU_SH_NON,    // Non-shareable
        //                 0,                 // Read-only
        //                 1,                 // Non-Privileged
        //                 1),                // eXecute Never enabled
        //    ARM_MPU_RLAR((((unsigned int)pu8ImgFB_rgb) + OMV_FB_CALC_SIZE - 1),        // Limit
        //                 eMPU_ATTR_NON_CACHEABLE) // NonCache
        //}
    };

    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

    uint8_t u8ImgIdx = 1;


    //label information
    std::vector <std::string> labels;
    GetLabelsVector(labels);
    std::string predictLabelInfo;

    //display framebuffer
    image_t frameBuffer;
    rectangle_t roi;

    //omv library init
    omv_init();
    framebuffer_init_image(&frameBuffer);

#if defined(__PROFILE__)
    arm::app::Profiler profiler;
    uint64_t u64StartCycle;
    uint64_t u64EndCycle;
    uint64_t u64CCAPStartCycle;
    uint64_t u64CCAPEndCycle;
#else
    pmu_reset_counters();
#endif

#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = pmu_get_systick_Count();
    u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);

#if defined (__USE_CCAP__)
    //Setup image senosr
    ImageSensor_Init();
    ImageSensor_Config(eIMAGE_FMT_RGB565, frameBuffer.w, frameBuffer.h, true);
#endif

#if defined (__USE_DISPLAY__)
    char szDisplayText[160];

    Display_Init();
    Display_ClearLCD(C_WHITE);
#endif

#if defined (__USE_UVC__)
    UVC_Init();
    HSUSBD_Start();
#endif

    image_t frameBuffer_bg_rgb;
    frameBuffer_bg_rgb.w = IMAGE_WIDTH;
    frameBuffer_bg_rgb.h = IMAGE_HEIGHT;
    frameBuffer_bg_rgb.size = IMAGE_WIDTH * IMAGE_HEIGHT * 2;
    frameBuffer_bg_rgb.pixfmt = PIXFORMAT_RGB565;
    frameBuffer_bg_rgb.data = (uint8_t *)pu8ImgFB_bg_rgb;

    roi.x = 0;
    roi.y = 0;
    roi.w = IMAGE_WIDTH;
    roi.h = IMAGE_HEIGHT;

#if !defined (__USE_CCAP__)
    /* load the first background img */
    uint8_t u8ImgIdx_bg = 0;
    const uint8_t *pu8ImgSrc_bg = get_img_array(u8ImgIdx_bg);

    if (nullptr == pu8ImgSrc_bg)
    {
        printf_err("Failed to get image index %" PRIu32 " (max: %u)\n", u8ImgIdx_bg,
                   NUMBER_OF_FILES - 1);
        return 4;
    }

    //resize source image to framebuffer as RGB565
    image_t srcImg_bg;

    srcImg_bg.w = IMAGE_WIDTH;
    srcImg_bg.h = IMAGE_HEIGHT;
    srcImg_bg.pixfmt = PIXFORMAT_RGB888;
    srcImg_bg.data = (uint8_t *)pu8ImgSrc_bg;

    imlib_nvt_scale(&srcImg_bg, &frameBuffer_bg_rgb, &roi);

    char chStdIn;
    info("Press 'n' to run next image inference \n");
    info("Press 'q' to exit program \n");

    while ((chStdIn = getchar()))
    {
        if (chStdIn == 'q')
            break;
        else if (chStdIn != 'n')
            continue;

#else

    char chStdIn;
    info("Press 'b' to save the image background \n");

    while ((chStdIn = getchar()))
    {
        if (chStdIn == 'b')
        {

            int init_c = 100;

            while (init_c--)
            {
                ImageSensor_TriggerCapture((uint32_t)frameBuffer_bg_rgb.data);
                ImageSensor_WaitCaptureDone();
            }

            break;
        }
        else if (chStdIn != 'b')
            continue;
    }

    while (1)
    {
        //info("Start to test the object exist !\n");
#endif
        //image_t frameBuffer_rgb;
        //
        //frameBuffer_rgb.w = IMAGE_WIDTH;
        //frameBuffer_rgb.h = IMAGE_HEIGHT;
        //frameBuffer_rgb.size = IMAGE_WIDTH * IMAGE_HEIGHT * 2;
        //frameBuffer_rgb.pixfmt = PIXFORMAT_RGB565;
        //frameBuffer_rgb.data = (uint8_t *)pu8ImgFB_rgb;

        roi.x = 0;
        roi.y = 0;
        roi.w = IMAGE_WIDTH;
        roi.h = IMAGE_HEIGHT;

#if !defined (__USE_CCAP__)

        const uint8_t *pu8ImgSrc = get_img_array(u8ImgIdx);

        if (nullptr == pu8ImgSrc)
        {
            printf_err("Failed to get image index %" PRIu32 " (max: %u)\n", u8ImgIdx,
                       NUMBER_OF_FILES - 1);
            return 4;
        }

#if defined(__PROFILE__)
        u64StartCycle = pmu_get_systick_Count();
#endif
        // resize source image to framebuffer as RGB565
        image_t srcImg;

        srcImg.w = IMAGE_WIDTH;
        srcImg.h = IMAGE_HEIGHT;
        srcImg.pixfmt = PIXFORMAT_RGB888;
        srcImg.data = (uint8_t *)pu8ImgSrc;

        imlib_nvt_scale(&srcImg, &frameBuffer, &roi);

#else

        ImageSensor_TriggerCapture((uint32_t)frameBuffer.data);
        ImageSensor_WaitCaptureDone();

#if defined(__PROFILE__)
        u64StartCycle = pmu_get_systick_Count();
#endif
        //imlib_nvt_scale(&frameBuffer, &frameBuffer_rgb, &roi);
        //memcpy(&frameBuffer_rgb, &frameBuffer, sizeof(image_t));

        //Capture new image
#if defined(__PROFILE__)
        u64CCAPStartCycle = pmu_get_systick_Count();
#endif

#endif


#if defined (__BGDI_EDGE__)

        // difference the two imags (BG subtraction)
        imlib_difference(&frameBuffer, NULL, &frameBuffer_bg_rgb, 1, NULL);

        /* create image binary */
        list_t thresholds;
        list_init(&thresholds, sizeof(color_thresholds_list_lnk_data_t));
        color_thresholds_list_lnk_data_t lnk_data;
        lnk_data.LMin = 25;
        lnk_data.LMax = 100;
        lnk_data.AMin = -128;
        lnk_data.AMax = 127;
        lnk_data.BMin = -128;
        lnk_data.BMax = 127;
        list_push_back(&thresholds, &lnk_data);

        imlib_binary(&frameBuffer, &frameBuffer, &thresholds, false, false, NULL);

        /* dilate & erode*/
        int8_t iter = 1;
        int k_size = 3;
        int erode_threshold = (((k_size * 2) + 1) * ((k_size * 2) + 1)) - 1;

        for (int i = 0; i < iter; i++)
            imlib_dilate(&frameBuffer, k_size, 0, NULL);

        for (int i = 0; i < iter; i++)
            imlib_erode(&frameBuffer, k_size, erode_threshold, NULL);

        for (int i = 0; i < ceil(iter / 2); i++)
            imlib_dilate(&frameBuffer, k_size, 0, NULL);

        /* find blobs */
        list_t output_blobs;
        unsigned int area_threshold_val = 50;
        unsigned int pixels_threshold_val = 50;

        imlib_find_blobs(&output_blobs, &frameBuffer, &roi, 4, 4,
                         &thresholds, false, area_threshold_val, pixels_threshold_val,
                         true, 10,
                         NULL, NULL,
                         NULL, NULL,
                         0, 0);
        uint16_t output_blobs_size = list_size(&output_blobs);
        info("Find output blobs size = %u\n", output_blobs_size);

        uint8_t blobs_c = 1;

        while (list_size(&output_blobs))
        {
            find_blobs_list_lnk_data_t lnk_blob;
            list_pop_front(&output_blobs, &lnk_blob);

            info("%d blobs size = %d\n", blobs_c++, lnk_blob.pixels);
        }

        // free the malloc
        list_free(&output_blobs);

#if defined(__PROFILE__)
        u64EndCycle = pmu_get_systick_Count();
        info("Find difference's cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif


#endif

#if defined (__USE_DISPLAY__)
        //Display image on LCD
        S_DISP_RECT sDispRect;

        sDispRect.u32TopLeftX = 0;
        sDispRect.u32TopLeftY = 0;
        sDispRect.u32BottonRightX = ((frameBuffer.w * IMAGE_DISP_UPSCALE_FACTOR) - 1);
        sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) - 1);

#if defined(__PROFILE__)
        u64StartCycle = pmu_get_systick_Count();
#endif

        Display_FillRect((uint16_t *)frameBuffer.data, &sDispRect, IMAGE_DISP_UPSCALE_FACTOR);

#if defined(__PROFILE__)
        u64EndCycle = pmu_get_systick_Count();
        info("display image cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif

#endif

#if defined (__USE_UVC__)

        if (UVC_IsConnect())
        {
#if (UVC_Color_Format == UVC_Format_YUY2)
            image_t RGB565Img;
            image_t YUV422Img;

            RGB565Img.w = frameBuffer.w;
            RGB565Img.h = frameBuffer.h;
            RGB565Img.data = (uint8_t *)frameBuffer.data;
            RGB565Img.pixfmt = PIXFORMAT_RGB565;

            YUV422Img.w = RGB565Img.w;
            YUV422Img.h = RGB565Img.h;
            YUV422Img.data = (uint8_t *)frameBuffer.data;
            YUV422Img.pixfmt = PIXFORMAT_YUV422;

            roi.x = 0;
            roi.y = 0;
            roi.w = RGB565Img.w;
            roi.h = RGB565Img.h;
            imlib_nvt_scale(&RGB565Img, &YUV422Img, &roi);

#else
            image_t origImg;
            image_t vflipImg;

            origImg.w = frameBuffer.w;
            origImg.h = frameBuffer.h;
            origImg.data = (uint8_t *)frameBuffer.data;
            origImg.pixfmt = PIXFORMAT_RGB565;

            vflipImg.w = origImg.w;
            vflipImg.h = origImg.h;
            vflipImg.data = (uint8_t *)frameBuffer.data;
            vflipImg.pixfmt = PIXFORMAT_RGB565;

            imlib_nvt_vflip(&origImg, &vflipImg);
#endif
            UVC_SendImage((uint32_t)frameBuffer.data, IMAGE_FB_SIZE, uvcStatus.StillImage);
        }

#endif




#if !defined (__USE_CCAP__)
        /* Run inference over this image. */
        info("Running on image %" PRIu32 " => %s\n", u8ImgIdx, get_filename(u8ImgIdx));
#endif

#if defined (__USE_CCAP__)
#if defined(__PROFILE__)
        u64CCAPEndCycle = pmu_get_systick_Count();
        info("ccap capture cycles %llu \n", (u64CCAPEndCycle - u64CCAPStartCycle));
#endif
#endif

        //results.clear();
        predictLabelInfo.clear();

        if (output_blobs_size)
            predictLabelInfo = std::string("Object") + std::string(" Detected");
        else
            predictLabelInfo = std::string(" Nothing") + std::string(" Detected");

        //show result
        info("Final results:\n");
        info("%s\n", predictLabelInfo.c_str());

#if defined (__USE_DISPLAY__)
        sprintf(szDisplayText, "%s", predictLabelInfo.c_str());

        sDispRect.u32TopLeftX = 0;
        sDispRect.u32TopLeftY = frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR;
        sDispRect.u32BottonRightX = (Disaplay_GetLCDWidth() - 1);
        sDispRect.u32BottonRightY = ((frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR) + (FONT_DISP_UPSCALE_FACTOR * FONT_HTIGHT) - 1);

        Display_ClearRect(C_WHITE, &sDispRect);
        Display_PutText(
            szDisplayText,
            strlen(szDisplayText),
            0,
            frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR,
            C_BLUE,
            C_WHITE,
            false,
            FONT_DISP_UPSCALE_FACTOR
        );
#endif

#if defined(__PROFILE__)
        profiler.PrintProfilingResult();
#endif

        u64PerfFrames ++;

        if (pmu_get_systick_Count() > u64PerfCycle)
        {
            info("Total inference rate: %llu\n", u64PerfFrames / EACH_PERF_SEC);

#if defined (__USE_DISPLAY__)
            sprintf(szDisplayText, "Frame Rate %llu", u64PerfFrames / EACH_PERF_SEC);

            sDispRect.u32TopLeftX = 0;
            sDispRect.u32TopLeftY = frameBuffer.h + (FONT_HTIGHT * FONT_DISP_UPSCALE_FACTOR);
            sDispRect.u32BottonRightX = (frameBuffer.w);
            sDispRect.u32BottonRightY = (frameBuffer.h + (2 * FONT_HTIGHT * FONT_DISP_UPSCALE_FACTOR) - 1);

            Display_ClearRect(C_WHITE, &sDispRect);
            Display_PutText(
                szDisplayText,
                strlen(szDisplayText),
                0,
                (frameBuffer.h * IMAGE_DISP_UPSCALE_FACTOR)+ FONT_HTIGHT * FONT_DISP_UPSCALE_FACTOR,
                C_BLUE,
                C_WHITE,
                false,
                FONT_DISP_UPSCALE_FACTOR
            );
#endif

            u64PerfCycle = pmu_get_systick_Count();
            u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);
            u64PerfFrames = 0;
        }

round_done:
        u8ImgIdx ++;

        if (u8ImgIdx >= (NUMBER_OF_FILES))
            u8ImgIdx = 0;

    }
}