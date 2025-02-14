/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    A demonstration of agast Feature-Detection-Keypoints.
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

#define __PROFILE__
//#define __USE_CCAP__
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

/* openMV orb.c */
#define KDESC_SIZE  (32) // 32 bytes
#define MAX_KP_DIST (KDESC_SIZE*8)
#define __AGAST_KPS__

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
#define OMV_FB_CALC_SIZE (100352 + 1024)

__attribute__((section(".bss.sram.data"), aligned(32))) static char fb_array[OMV_FB_SIZE + OMV_FB_ALLOC_SIZE];
__attribute__((section(".bss.sram.data"), aligned(32))) static char jpeg_array[OMV_JPEG_BUF_SIZE];

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
            ARM_MPU_RLAR((((unsigned int)pu8ImgFB_bg_rgb) + OMV_FB_CALC_SIZE - 1),        // Limit
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


        frameBuffer.w = IMAGE_WIDTH;
        frameBuffer.h = IMAGE_HEIGHT;
        imlib_nvt_scale(&srcImg, &frameBuffer, &roi);

#else

        ImageSensor_TriggerCapture((uint32_t)frameBuffer.data);
        ImageSensor_WaitCaptureDone();

        //Capture new image
#if defined(__PROFILE__)
        u64CCAPStartCycle = pmu_get_systick_Count();
#endif

#endif

#if defined (__AGAST_KPS__)
        // find key points
        array_t *kpts_gd;
        int16_t kpts_gd_x_list[100] = {-1};
        int16_t kpts_gd_y_list[100] = {-1};
        //array_alloc(&kpts_gd, xfree);

        kpts_gd = orb_find_keypoints(&frameBuffer_bg_rgb, false, 10,
                                     1.2, 100, CORNER_AGAST, &roi);

        for (int k = 0; k < kpts_gd->index; k++)
        {
            // Set keypoint octave/scale
            kp_t *kpt = (kp_t *)kpts_gd->data[k];
            kpts_gd_x_list[k] = kpt->x;
            kpts_gd_y_list[k] = kpt->y;
            info("kp %d: x=%d, y=%d\n", k, kpt->x, kpt->y);
        }

#if defined(__PROFILE__)
        u64StartCycle = pmu_get_systick_Count();
#endif

        // find another pic key points
        array_t *kpts_pd;
        //array_alloc(&kpts_pd, xfree);

        kpts_pd = orb_find_keypoints(&frameBuffer, true, 10,
                                     1.2, 100, CORNER_AGAST, &roi);

        // match the 2 set of key points
        int match_p;
        rectangle_t rectangle_p;
        point_t c;
        int angle_p;

        int match_num = orb_match_keypoints(kpts_gd, kpts_pd, &match_p, 45, &rectangle_p, &c, &angle_p);
#if defined(__PROFILE__)
        u64EndCycle = pmu_get_systick_Count();
        info("AGAST keypoints's cycles %llu \n", (u64EndCycle - u64StartCycle));
#endif
        info("match_num: %d, theta: %d\n", match_num, angle_p);

        if (match_num > 0)
        {
            imlib_draw_rectangle(&frameBuffer, rectangle_p.x, rectangle_p.y, rectangle_p.w, rectangle_p.h, COLOR_B5_MAX, 1, false);

            for (int k = 0; k < kpts_gd->index; k++)
            {
                //kp_t *kpt = (kp_t *)kpts_gd->data[k];
                //info("kp %d: x=%d, y=%d\n", k, kpt->x, kpt->y);
                if (kpts_gd_x_list[k] > 0 && kpts_gd_y_list[k] > 0)
                    imlib_draw_circle(&frameBuffer, kpts_gd_x_list[k], kpts_gd_y_list[k], 4, COLOR_B5_MAX, 2, false);
            }
        }

#endif



#if defined (__USE_DISPLAY__)
        //Display image on LCD
        S_DISP_RECT sDispRect;

        sDispRect.u32TopLeftX = 0;
        sDispRect.u32TopLeftY = 0;
        sDispRect.u32BottonRightX = (frameBuffer.w - 1);
        sDispRect.u32BottonRightY = (frameBuffer.h - 1);

#if defined(__PROFILE__)
        u64StartCycle = pmu_get_systick_Count();
#endif

        Display_FillRect((uint16_t *)frameBuffer.data, &sDispRect);

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

        if (match_num)
            predictLabelInfo = std::string("Object") + std::string(" Same");
        else
            predictLabelInfo = std::string(" Object") + std::string(" Not Same");

        //show result
        info("Final results:\n");
        info("%s\n", predictLabelInfo.c_str());

#if defined (__USE_DISPLAY__)
        sprintf(szDisplayText, "%s", predictLabelInfo.c_str());

        sDispRect.u32TopLeftX = 0;
        sDispRect.u32TopLeftY = frameBuffer.h;
        sDispRect.u32BottonRightX = (Disaplay_GetLCDWidth() - 1);
        sDispRect.u32BottonRightY = (frameBuffer.h + FONT_HTIGHT - 1);

        Display_ClearRect(C_WHITE, &sDispRect);
        Display_PutText(
            szDisplayText,
            strlen(szDisplayText),
            0,
            frameBuffer.h,
            C_BLUE,
            C_WHITE,
            false
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
            sDispRect.u32TopLeftY = frameBuffer.h + FONT_HTIGHT;
            sDispRect.u32BottonRightX = (frameBuffer.w);
            sDispRect.u32BottonRightY = (frameBuffer.h + (2 * FONT_HTIGHT) - 1);

            Display_ClearRect(C_WHITE, &sDispRect);
            Display_PutText(
                szDisplayText,
                strlen(szDisplayText),
                0,
                frameBuffer.h + FONT_HTIGHT,
                C_BLUE,
                C_WHITE,
                false
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