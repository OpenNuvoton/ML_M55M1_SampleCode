/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    network inference sample with Edge Impulse SDK. Demonstrate network infereence
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2025 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/
#include "BoardInit.hpp"      /* Board initialisation */

#undef PI /* PI macro conflict with CMSIS/DSP */
#include "NuMicro.h"

#include "ModelFileReader.h"
#include "ff.h"


#define EI_CLASSIFIER_ALLOCATION_STATIC // use static memory 
#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "edge-impulse-sdk/classifier/inferencing_engines/tflite_micro_extern.h" // need extern addr to set MPU

//#define LOG_LEVEL_TRACE       0
//#define LOG_LEVEL_DEBUG       1
//#define LOG_LEVEL_INFO        2
//#define LOG_LEVEL_WARN        3
//#define LOG_LEVEL_ERROR       4

#define LOG_LEVEL             2
#include "log_macros.h"      /* Logging macros (optional) */

//#define __LOAD_MODEL_FROM_SD__

#define MODEL_AT_HYPERRAM_ADDR 0x82400000

#include "Profiler.hpp"

// Callback function declaration
static int get_signal_data(size_t offset, size_t length, float *out_ptr);

// Raw features copied from test sample (Edge Impulse > Model testing)
static float input_buf[] =
{
    -0.1285, 0.0104, 1.0353, -0.1280, 0.0156, 1.0275, -0.1284, 0.0206, 1.0228, -0.1257, 0.0231, 1.0208, -0.1247, 0.0240, 1.0172, -0.1294, 0.0253, 1.0165, -0.1279, 0.0234, 1.0207, -0.1304, 0.0220, 1.0278, -0.1273, 0.0197, 1.0353, -0.1283, 0.0187, 1.0359, -0.1273, 0.0200, 1.0380, -0.1249, 0.0271, 1.0324, -0.1261, 0.0327, 1.0246, -0.1307, 0.0345, 1.0211, -0.1345, 0.0329, 1.0187, -0.1377, 0.0325, 1.0148, -0.1385, 0.0340, 1.0097, -0.1397, 0.0387, 1.0089, -0.1412, 0.0349, 1.0147, -0.1377, 0.0361, 1.0154, -0.1362, 0.0408, 1.0147, -0.1373, 0.0450, 1.0154, -0.1414, 0.0491, 1.0156, -0.1426, 0.0512, 1.0236, -0.1456, 0.0541, 1.0291, -0.1447, 0.0564, 1.0204, -0.1470, 0.0619, 1.0251, -0.1473, 0.0686, 1.0275, -0.1500, 0.0740, 1.0361, -0.1581, 0.0807, 1.0363, -0.1686, 0.0859, 1.0421, -0.1747, 0.0941, 1.0372, -0.1851, 0.1046, 1.0215, -0.1857, 0.1138, 1.0080, -0.1790, 0.1196, 0.9958, -0.1782, 0.1216, 0.9926, -0.1685, 0.1209, 0.9908, -0.1666, 0.1223, 0.9816, -0.1563, 0.1304, 0.9670, -0.1507, 0.1391, 0.9570, -0.1389, 0.1422, 0.9530, -0.1379, 0.1484, 0.9442, -0.1321, 0.1512, 0.9357, -0.1317, 0.1552, 0.9119, -0.1377, 0.1589, 0.8907, -0.1493, 0.1691, 0.8856, -0.1473, 0.1526, 0.8416, -0.1429, 0.1703, 0.8156, -0.1375, 0.1726, 0.7975, -0.1412, 0.1757, 0.7957, -0.1396, 0.1594, 0.7714, -0.1301, 0.1428, 0.7480, -0.1077, 0.1376, 0.7457, -0.0795, 0.1093, 0.7425, -0.0598, 0.0875, 0.7396, -0.0272, 0.0715, 0.7298, -0.0063, 0.0541, 0.7225, -0.0012, 0.0376, 0.7249, 0.0040, 0.0163, 0.7327, 0.0156, 0.0200, 0.7314, 0.0168, -0.0004, 0.7419, 0.0262, -0.0145, 0.7361, 0.0498, -0.0236, 0.7415, 0.0845, -0.0402, 0.7455, 0.1125, -0.0637, 0.7556, 0.1318, -0.0808, 0.7534, 0.1716, -0.0901, 0.7368, 0.1535, -0.1061, 0.7697, 0.1444, -0.1077, 0.8236, 0.1326, -0.1197, 0.8831, 0.1391, -0.1296, 0.9296, 0.1620, -0.1367, 0.9584, 0.1992, -0.1568, 0.9801, 0.2285, -0.1556, 0.9987, 0.2563, -0.1657, 0.9980, 0.2707, -0.1516, 1.0200, 0.2607, -0.1609, 1.0872, 0.2344, -0.1655, 1.1602, 0.2089, -0.1719, 1.2430, 0.2136, -0.1829, 1.3123, 0.2321, -0.1779, 1.3667, 0.2787, -0.1742, 1.4114, 0.3171, -0.1497, 1.4295, 0.3685, -0.1176, 1.4508, 0.3915, -0.1013, 1.4841, 0.3821, -0.1006, 1.5266, 0.3573, -0.1008, 1.5474, 0.3416, -0.0944, 1.5417, 0.3423, -0.0678, 1.5160, 0.3426, -0.0326, 1.4855, 0.3270, 0.0234, 1.4586, 0.2941, 0.0700, 1.4711, 0.2512, 0.1140, 1.4761, 0.2197, 0.1605, 1.4781, 0.2005, 0.1882, 1.4733, 0.2075, 0.2144, 1.4590, 0.2160, 0.2370, 1.4399, 0.2313, 0.2695, 1.4257, 0.2264, 0.3090, 1.4247, 0.2002, 0.3474, 1.4382, 0.1569, 0.3862, 1.4518, 0.1118, 0.4256, 1.4747, 0.0762, 0.4495, 1.4746, 0.0524, 0.4734, 1.4603, 0.0458, 0.4971, 1.4332, 0.0436, 0.5137, 1.4028, 0.0518, 0.5297, 1.3722, 0.0627, 0.5395, 1.3413, 0.0600, 0.5446, 1.3073, 0.0465, 0.5510, 1.2685, 0.0314, 0.5588, 1.2274, 0.0165, 0.5597, 1.1848, 0.0032, 0.5629, 1.1483, 0.0018, 0.5633, 1.1113, 0.0010, 0.5558, 1.0862, 0.0206, 0.5406, 1.0534, 0.0434, 0.5303, 1.0147, 0.0595, 0.5175, 0.9828, 0.0669, 0.5003, 0.9703, 0.0605, 0.4779, 0.9709, 0.0403, 0.4599, 0.9794, 0.0138, 0.4507, 0.9684, -0.0097, 0.4384, 0.9497, -0.0165, 0.4227, 0.9681, 0.0035, 0.4177, 0.9748, 0.0247, 0.4130, 0.9486, 0.0320, 0.4051, 0.9148, 0.0295, 0.3941, 0.8903, 0.0115, 0.3788, 0.8936, -0.0027, 0.3666, 0.8975, -0.0151, 0.3516, 0.9084, -0.0285, 0.3286, 0.9204, -0.0428, 0.3079, 0.9236, -0.0633, 0.2825, 0.9128, -0.0505, 0.2811, 0.8859, -0.0526, 0.2595, 0.8616, -0.0378, 0.2691, 0.8397, -0.0322, 0.2585, 0.8295, -0.0533, 0.2456, 0.8355, -0.0679, 0.2403, 0.8417, -0.0710, 0.2285, 0.8773, -0.1014, 0.2023, 0.8774, -0.0990, 0.1774, 0.8898, -0.0897, 0.1529, 0.8878, -0.0690, 0.1434, 0.8652, -0.0528, 0.1372, 0.8315, -0.0446, 0.1302, 0.8121, -0.0535, 0.1179, 0.7997, -0.0677, 0.1071, 0.8034, -0.0825, 0.1014, 0.7984, -0.1180, 0.0818, 0.8137, -0.1324, 0.0684, 0.8359, -0.1315, 0.0540, 0.8388, -0.1089, 0.0490, 0.8334, -0.0851, 0.0350, 0.8269, -0.0717, 0.0157, 0.8286, -0.0678, -0.0060, 0.8122, -0.0820, -0.0282, 0.8076, -0.0945, -0.0466, 0.8107, -0.1014, -0.0649, 0.8267, -0.1122, -0.0724, 0.8316, -0.1141, -0.0771, 0.8380, -0.1104, -0.0780, 0.8462, -0.1104, -0.0792, 0.8602, -0.1080, -0.0833, 0.8706, -0.1057, -0.0876, 0.8694, -0.1138, -0.1052, 0.8873, -0.1217, -0.1212, 0.9075, -0.1248, -0.1315, 0.9290, -0.1392, -0.1425, 0.9413, -0.1483, -0.1441, 0.9524, -0.1514, -0.1446, 0.9545, -0.1604, -0.1439, 0.9609, -0.1723, -0.1463, 0.9668, -0.1789, -0.1468, 0.9732, -0.1732, -0.1469, 0.9819, -0.1665, -0.1574, 0.9980, -0.1519, -0.1595, 1.0094, -0.1337, -0.1643, 1.0311, -0.1212, -0.1697, 1.0515, -0.1231, -0.1658, 1.0728, -0.1316, -0.1603, 1.0811, -0.1398, -0.1476, 1.0862, -0.1481, -0.1340, 1.0900, -0.1514, -0.1282, 1.1052, -0.1530, -0.1164, 1.1156, -0.1502, -0.1094, 1.1150, -0.1440, -0.1000, 1.1090, -0.1400, -0.0949, 1.1085, -0.1392, -0.0871, 1.1105, -0.1373, -0.0810, 1.1057, -0.1445, -0.0727, 1.0959, -0.1542, -0.0606, 1.0854, -0.1678, -0.0512, 1.0790, -0.1733, -0.0375, 1.0682, -0.1761, -0.0235, 1.0559, -0.1791, -0.0168, 1.0433, -0.1792, -0.0062, 1.0310, -0.1788, -0.0033, 1.0240, -0.1752, 0.0009, 1.0159
    };

static int32_t PrepareModelToHyperRAM(void)
{
#define MODEL_FILE "0:\\nn_model.tflite"
#define EACH_READ_SIZE 512

    TCHAR sd_path[] = { '0', ':', 0 };    /* SD drive started from 0 */
    f_chdrive(sd_path);          /* set default path */

    int32_t i32FileSize;
    int32_t i32FileReadIndex = 0;
    int32_t i32Read;

    if (!ModelFileReader_Initialize(MODEL_FILE))
    {
        printf_err("Unable open model %s\n", MODEL_FILE);
        return -1;
    }

    i32FileSize = ModelFileReader_FileSize();
    info("Model file size %i \n", i32FileSize);

    while (i32FileReadIndex < i32FileSize)
    {
        i32Read = ModelFileReader_ReadData((BYTE *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), EACH_READ_SIZE);

        if (i32Read < 0)
            break;

        i32FileReadIndex += i32Read;
    }

    if (i32FileReadIndex < i32FileSize)
    {
        printf_err("Read Model file size is not enough\n");
        return -2;
    }

#if 0
    /* verify */
    i32FileReadIndex = 0;
    ModelFileReader_Rewind();
    BYTE au8TempBuf[EACH_READ_SIZE];

    while (i32FileReadIndex < i32FileSize)
    {
        i32Read = ModelFileReader_ReadData((BYTE *)au8TempBuf, EACH_READ_SIZE);

        if (i32Read < 0)
            break;

        if (std::memcmp(au8TempBuf, (void *)(MODEL_AT_HYPERRAM_ADDR + i32FileReadIndex), i32Read) != 0)
        {
            printf_err("verify the model file content is incorrect at %i \n", i32FileReadIndex);
            return -3;
        }

        i32FileReadIndex += i32Read;
    }

#endif
    ModelFileReader_Finish();

    return i32FileSize;
}

// Callback: fill a section of the out_ptr buffer when requested
static int get_signal_data(size_t offset, size_t length, float *out_ptr)
{
    for (size_t i = 0; i < length; i++)
    {
        out_ptr[i] = (input_buf + offset)[i];
    }

    return EIDSP_OK;
}

int main()
{

    /* Initialise the UART module to allow printf related functions (if using retarget) */
    BoardInit();

#if defined(__LOAD_MODEL_FROM_SD__)

    /* Copy model file from SD to HyperRAM*/
    int32_t i32ModelSize;

    printf("==================== Load model file from SD card =================================\n");
    printf("Please copy NN_ModelInference/Model/xxx_vela.tflite to SDCard:/nn_model.tflite     \n");
    printf("===================================================================================\n");
    i32ModelSize = PrepareModelToHyperRAM();

    if (i32ModelSize <= 0)
    {
        printf_err("Failed to prepare model\n");
        return 1;
    }

    /* Model object creation and initialisation. */
    arm::app::NNModel model;

    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    (unsigned char *)MODEL_AT_HYPERRAM_ADDR,
                    i32ModelSize))
    {
        printf_err("Failed to initialise model\n");
        return 1;
    }

#else

    /* Model object creation and initialisation. */
    // summary of inferencing settings (from model_metadata.h)
    printf("Edge Impulse Inferencing settings:\n");
    printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) / sizeof(ei_classifier_inferencing_categories[0]));

    //run_classifier_init();
#endif

    /* Setup cache poicy of tensor arean buffer */
    info("Set tesnor arena cache policy to WTRA \n");
    const std::vector<ARM_MPU_Region_t> mpuConfig =
    {
        {
            // SRAM for tensor arena
            ARM_MPU_RBAR(((unsigned int)tensor_arena),        // Base
                         ARM_MPU_SH_NON,    // Non-shareable
                         0,                 // Read-only
                         1,                 // Non-Privileged
                         1),                // eXecute Never enabled
            ARM_MPU_RLAR((((unsigned int)tensor_arena) + EI_CLASSIFIER_TFLITE_LARGEST_ARENA_SIZE - 1),        // Limit
                         eMPU_ATTR_CACHEABLE_WTRA) // Attribute index - Write-Through, Read-allocate
        },
    };

    // Setup MPU configuration
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

    //Edge Impulse structure initial
    signal_t signal;            // Wrapper for raw input buffer
    ei_impulse_result_t result = {0}; // Used to store inference output
    EI_IMPULSE_ERROR res;       // Return code from inference

    // Calculate the length of the buffer
    size_t buf_len = sizeof(input_buf) / sizeof(input_buf[0]);

    // Make sure that the length of the buffer matches expected input length
    if (buf_len != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE)
    {
        printf("ERROR: The size of the input buffer is not correct.\r\n");
        printf("Expected %d items, but got %d\r\n",
               EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE,
               (int)buf_len);
        return 1;
    }

    pmu_reset_counters();

    // Assign callback function to fill buffer used for preprocessing/inference
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &get_signal_data;
    //numpy::signal_from_buffer(&input_buf[0], EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);

    // Perform DSP pre-processing and inference
    printf("\r\n");
    printf("run_classifier\r\n");
    res = run_classifier(&signal, &result, false);

    // Print return code and how long it took to perform inference
    printf("run_classifier returned: %d\r\n", res);
    printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
           result.timing.dsp,
           result.timing.classification,
           result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    printf("Object detection bounding boxes:\r\n");

    for (uint32_t i = 0; i < EI_CLASSIFIER_OBJECT_DETECTION_COUNT; i++)
    {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];

        if (bb.value == 0)
        {
            continue;
        }

        printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
               bb.label,
               bb.value,
               bb.x,
               bb.y,
               bb.width,
               bb.height);
    }

    // Print the prediction results (classification)
#else
    printf("Predictions:\r\n");

    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
    {
        printf("  %s: ", ei_classifier_inferencing_categories[i]);
        printf("%.5f\r\n", result.classification[i].value);
    }

#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
    printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

#define EACH_PERF_SEC 5
    uint64_t u64PerfCycle;
    uint64_t u64PerfFrames = 0;

    u64PerfCycle = pmu_get_systick_Count();
    u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);


    while (1)
    {
        //__NOP();

        // Perform DSP pre-processing and inference
        res = run_classifier(&signal, &result, false);

        u64PerfFrames ++;

        if (pmu_get_systick_Count() > u64PerfCycle)
        {
            info("Model inference rate: %llu inf/s \n", u64PerfFrames / EACH_PERF_SEC);
            info("Accumulated time: %llu (s) \n", pmu_get_systick_Count() / SystemCoreClock);
            u64PerfCycle = pmu_get_systick_Count();
            u64PerfCycle += (SystemCoreClock * EACH_PERF_SEC);
            u64PerfFrames = 0;

            // output
            // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
            printf("Object detection bounding boxes:\r\n");

            for (uint32_t i = 0; i < EI_CLASSIFIER_OBJECT_DETECTION_COUNT; i++)
            {
                ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];

                if (bb.value == 0)
                {
                    continue;
                }

                printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                       bb.label,
                       bb.value,
                       bb.x,
                       bb.y,
                       bb.width,
                       bb.height);
            }

#else
            // Print the prediction results (classification)
            info("Predictions:\r\n");

            for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++)
            {
                printf("  %s: ", ei_classifier_inferencing_categories[i]);
                printf("%.5f\r\n", result.classification[i].value);
            }

#endif

            // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY == 1
            printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

        }
    }
}