# YOLOX-Nano Object Detection Sample #

This is a YOLOX-Nano neural network inference sample for object detection running on the Nuvoton M55M1 microcontroller with the Zephyr RTOS. Here's what it does:

Key Features:
1. Real-time Object Detection: Uses the YOLOX-nano model (a lightweight variant of YOLO) to detect objects in images captured from an image sensor (HM1055)
2. Hardware Acceleration:
    - Leverages the Ethos-U NPU (Neural Processing Unit) for accelerated AI inference
    - Uses HyperRAM for model storage (external high-capacity memory)
    - Supports I/D caching for improved performance
    - Custom SRAM2 configuration for optimal memory layout
3. Image Capture & Processing:
    - Captures images from an HM1055 image sensor via CCAP (Camera Capture)
    - Resizes images to the model's input dimensions (RGB565 → RGB888)
    - Quantizes data for efficient model inference
    - Draws bounding boxes on detected objects
4. Multi-threaded Architecture:
    - Main task: Handles image capture and display
    - Inference task: Processes AI inference independently
    - Uses Zephyr message queues for thread synchronization
5. Optional Features:
    - UVC (USB Video Class): Stream results image over USB
    - SD Card Support: Load the model from SD card instead of embedding it
    - Profiling: Performance monitoring and cycle counting
    - LCD display: Show results image over LCD

Software Stack:
- TensorFlow Lite Micro with CMSIS-NN optimization
- Arm ML Embedded Evaluation Kit for common ML utilities
- OpenMV (OMV) library for image processing
- Zephyr OS for real-time task scheduling
- FatFS for SD card file system

## Model Information ##
YOLOX (You Only Look Once – X) is a high-performance, anchor-free object detection framework.
It improves upon earlier YOLO versions (YOLOv3–YOLOv5) by introducing:
- Anchor-free design → simplifies training and improves generalization.
- Decoupled head architecture → separates classification and localization tasks for better accuracy.
- Advanced training strategies like label assignment and strong augmentation.

YOLOX-Nano is an ultra-lightweight YOLOX model optimized for low-power devices (MCUs, mobile CPUs, NPUs). It offers lower accuracy compared to larger YOLOX models (YOLOX-S, YOLOX-M and YOLOX-L), but much faster and lighter.

|Information||
|:----|:----|
|Framework|PyTorch|
|Paper|https://arxiv.org/abs/2107.08430|
|Provenance|https://github.com/Megvii-BaseDetection/YOLOX|
|Parameters| ~0.9M|
|COCO mAP(0.5:0.95)| 0.200|
|Model ROM size|1289KB|
|Model RAM(arena) size|802KB|

[NuEdgeWise](https://github.com/OpenNuvoton/ML_YOLO) provides YOLOX Nano-related transfer learning scripts and model conversion tools (from PyTorch to tflite). You can customize your own classes in this environment and deploy to M55M1.

## Memory(RAM) Region ##
This sample demonstrates the regions of the M55M1 memory in the following:

| Region | Address | Size | Memory Type | Data Context | DTC Overlay | Memory Attribute |
|:----|:----|:----|:----|:----|:----|:----| 
|RAM|0x20100000|128KB|Part of SRAM0|Kernel read-write data|sram0_128K.overlay|DT_MEM_ARM_MPU_RAM|
|DTCM|0x20000000|128KB|DTCM|System heap and stack||DT_MEM_ARM_MPU_RAM|
|SRAM2|0x20200000|320KB|SRAM2|Non-cachable, CCAP image frame buffer|sram2_region.overlay|DT_MEM_ARM_MPU_RAM_NOCACHE|
|SRAM_HYPERRAM|0x81F20000|4000KB|Part of SRAM0, SRAM1 and HyperRAM|Model arena cache|sram_hyperram_region.overlay|DT_MEM_ARM_MPU_RAM_NOCACHE|
|EBI0|0x60000000|1024KB|EBI0|MPU-type LCD device|ebi_lcd_region.overlay|DT_MEM_ARM_MPU_DEVICE|

## Setup and Build ##
This sample is based on Zephyr RTOS. Please follow the steps below to install and setup the compilation environment.
1. Follow [Zephyr IDE with NuMicro Cortex-M on VSCode](../../Doc/Zephyr/Zephyr%20IDE%20with%20NuMicro%20Cortex-M%20on%20VSCode.md) guiding to install Zephyr host tool, SDK and workspace. Make sure the zephyr revision of workspace >= v4.3.0

![ws revision](pic/workspace_revision.png)

2. Install tflite-micro external module
```
west config manifest.project-filter -- +tflite-micro
west update
```
![tflite-micro](pic/tflite_micro_install.png)

3. Add sample project

![Add project](pic/add_project.png)

![Select folder](pic/select_project_folder.png)

4. Add project build

![Add build](pic/add_build.png)

![select build folder](pic/select_build_folder.png)

![select build folder_1](pic/select_build_folder_1.png)

![pick board](pic/pick_board.png)

![setup_build_folder](pic/setup_build_folder.png)

![setup build optimize](pic/setup_build_optimize.png)

![setup build config](pic/setup_build_conf.png)

![setup cmake argu](pic/setup_cmake_argu.png)

![add build result](pic/add_build_result.png)

5. Add DTC overlay

![add DTC overlay](pic/add_dtc_overlay.png)

![add DTC extra](pic/add_DTC_extra.png) 

![add DTC files](pic/add_DTC_files.png)

![add DTC finish](pic/Add_DTC_finish.png)

![add DTC result](pic/Add_DTC_result.png)

6. Pristine build

![pristine build](pic/build_pristine.png)

## Program Flash and Debug ##
Please follow [Zephyr IDE with NuMicro Cortex-M on VSCode](../../Doc/Zephyr/Zephyr%20IDE%20with%20NuMicro%20Cortex-M%20on%20VSCode.md) guiding to setup openocd runner.

## Configuration ##
This sample supports the following application configurations. You can change the settings by MenuConfig/GuiConfig, or modify prj.conf file directly.
- ```CONFIG_APP_OD_PROFILE_ENABLED``` (default n): Enable/disable application profiling on each stage, which includes NPU inference, CCAP capture, image resize, post processing cycles.
- ```CONFIG_APP_OD_UVC_SHOW_IMAGE``` (default y): Enable/disable the display of the result image over UVC connect.
- ```CONFIG_APP_OD_MODEL_FROM_SD``` (default n): Support loading model from SD card to HyperRAM
- ```CONFIG_APP_OD_USING_HYPERRAM``` (default n): Using HyperRAM for arena/model space 
- ```CONFIG_APP_OD_LCD_SHOW_IMAGE``` (default n): Enable/disable the display of the result image over LCD.

## Performance
1. Memory usage

![memory usage](pic/memory_usage.png)  

PS. The SRAM_HYPERRAM region only uses 816KB, meaning only the M55M1 SRAM is actually used. Therefore, the ```CONFIG_APP_OD_USING_HYPERRAM``` can be disabled.

2. Frame rate and inference rate  
System clock: 220MHz  

| Model |Input Dimension | Model Inference Rate (inf/sec) |  
|:------|:---------------|:-------------------------|
|YOLOX-Nano|320x320x3| 43.6|  

|Display(show result)| Applicaton Frame Rate (fps) |  
|:------|:-------------------------|
|Console| 16|
|Console + UVC| 14|
|Console + LCD| 15|

