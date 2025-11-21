# M55M1 Zephyr Neural Network Inference Samples
This repository contains machine learning inference examples for the NuMicro M55M1 microcontroller running Zephyr RTOS, featuring ARM Cortex-M55 with Ethos-U55 NPU support.

## Repository Structure ##
```
SampleCode/
├── boards/
│   └── numaker.txt
├── ObjectDetection_Zephyr_yoloxn/   # YOLOX-nano object detection example
ThirdParty/
├── ml-embedded-evaluation-kit/      # ARM ML embedded evaluation kit
└── openmv/                         # OpenMV computer vision library
```
## Samples Overview ##
1. ObjectDetection_Zephyr_yoloxn  
Advanced object detection using YOLOX-nano model featuring:  
    - Real-time object detection with camera input  
    - Multi-threaded inference processing  
    - Post-processing for bounding box generation  
    - UVC (USB Video Class) output support  
    - Performance optimization with HyperRAM  

    Key Files:   
    - main.cpp - Main application with camera integration  
    - InferenceTask.cpp - Inference task management  
    - DetectorPostProcessing.cpp - Object detection post-processing  

## Hardware Features ##
M55M1 Platform Support  
- ARM Cortex-M55 with Ethos-U55 NPU acceleration  
- HyperRAM integration for model storage (SampleCode/NN_Inference/src/Device/HyperRAM/hyperram_code.c)  
- SD Card support for model loading  
- Image sensor integration (HM1055)  
- USB Video Class (UVC) output  

Memory Configuration  
The projects use advanced memory management with overlay files:  
- sram_hyperram_region.overlay - HyperRAM memory region configuration  
- Memory-mapped regions for optimal performance  
- MPU (Memory Protection Unit) configuration for cache optimization  

## Key Technologies ##
Machine Learning Framework  
- TensorFlow Lite Micro integration via ml-embedded-evaluation-kit  
- ARM Ethos-U NPU acceleration with performance monitoring (SampleCode/NN_Inference/src/NPU/include/ethosu_profiler.h)  
- Quantized INT8 models for efficient inference

Computer Vision  
- OpenMV integration for image processing (ThirdParty/openmv)
- Real-time image capture and processing

RTOS Integration  
- Zephyr RTOS with multi-threading support  
- Message queue-based task communication  
- Real-time performance optimization  

## Environment Setup ##
Development Environment Tools
- VSCode
- Zephyr IDE extension pack
- NuMicro Cortex-M pack  

For detailed setup instructions, please refer to [Zephyr IDE with NuMicro Cotrext-M on VSCode](Doc/Zephyr/Zephyr%20IDE%20with%20NuMicro%20Cotrext-M%20on%20VSCode.md)