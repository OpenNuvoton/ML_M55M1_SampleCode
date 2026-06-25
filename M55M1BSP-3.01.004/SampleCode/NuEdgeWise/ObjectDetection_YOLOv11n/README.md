## Overview
This project demonstrates real-time image object detection on the edge using the **YOLOv11 nano (YOLOv11n)** model deployed on the **Nuvoton M55M1** microcontroller. The neural network operations are accelerated by the on-chip **ARM Ethos-U55 NPU**, achieving efficient INT8 inference. 

The application captures live imagery from an attached image sensor, processes it through the YOLOv11n object detection model, and outputs the bounding boxes and class labels to an LCD display and over USB Video Class (UVC).
## Key Features
* **Edge AI Acceleration:** Utilizes the ARM Ethos-U55 NPU for hardware-accelerated TFLite Micro inference.
* **YOLOv11n Object Detection:** Runs a pre-compiled, INT8-quantized YOLOv11 nano object detection model (`vela` optimized).
* **Real-time Sensor Input:** Integrates with the HM1055 camera sensor via SWI2C.
* **Multi-Display Support:** Includes drivers for various LCD modules (ILI9341, FSA506, LT7381).
* **USB Video Class (UVC):** Streams the output frames to a host PC as a standard USB webcam.
* **Performance Profiling:** Built-in PMU (Performance Monitoring Unit) counters to profile CPU and NPU cycle counts.
## Directory Structure
```
ObjectDetection_YOLOv11n/
├── Device/                # Hardware peripheral drivers
│   ├── Display/           # LCD drivers and PDMA configurations
│   ├── HyperRAM/          # HyperRAM initialization
│   ├── ImageSensor/       # HM1055 camera sensor drivers
│   ├── SDCard/            # SD Card glue logic
│   └── UVC/               # USB Video Class descriptors and logic
├── Keil/                  # Keil uVision 5 MDK project files
├── Model/                 # Neural Network models and wrappers
│   ├── YOLOv11n-od.tflite # NPU-optimized TFLite model
│   ├── YOLOv11nODModel.cpp # Model initialization and invocation wrapper
│   └── Labels.cpp         # Class labels
├── NPU/                   # ARM Ethos-U NPU drivers and cache management
├── ProfilerCounter/       # PMU and inference profiling tools
├── main.cpp               # Main application loop
├── BoardInit.cpp          # Board-specific clock and pinmux setup
├── YOLOv11nODPostProcessing.cpp # Post-processing for YOLOv11n bounding boxes
└── board_config.h         # Global hardware configuration definitions
```
## Hardware Requirements
* MCU: Nuvoton M55M1 Evaluation Board
* Camera: HM1055 Image Sensor module
* Display: Compatible LCD module (ILI9341, FSA506, or LT7381)
* Memory: External HyperRAM module
* Debugger: Nu-Link (for flashing and debugging via Keil)
## Software Requirements
* IDE: Keil MDK (uVision 5) with M55M1 device packs installed.
* Compiler: ARM Compiler 6 (AC6).
* Model Optimization: ARM Vela compiler (if you intend to recompile custom `.tflite` models for the Ethos-U55 NPU).
## Getting Started
1. Hardware Setup: Connect the HM1055 camera module, LCD display, and HyperRAM to their respective headers on the M55M1 board. Connect the board to your PC via USB.
2. Prepare Model: Copy Model/YOLOv11n-od.tflite file to SD card root directory.
3. Open Project: Navigate to the `Keil/` directory and open `ObjectDetection.uvprojx` in Keil uVision.
4. Build: Click the Build button (F7) to compile the project.
5. Flash: Ensure your Nu-Link debugger is connected and click Download (F8) to flash the firmware onto the M55M1.
6. Run: Press the reset button on the board. The application will initialize the NPU, camera, and display. Live segmentation masks will appear on the LCD.
## Post-Processing Details
The post-processing is handled in YOLOv11nODPostProcessing.cpp. It decodes the raw output tensors from the NPU, applies Non-Maximum Suppression (NMS) to filter redundant bounding boxes to generate the final bounding box for the objects.
## Performance
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|yolov11n-od|192x192x3|2432|921.6|23.3|

Total frame rate: 14 fps
