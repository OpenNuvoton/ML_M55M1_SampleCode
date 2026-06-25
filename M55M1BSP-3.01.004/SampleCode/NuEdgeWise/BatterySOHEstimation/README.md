# Battery SOH Estimation

## **Overview**

This project implements an embedded machine learning application for estimating the State of Health (SOH) of lithium-ion batteries. Designed for Nuvoton microcontrollers, it runs quantized neural network models (such as those trained in TensorFlow/Keras) directly on the edge device. By analyzing time-series data of battery charging cycles (Voltage, Current, and Temperature), it provides real-time, on-device health predictions.

## **Key Features**

* **On-Device AI**: Utilizes TensorFlow Lite for Microcontrollers (TFLM) to perform inference without cloud connectivity.  
* **Dynamic Model Loading**: Supports reading .tflite models directly from an SD Card into the MCU's HyperRAM, facilitating easy model updates without reflashing the firmware.  
* **Optimized Memory Management**: Configures hardware Memory Protection Unit (MPU) caching policies for efficient tensor arena operations.  
* **Integrated Preprocessing**: Automatically normalizes and quantizes (int8\_t) raw sensor data on the fly before feeding it to the input tensor.  
* **Performance Profiling**: Built-in Systick-based profiling to monitor inference rate (frames per second) and compute the Mean Absolute Error (MAE) against test datasets.

## **Model Information**
SOH estimation is a deep learning-based Battery Management System (BMS) model architecture designed to accurately predict the State of Health (SOH) of lithium-ion batteries using charge data.

|Information||
|:----|:----|
|Framework|TensorFlow Lite|
|Quantization| INT8|
|Provenance|https://github.com/chchen59/BatteryStateEstimation|

## **System Requirements**

* **Hardware**: NuMaker-M55M1 with HyperRAM support and SD Card interface.  
* **Software**:  
  * ARM CMSIS-NN Library  
  * TensorFlow Lite for Microcontrollers (TFLM)  
  * FatFs (for SD card file I/O)

## **Getting Started**

#### 1\. Prepare the Model

Rename Model/BMS_SOH_INT8_vela.tflite to nn\_model.tflite and place it in the root directory of your SD Card.

#### 2\. Configure the Test Data

The application expects test patterns and ground-truth data to be defined in Pattern/SOH\_test\_data.h. Ensure this header file includes:

* test\_x: The raw input sequence data.  
* test\_y: The ground-truth SOH values.  
* test\_x\_dim: Dimensions of the input data \[cycles, sequence\_length, features(V, I, T)\].  
* normalize\_scale\_max & normalize\_scale\_min: Arrays containing the maximum and minimum values used for scaling the raw data.

#### 3\. Build and Flash

Compile the project using Keil and flash the firmware to your Nuvoton board.

#### 4\. Execution

Upon booting, the system will:

1. Initialize the board and UART console.  
2. Mount the SD card and copy nn\_model.tflite into HyperRAM.  
3. Initialize the Tensor Arena.  
4. Iterate through the charging cycles defined in the test data, running inference on each step.  
5. Output the individual SOH predictions and the final Mean Absolute Error (MAE) via the serial console.

## **Serial Console Output**

When running, you can monitor the application via UART. It will print the model loading status, inference speed (inferences/sec), the dequantized output tensors, and the final prediction MAE.

## **Performance**
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|SOH(CNN)|856x3|2220|214|50|

