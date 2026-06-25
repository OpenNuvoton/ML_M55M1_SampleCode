
# Battery State of Charge (SOC) Estimation Sample

## **Overview**

This repository contains a C++ application designed to run on Nuvoton M55M1. It leverages an artificial neural network (via TensorFlow Lite for Microcontrollers and ARM CMSIS-NN) to estimate a battery's State of Charge (SOC) based on time-series inputs.

The application initializes the board, loads a pre-trained .tflite model (either directly from memory or an SD card), processes input telemetry, runs inference, and calculates the Mean Absolute Error (MAE) of the predictions against expected test data.

## **Features**

* **Machine Learning Inference:** Utilizes arm::app::NNModel to run quantized TFLite models on edge hardware.  
* **Flexible Model Loading:** Supports loading the model directly from an SD card into HyperRAM or using a compiled-in C-array.  
* **Data Normalization and Quantization:** Automatically scales and quantizes raw floating-point test data (Voltage, Current, Temperature, State of Health) into int8\_t for efficient edge inference.  
* **Memory Optimization:** Configures the Memory Protection Unit (MPU) to apply a Write-Through, Read-Allocate (WTRA) cache policy to the Tensor Arena for improved performance.  
* **Profiling and Logging:** Includes optional profiling hooks and customizable logging levels to measure inference speeds and debug execution.

## **Model Information**
SOC estimation is a deep learning-based Battery Management System (BMS) model architecture designed to accurately predict the State of Charge (SOC) of lithium-ion batteries using discharge data.

|Information||
|:----|:----|
|Framework|TensorFlow Lite|
|Quantization| INT8|
|Provenance|https://github.com/chchen59/BatteryStateEstimation|

## **Build Configurations (Macros)**

You can customize the build behavior by toggling the following macros in the source code or your build system:

* \_\_LOAD\_MODEL\_FROM\_SD\_\_: If defined, the application will look for nn\_model.tflite on the SD card and load it into HyperRAM at address 0x82400000.    
* \_\_PROFILE\_\_: Enables cycle counting and profiling using arm::app::Profiler.  
* LOG\_LEVEL: Adjusts the verbosity of the console output (e.g., 2 for INFO, 0 for TRACE).

## **Data Processing**

The model expects a specific time-series sequence as input. The input variables include Voltage, Current, Temperature, and SOH.

Before inference, the application performs Min-Max normalization followed by quantization using the model's input tensor parameters:

1. **Normalization:** The raw floating-point data is normalized using pre-defined min/max scales from Pattern/SOC\_test\_data.h.  
2. **Quantization:** The normalized data is mapped to an int8\_t space using the tensor's scale and offset.

## **Getting Started**

#### 1\. Prepare the Model

Rename Model/BMS_SOC_INT8_vela.tflite to nn\_model.tflite and place it in the root directory of your SD Card.

#### 2\. Configure the Test Data

The application expects test patterns and ground-truth data to be defined in Pattern/SOC\_test\_data.h. Ensure this header file includes:

* test\_x\_seq: The raw input sequence data.  
* test\_y\_seq: The ground-truth SOC values.  
* test\_x\_seq\_dim: Dimensions of the input data \[samples, time\_steps, features(V, I, T, SOH)\].  
* normalize\_scale\_max and normalize\_scale\_min: Arrays containing the maximum and minimum values used for scaling the raw data.

#### 3\. Build and Flash

Compile the project using Keil and flash the firmware to your Nuvoton board.


## **Execution Flow**

1. **Hardware Initialization:** BoardInit() configures the system clocks and UART for printf output.  
2. **Model Instantiation:** Reads the model file from the SD card (if configured) or gets the model pointer.  
3. **Tensor Arena Setup:** Allocates a 9 KB buffer (ACTIVATION\_BUF\_SZ) for intermediate neural network calculations and configures the MPU cache policy.  
4. **Inference Loop:**  Iterates through the test data sequences.  
   * Normailze and quantizes input raw data and feeds it into the model.  
   * Executes model.RunInference().  
   * Dequantizes the output tensor back to floating-point values.  
5. **Evaluation:** Calculates the continuous Mean Absolute Error (MAE) across all samples.

## **Output**

When run, the console will output:

* Initialization status and model file size.  
* The current MPU cache policy.  
* Real-time inference rates (Inferences per second).  
* The final Prediction MAE score evaluating the accuracy of the SOC estimations against the test dataset.

## **Performance**
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|SOC(LSTM)|8x4|1798|9|19.1|
