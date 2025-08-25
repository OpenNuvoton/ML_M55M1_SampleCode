# Edge Impulse's Simple Example for M55M1 NuMaker
A demonstration sample for Edge Impulse's C++ SDK
## Requirement
1. Keil uVision5 or VSCode CMSIS
## Howto
1. Download your deployment package (Ethos-U55-256 library (High End Embedded, Shared SRAM)) from Edge Impulse’s Deployment page and unzip it.
2. Replace the `model-parameters` and `tflite-model` folders inside the `edgeimpulse_model/` folder.
1. Build by Keil or VSCode CMSIS.
2. The required libraries are prebuilt in `ThirdParty/edgeimpulse/Lib`, including tflite micro, CMSIS-DSP and CMSIS-NN.
2. Run.



