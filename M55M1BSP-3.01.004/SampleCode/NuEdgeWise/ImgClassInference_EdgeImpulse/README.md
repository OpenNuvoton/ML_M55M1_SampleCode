# Edge Impulse's image classification Example for M55M1 NuMaker
A demonstration sample for image classification with Edge Impulse's C++ SDK
## Requirement
1. Keil uVision5 or VSCode CMSIS
## Howto
1. Download your deployment package (Ethos-U55-256 library (High End Embedded, Shared SRAM)) from Edge Impulse’s Deployment page and unzip it.
2. Replace the `model-parameters` and `tflite-model` folders inside the `edgeimpulse_model/` folder.
    1. add `uint8_t tensor_arena[<kTensorArenaSize of your model>]	__attribute__((aligned(16), section(".bss.NoInit.activation_buf_sram")));`  in your`tflite_learn_XXXXXX_3_compiled.cpp`.
    2. comment out the original `uint8_t tensor_arena`.
    3. Update tensor_arena's size at `mpuConfig` in `main.cpp`.
3. Build by Keil or VSCode CMSIS.
4. The required libraries are prebuilt in `ThirdParty/edgeimpulse/Lib`, including tflite micro, CMSIS-DSP and CMSIS-NN.
5. Run.



