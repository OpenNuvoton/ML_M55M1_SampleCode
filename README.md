# ML_M55M1_SampleCode
M55M1 machine learning application project. Supported the following two application frameworks.
- BSP samples  
The sample codes were developed based on the M55M1 BSP environment.
- Zephyr samples   
The sample codes were developed based on the Zephyr environment.

## Requirement
- BSP samples
    1. M55M1BSP V3.01.004
    2. Keil uVision5
- Zephyr samples
    1. VSCode
    2. Zephyr IDE extension pack
    3. Nuvoton cortex-M pack

## Install for BSP samples
- Manual  
    1. Download M55M1BSP from [BSP release](https://github.com/OpenNuvoton/M55M1BSP/releases)
    2. Unzip BSP zip file
    3. Copy patch files to BSP
- Auto
```
python install.py
```
- Folder structure
```
M55M1BSP-3.01.004
|--- Document
|--- Library
|--- SampleCode
|    |--- CotrexM55
|    |--- Crypto
|    |--- FreeRTOS
|    |--- Hard_Fault_Sample
|    |--- ISP
|    |--- MachineLearning
|    |--- NuEdgeWise
|    |--- NuMaker_M55M1
|    |--- PowerDelivery
|    |--- PowerManagement
|    |--- SecureApplication
|    |--- Semihost
|    |--- StdDriver
|    |--- Template
|    |--- TrustZone
|    |--- XOM
|--- ThirdParty
|    |--- executorch
|    |--- FatFs
|    |--- FreeRTOS
|    |--- libjpeg
|    |--- libmad
|    |--- lwIP
|    |--- mbedtls
|    |--- ml-embedded-evaluation-kit
|    |--- openmv
|    |--- paho.mqtt.embedded-c
|    |--- shine
|    |--- tflite_micro
|--- LICENSE
|--- README.md

```

## Install for Zephyr samples

Please reference [Zephyr IDE with NuMicro Cotrext-M on VSCode](ZephyrSamples/Doc/Zephyr/Zephyr%20IDE%20with%20NuMicro%20Cortex-M%20on%20VSCode.md)