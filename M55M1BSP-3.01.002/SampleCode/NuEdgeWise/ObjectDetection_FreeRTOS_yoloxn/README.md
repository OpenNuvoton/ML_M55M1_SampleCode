# ObjectDetection_YOLOv8n
A demonstration sample for [yolox-nano-ti-tflite](M55M1BSP-3.00.001/SampleCode/NuEdgeWise/ObjectDetection_FreeRTOS_yoloxn/README.md) model
## Requirement
1. Keil uVision5
## Howto
1. Build by Keil
2. Run
## Performance
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|yolox-nano-ti-tflite|320x320x3|1277|802| 29.1|

Total frame rate: 13 fps
Accuracy: mAP(0.5:0.95): 0.200, mAP(0.5): 0.337