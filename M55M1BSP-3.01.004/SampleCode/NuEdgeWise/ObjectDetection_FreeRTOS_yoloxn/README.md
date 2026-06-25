# ObjectDetection_YOLOX
A demonstration sample for yolox-nano-ti-tflite model
## Requirement
1. Keil uVision5
## Howto
1. Build by Keil
2. Run
## Performances
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|yolox-nano-ti-tflite|320x320x3|1277|802| 29.1|

Total frame rate: 13 fps
Accuracy: mAP(0.5:0.95): 0.200, mAP(0.5): 0.337