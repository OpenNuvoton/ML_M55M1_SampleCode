# PoseLandmark_YOLOv8n
A demonstration sample for YOLOv8n-pose model
## Requirement
1. Keil uVision5
## Howto
1. Build by Keil
2. Copy Model/YOLOv8n-pose.tflite file to SD card root directory.
3. Insert SD card to NUMAKER-M55M1 board
4. Run
## Performance
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|YOLOv8n-pose|192x192x3|2240|300| 32.7|

Total frame rate: 17 fps