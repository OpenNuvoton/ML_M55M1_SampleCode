# ObjectDetection_YOLOv8n
A demonstration sample for YOLOv8n-od model
## Requirement
1. Keil uVision5
## Howto
1. Build by Keil
2. Copy Model/YOLOv8n-od.tflite file to SD card root directory.
3. Insert SD card to NUMAKER-M55M1 board
4. Run
## Performance
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|YOLOv8n-od|192x192x3|2161|300| 34.0|

Total frame rate: 16 fps