# FaceRecognition
A demonstration sample for face recognition
## Requirement
1. Keil uVision5
## Howto
1. Build by Keil
2. Copy Model/face_mobilenet.tflite file to SD card root directory.
3. Insert SD card to NUMAKER-M55M1 board
4. Run
## Performance
System clock: 180MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|Yolo fastest|192x192x3|441|443| 109.8|
|MobileFaceNet|112x112x3|3249|305|39.1|

Total frame rate: 12 fps


