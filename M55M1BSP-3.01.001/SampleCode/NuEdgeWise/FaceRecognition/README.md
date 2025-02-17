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
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|Yolo fastest|192x192x1|441|443|131.1|
|MobileFaceNet|112x112x3|3249|305|43.4|

Total frame rate: 14 fps


