# HandLandmark
A demonstration sample for hand landmark
## Requirement
1. Keil uVision5
2. SD card
## Howto
1. Build by Keil
2. Copy Model/hand_landmark.tflite file to SD card root directory.
3. Insert SD card to NUMAKER-M55M1 board
4. Run
## Performance
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|HandLandmark|224x224x3|2216|1011|39.5|

Total frame rate: 17 fps


