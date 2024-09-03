# HandPoseRecognition
The hand pose recognition sample is reference to Kazuhito's [hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) to complete. In this sample, it used two models. First, using the hand landmark model to detect the keypoint localization of 21 hand-knuckle coordinates. Then record the trajectory of the index finger keypoint and use a point history classifier model to classify the current hand posture is stopped, moving, clockwise or counter clockwise.
## Requirement
1. Keil uVision5
2. SD card
## Howto
1. Build by Keil
2. Copy Model/hand_landmark.tflite file to SD card root directory.
3. Insert SD card to NUMAKER-M55M1 board
4. Run
## Performance
System clock: 180MHz
| Model |Input Dimension |ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|HandLandmark|224x224x3|2216|1011|32|
|PointHistoryClassifier|32|74|0.7|2500|

Total frame rate: 11 fps

