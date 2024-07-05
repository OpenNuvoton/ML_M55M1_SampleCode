# NuEdgeWise
Collect sample codes related to machine learning on M55M1.
## Sample Codes
|Sample Code|Use case|Framework|Model|Description|Note|
|:------------|:-------- |:----------|:------|:------------| :------------|
|HandLandmrk|Hand posture recognition | TFLM | HandLandmark |Example of hand landmark. Reference source comes from MediaPipe||
|NN_ModelEasyDeploy|Image classification |TFLM|MobileNetV2|Demo easily deploy new model and label to target||
|ObjectDetection_FreeRTOS_yoloxn|Object detection |TFLM|yolox-nano-ti-nu|Example of yolox-nano inference, including coco80, medicine, and hand gesture|320X320 model only need SRAM&FLASH|
|_NN_ExecuTorch||executorch|| Template sample for executorch Arm backend |Experimental|
|HandPoseRecognition|Hand posture recogniton|TFLM|HandLandmark and PointHistoryClassifier|Classify the current hand posture is stopped, moving, clockwise or counter clockwise||
|PoseLandmark|Pose detection|TFLM|PoseLandmark|Detect landmarks of human body||