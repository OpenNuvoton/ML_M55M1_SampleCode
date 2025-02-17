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
|FaceLandmark|Face landmark|TFLM|Yolo fastest and FaceLandmark|Detect face landmarks||
|FaceDetection|Face detection|TFLM|Yolo fastest|Detect face region||
|PoseLandmark_YOLOv8n|Pose detection|TFLM|YOLOv8n-pose|Detect landmarks of human body||
|FaceEnrollment|Face recogniton|TFLM|Yolo fastest and mobilefacenet|Enrollment face features||
|FaceRecognition|Face recogniton|TFLM|Yolo fastest and mobilefacenet|Face recognition||
|ImageClassification|Image classification|TFLM|MobileNetV2|Image object classification||
|ImageClassification_TVM|Image classification|TVM|MobileNetV2|Image object classification||
|AnomalyDetection|Anomaly detetcion|TFLM|AutoEncoder|Anomaly detection using IMU sensor||
|ObjectDetection_YOLOv8n|Object detection| TLFM|YOLOv8n|Example of YOLOv8n inference||
|AudioDenoise|Audio denoise|TFLM|RNNoise|Audio RNN denoise sample||
|SafetyRecognition|Face and fingerprint recognition|TFLM|Yolo fastest, mobilefacenet and anti-spoof model|Demonstrate MobileFaceNet recognition with antiSpoofing and fingerprint module together||
|ImageSegmentation|Image segmentation|TFLM|Deeplab_v3|Image object segmentation||
|FaceLandmark_PoseCheck|Face pose check |TFLM|Yolo fastest, FaceLandmark and DNN|Detect face landmarks and use them for classification||