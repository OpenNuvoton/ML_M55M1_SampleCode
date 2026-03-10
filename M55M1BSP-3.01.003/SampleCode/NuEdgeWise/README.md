# NuEdgeWise
Collect sample codes related to machine learning on M55M1.
## Sample Codes
|Sample Code|Board|Use case|Framework|Model|Description|Note|
|:----------|:----|:-------|:--------|:----|:----------|:------------|
|HandLandmrk|NuMaker-X-M55M1D|Hand posture recognition | TFLM | HandLandmark |Example of hand landmark. Reference source comes from MediaPipe||
|NN_ModelEasyDeploy|NuMaker-X-M55M1D|Image classification |TFLM|MobileNetV2|Demo easily deploy new model and label to target||
|ObjectDetection_FreeRTOS_yoloxn|NuMaker-X-M55M1D|Object detection |TFLM|yolox-nano-ti-nu|Example of yolox-nano inference, including coco80, medicine, and hand gesture|320X320 model only need SRAM&FLASH|
|NN_ExecuTorch|NuMaker-X-M55M1D||executorch|| Template sample for executorch Arm backend ||
|HandPoseRecognition|NuMaker-X-M55M1D|Hand posture recogniton|TFLM|HandLandmark and PointHistoryClassifier|Classify the current hand posture is stopped, moving, clockwise or counter clockwise||
|PoseLandmark|NuMaker-X-M55M1D|Pose detection|TFLM|PoseLandmark|Detect landmarks of human body||
|FaceLandmark|NuMaker-X-M55M1D|Face landmark|TFLM|Yolo fastest and FaceLandmark|Detect face landmarks||
|FaceDetection|NuMaker-X-M55M1D|Face detection|TFLM|Yolo fastest|Detect face region||
|PoseLandmark_YOLOv8n|NuMaker-X-M55M1D|Pose detection|TFLM|YOLOv8n-pose|Detect landmarks of human body||
|FaceEnrollment|NuMaker-X-M55M1D|Face recogniton|TFLM|Yolo fastest and mobilefacenet|Enrollment face features||
|FaceRecognition|NuMaker-X-M55M1D|Face recogniton|TFLM|Yolo fastest and mobilefacenet|Face recognition||
|ImageClassification|NuMaker-X-M55M1D|Image classification|TFLM|MobileNetV2|Image object classification||
|ImageClassification_TVM|NuMaker-X-M55M1D|Image classification|TVM|MobileNetV2|Image object classification||
|AnomalyDetection|NuMaker-X-M55M1D|Anomaly detetcion|TFLM|AutoEncoder|Anomaly detection using IMU sensor||
|ObjectDetection_YOLOv8n|NuMaker-X-M55M1D|Object detection| TLFM|YOLOv8n|Example of YOLOv8n inference||
|AudioDenoise|NuMaker-X-M55M1D|Audio denoise|TFLM|RNNoise|Audio RNN denoise sample||
|SafetyRecognition|NuMaker-X-M55M1D|Face and fingerprint recognition|TFLM|Yolo fastest, mobilefacenet and anti-spoof model|Demonstrate MobileFaceNet recognition with antiSpoofing and fingerprint module together||
|ImageSegmentation|NuMaker-X-M55M1D|Image segmentation|TFLM|Deeplab_v3|Image object segmentation||
|FaceLandmark_PoseCheck|NuMaker-X-M55M1D|Face pose check |TFLM|Yolo fastest, FaceLandmark and DNN|Detect face landmarks and use them for classification||
|ObjectTracker_YOLOv8n|NuMaker-X-M55M1D|Object tracking|TFLM|YOLOv8n|Object tracking sample||
|ModelInference_EdgeImpulse|NuMaker-X-M55M1D|General Case|EdgeImpulse (TFLM)|Easy dnn|Model inference sample||
|KeywordSpotting_EdgeImpulse|NuMaker-X-M55M1D|KWS|EdgeImpulse (TFLM + EON)|Mobilenet|Key word spotting with DMIC||
|ImgClassInference_EdgeImpulse|NuMaker-X-M55M1D|Image classification|EdgeImpulse (TFLM + EON)|Mobilenet|Image classification with CCAP and UVC||
|DepthEstimation|NuMaker-X-M55M1D|Depth estimation|TFLM|FastDepth|Depth estimation from a RGB image||
|BatterySOHEstimation|NuMaker-X-M55M1D|Battery SOH estimation|TFLM|CNN|Battery SOH estimation from battery voltage, current and temperature||
|BatterySOCEstimation|NuMaker-X-M55M1D|Battery SOC estimation|TFLM|LSTM|Battery SOC estimation from battery voltage, current, temperature and SOH||
