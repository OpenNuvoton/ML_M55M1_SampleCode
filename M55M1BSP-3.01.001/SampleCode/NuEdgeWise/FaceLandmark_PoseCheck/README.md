# FaceLandmark_PoseCheck
A demonstration sample for face landmarks that uses the landmarks to determine whether the head pose is normal or not.
## Requirement
1. Keil uVision5
## Howto
1. Build by Keil
2. Run
## Performance
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|Yolo fastest|192x192x1|441|443|131.1|
|FaceLandmark|192x192x3|679|460|38.6|
|FaceLandmark classification|1x936|19|0.625|12573|

Total frame rate: 13 fps

## Training model
1. Please contact Nuvoton to request access to these training scripts. [Link](https://www.nuvoton.com/ai/contact-us/)
2. Includes dataset preparation, training, conversion to TFLite, and testing.


