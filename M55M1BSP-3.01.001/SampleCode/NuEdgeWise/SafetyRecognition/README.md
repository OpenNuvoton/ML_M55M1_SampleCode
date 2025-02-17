# Safety Recognition
- A demonstration of safety recognition sample. Demonstrate MobileFaceNet recognition
with antiSpoofing and fingerprint module together.
- Fingerprint recognition, if pass => face recognition, if pass => RGB only face anti-spoofing
## Requirement
1. Keil uVision5
## Howto
1. Build by Keil
2. Copy Model/face_mobilenet.tflite and Model/anti-spoof-mn3_full_integer_quant_per_tensor_vela.tflite or AntiSpoofing_80_cnnhead_vela.tflite files to SD card root directory.
3. Insert SD card to NUMAKER-M55M1 board
4. Choose one of the anti-spoof models by the define __ANTI_MODEL_CNNH__ or __ANTI_MODEL_MN3S__
5. Enable or disable the fingerprint module by the define __USE_FINGERP__
6. Run
## Performance
System clock: 220MHz
| Model |Input Dimension | ROM (KB) | RAM (KB) | Inference Rate (inf/sec) |  
|:------|:---------------|:--------|:--------|:-------------------------|
|Yolo fastest|192x192x3|441|443| 109.8|
|MobileFaceNet|112x112x3|3249|305|39.1|
|anti-spoof Mobilenet3|128x128x3|2753|323|55.6|
|anti-spoof CNN|80x80x3|509|525|41.36|

- Not detect the recognized face frame rate: 12 fps 
- If detect the recognized face frame rate: Mobilenet3: 4 fps, CNN: 7 fps

## Reference
- anti-spoof Mobilenet3: [light-weight-face-anti-spoofing](https://github.com/kprokofi/light-weight-face-anti-spoofing)
- anti-spoof CNN: [CNN Face-AntiSpoofing](https://github.com/hairymax/Face-AntiSpoofing)


