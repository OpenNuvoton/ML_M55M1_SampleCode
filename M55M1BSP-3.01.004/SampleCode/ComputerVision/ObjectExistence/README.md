# ObjectExistence
A demonstration of object existence detection.
This example showcases the detection of objects in a relatively simple background scenario using computer vision methods.
## Requirement
1. Keil uVision5
## Howto
1. Build by Keil
2. Run
3. Users may need to adjust certain image processing parameters for different application scenarios:
    - For image_binary and find_blobs: LMin, LMax, AMin, AMax, BMax, BMin
    - For dilate and erode: k_size and iter(this setting will impact efficiency)
    - For find_blobs: area_threshold_val, pixels_threshold_val
## Performance
System clock: 180MHz
Using OpenMV
Methods: absDifference, binary, dilate & erode and finding blobs
Total frame rate: 2 fps


