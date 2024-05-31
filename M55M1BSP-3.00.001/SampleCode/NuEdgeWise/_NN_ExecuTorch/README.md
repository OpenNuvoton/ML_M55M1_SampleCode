# NN_ExecuTorch
A template sample code for executorch Arm backend
## Requirement
1. [executorch](https://github.com/pytorch/executorch)
2. Keil uVision5
## Howto
1. Compile your pytorch model by executorch. It will generate a model .pte file.[^1]
2. Covert .pte file to a C arrary header (.pth.h) file
3. Include .pte.h in main.cpp



[^1]:[Building and Running ExecuTorch with ARM Backend](https://pytorch.org/executorch/stable/executorch-arm-delegate-tutorial.html)
