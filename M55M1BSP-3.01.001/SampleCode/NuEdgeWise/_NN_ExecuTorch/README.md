# NN_ExecuTorch
A template sample code for executorch Arm backend
## Requirement
1. [executorch](https://github.com/pytorch/executorch) v0.7
2. Keil uVision5
## Howto
1. Compile your pytorch model by executorch. It will generate a .pte model file.[^1]
2. Copy .pte file to SD card and rename to model.pte
3. Insert SD card to NUMAKER-M55M1 board
4. Run



[^1]:[Building and Running ExecuTorch with ARM Backend](https://pytorch.org/executorch/stable/executorch-arm-delegate-tutorial.html)
