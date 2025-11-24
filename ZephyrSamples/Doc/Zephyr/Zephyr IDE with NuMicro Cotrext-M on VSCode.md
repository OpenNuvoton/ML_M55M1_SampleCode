---
title: Zephyr IDE with NuMicro Cotrext-M on VSCode
updated: 2025-11-21 09:42:25Z
created: 2025-10-27 06:28:29Z
latitude: 22.64839560
longitude: 120.32620850
altitude: 0.0000
---

* * *

**Required VSCode Extension**

1.  Nuvoton NuMicro Cortex-M pack
2.  Zephyr IDE Extension Pack

* * *

**Install Zephyr IDE**

1.  Install Zephyr IDE Extension Pack from VSCode  
    ![32a4535ecff90d2cf5a45469aea7bef0.png](../_resources/32a4535ecff90d2cf5a45469aea7bef0.png)
2.  Install Host Tools  
    ![a6b57619b8b93c5b00db31157e3d93bf.png](../_resources/a6b57619b8b93c5b00db31157e3d93bf.png)
3.  Install SDK(Toolchain)  
    Toolchain install path would be `C:\Users\chche\.zephyr_ide\toolchains`
![43af29d262dad98f997b1f3e754a0833.png](../_resources/43af29d262dad98f997b1f3e754a0833.png)  
![b7f7e103f8f6b09a85ca527463177d03.png](../_resources/b7f7e103f8f6b09a85ca527463177d03.png)  
![e1c1e5f0dc0390a73e23651add40779d.png](../_resources/e1c1e5f0dc0390a73e23651add40779d.png)  
![319b0806f6861dddc68679e418e2be72.png](../_resources/319b0806f6861dddc68679e418e2be72.png)  
![223cc2a898fa653a29935fb481efffc1.png](../_resources/223cc2a898fa653a29935fb481efffc1.png)  
4.  Open a workspace folder and run workspace setup  
![255b88efb2ced12c887587bd73f59798.png](../_resources/255b88efb2ced12c887587bd73f59798.png)  
![22ab97cc2a690a225f25ec207881b143.png](../_resources/22ab97cc2a690a225f25ec207881b143.png)  
![a33ab99cc0c067424edc5fd25ed9fe28.png](../_resources/a33ab99cc0c067424edc5fd25ed9fe28.png)  
![6fabead9080415449cd253be7c353720.png](../_resources/6fabead9080415449cd253be7c353720.png)  
![171fff85d2ad9af647bc5a27a32bd97d.png](../_resources/171fff85d2ad9af647bc5a27a32bd97d.png)  
5.  After a few minutes... The workspace folder would be  
![cc592de76e0b6ac4f3736ff1951f3be9.png](../_resources/cc592de76e0b6ac4f3736ff1951f3be9.png)  
6. Activate workspace  
![ad14229894d0f3820625fc225d8be0ac.png](../_resources/ad14229894d0f3820625fc225d8be0ac.png)  
7.  Run "West Update"  
![6c9e7b91468e9da9cf003fd6051df482.png](../_resources/6c9e7b91468e9da9cf003fd6051df482.png)  

* * *

**Create Project**

1.  Copy a basic sample from zephyr sample (Ex: E:\\Zephyr_Workspace\\external\\zephyr\\samples\\basic\\blinky) to your new project directory
2.  Modify new project name (NNInference) and CMakeLists.txt
3.  Add project to IDE  
    ![86496bea1d052e2bd29ea66b8beaa177.png](../_resources/86496bea1d052e2bd29ea66b8beaa177.png)  
    ![be8ec698839b49f6b66a6b9a2d52319f.png](../_resources/be8ec698839b49f6b66a6b9a2d52319f.png)  
    ![d8259278f40b8ddcac231b21b82f9b0c.png](../_resources/d8259278f40b8ddcac231b21b82f9b0c.png)

* * *

**Build Project**  
1.Add Build  
![b77b2491838e9d55ceabdfd1fc90bcb7.png](../_resources/b77b2491838e9d55ceabdfd1fc90bcb7.png)  
![22d8562f97c2a9d3b26912f940b2a34d.png](../_resources/22d8562f97c2a9d3b26912f940b2a34d.png)  
![a0c0b07552284c4233b408910b728152.png](../_resources/a0c0b07552284c4233b408910b728152.png)  
![3e162b5ec5736c7738d857581cc3491a.png](../_resources/3e162b5ec5736c7738d857581cc3491a.png)  
![1b099380df0734c84dfe88045077f163.png](../_resources/1b099380df0734c84dfe88045077f163.png)  
![7d32489326255eb63564de5b71053e65.png](../_resources/7d32489326255eb63564de5b71053e65.png)  
![8d39fc65c736caca9864de517377e754.png](../_resources/8d39fc65c736caca9864de517377e754.png)  
2\. Build  
![f3ad1a58d48f6d1211bdc0a0d5b39bb8.png](../_resources/f3ad1a58d48f6d1211bdc0a0d5b39bb8.png)  
![9d8edbdd87293593413edfbee6a103b2.png](../_resources/9d8edbdd87293593413edfbee6a103b2.png)

* * *

**Flash**

1.  Add runner configuration(OpenOCD). Configure the project runner to use OpenOCD  
    ![a315cc78c49a52b06bd300c7f5226c78.png](../_resources/a315cc78c49a52b06bd300c7f5226c78.png)  
    ![11db14703600300a5b69b49ea15f6793.png](../_resources/11db14703600300a5b69b49ea15f6793.png)  
    ![0a36f9c82afd2eef16cf17b10404cf7f.png](../_resources/0a36f9c82afd2eef16cf17b10404cf7f.png)  
    ![790b04cca5af341ad74a55dd744655f7.png](../_resources/790b04cca5af341ad74a55dd744655f7.png)
2.  Update runner settings  
    Go to `View -> Command Palette` and run `Update Zephyr Project Runner`. Select the project to update the runner and refresh the settings  
    ![53d0263fa45919d7373089a59f29b638.png](../_resources/53d0263fa45919d7373089a59f29b638.png)  
    ![4968564fd7d45688760a58057cf861a4.png](../_resources/4968564fd7d45688760a58057cf861a4.png)  
    ![a9124e60394720787e70891ea0142cd1.png](../_resources/a9124e60394720787e70891ea0142cd1.png)
3.  Set the target type from CMSIS extension. Please choose the target type.  
    ![8ef515fd59d43e59b671e1dc0318e6e2.png](../_resources/8ef515fd59d43e59b671e1dc0318e6e2.png)
4.  Build project and Flash  
    ![7326ef4b4f3332a2ba9dc78af0c5768d.png](../_resources/7326ef4b4f3332a2ba9dc78af0c5768d.png)  
    ![923c3dce2d6b50fd73de050b5cdb4169.png](../_resources/923c3dce2d6b50fd73de050b5cdb4169.png)  
    ![c85ce21986ce7fb5de6fd83c43f46dd7.png](../_resources/c85ce21986ce7fb5de6fd83c43f46dd7.png)

* * *

**Debug**

1.  Create launch.json for debugging  
    ![97572db9704c381e657e661db6041c61.png](../_resources/97572db9704c381e657e661db6041c61.png)
2.  Setup the runToEntryPoint(optional). You can specify the symbol function where your want to stop (for example, Zephyr's execution entry point is z_arm_reset). Default is main().  
    ![6ac6acac5c8b6c8c6398126eba86a247.png](../_resources/6ac6acac5c8b6c8c6398126eba86a247.png)
3.  Select Debug Setting. Select `Nuvoton Debug Zephyr`  
    ![010063e98698b07eae282dbee3ea3919.png](../_resources/010063e98698b07eae282dbee3ea3919.png)
4.  Enter Debug Mode and Start Monitior for output

* * *

**Others**

1.  Zephyr-ide.json location  
    \$Zephyr_Workspace/.vscode/zephyr-ide.json

* * *

**Reference**

1.  Nuvoton NuMicro Cortex-M pack  
    ![30adcf321ed634b77a1b689114e13b02.png](../_resources/30adcf321ed634b77a1b689114e13b02.png)