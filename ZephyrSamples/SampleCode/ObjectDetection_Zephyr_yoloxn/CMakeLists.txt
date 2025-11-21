# Copyright 2021-2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
set(APP_DEVICE_DIR src/Device)
set(APP_MODEL_DIR src/Model)
set(APP_SRC_DIR src/)
set(APP_NPU_DIR src/NPU)
set(APP_PROFILER_DIR src/ProfilerCounter)
set(APP_THIRDPARTY_MLEVK_DIR ../../ThirdParty/ml-embedded-evaluation-kit)
set(APP_THIRDPARTY_OMV ../../ThirdParty/openmv)

project(ObjectDetection-Zephyr-yoloxn)

target_include_directories(app PRIVATE
    ${APP_DEVICE_DIR}/include
)

target_include_directories(app PRIVATE
    ${APP_PROFILER_DIR}/include
)

target_include_directories(app PRIVATE
    ${APP_NPU_DIR}/include
)

target_include_directories(app PRIVATE
    ${APP_MODEL_DIR}/include
)

target_include_directories(app PRIVATE
    ${APP_THIRDPARTY_MLEVK_DIR}/source/application/api/common/include
    ${APP_THIRDPARTY_MLEVK_DIR}/source/log/include
    ${APP_THIRDPARTY_MLEVK_DIR}/source/profiler/include
    ${APP_THIRDPARTY_MLEVK_DIR}/source/math/include
)

target_include_directories(app PRIVATE
    ${APP_THIRDPARTY_OMV}/omv/imlib
    ${APP_THIRDPARTY_OMV}/omv/alloc
    ${APP_THIRDPARTY_OMV}/omv/common
    ${APP_THIRDPARTY_OMV}/omv/Lib
)

target_compile_definitions(app PRIVATE
    FS_FATFS_WINDOW_ALIGNMENT=32
    NVT_DCACHE_ON=1
    ARM_NPU
)

if(CONFIG_APP_OD_PROFILE_ENABLED)
    target_compile_definitions(app PRIVATE __PROFILE__)
endif()

if(CONFIG_APP_OD_UVC_SHOW_IMAGE)
    target_compile_definitions(app PRIVATE __USE_UVC__)
endif()

if(CONFIG_APP_OD_MODEL_FROM_SD)
    target_compile_definitions(app PRIVATE __LOAD_MODEL_FROM_SD__)
endif()

target_sources(app PRIVATE
    ${APP_SRC_DIR}/main.cpp
    ${APP_SRC_DIR}/BoardInit.cpp
    ${APP_SRC_DIR}/ModelFileReader.c
    ${APP_SRC_DIR}/DetectorPostProcessing.cpp
    ${APP_SRC_DIR}/InferenceTask.cpp
)

target_sources(app PRIVATE
    ${APP_DEVICE_DIR}/HyperRAM/hyperram_code.c
    ${APP_DEVICE_DIR}/SDCard/sdglue.c
    ${APP_DEVICE_DIR}/SDCard/diskio_SDH.c
    ${APP_DEVICE_DIR}/ImageSensor/ImageSensor.c
    ${APP_DEVICE_DIR}/ImageSensor/Sensor/Sensor_HM1055.c
    ${APP_DEVICE_DIR}/ImageSensor/Sensor/SWI2C.c
    ${APP_DEVICE_DIR}/UVC/UVC.c
    ${APP_DEVICE_DIR}/UVC/descriptors.c
)

target_sources(app PRIVATE
    ${APP_PROFILER_DIR}/pmu_counter.c
)

target_sources(app PRIVATE
    ${APP_NPU_DIR}/ethosu_profiler.c
)

target_sources(app PRIVATE
    ${APP_MODEL_DIR}/YoloXnanoNu.cpp
    ${APP_MODEL_DIR}/yolox_nano_nu_hg_full_integer_quant_vela.tflite.cpp
    ${APP_MODEL_DIR}/Labels.cpp
)

target_sources(app PRIVATE
    ${APP_THIRDPARTY_MLEVK_DIR}/source/application/api/common/source/Model.cpp
    ${APP_THIRDPARTY_MLEVK_DIR}/source/application/api/common/source/ImageUtils.cpp
    ${APP_THIRDPARTY_MLEVK_DIR}/source/application/api/common/source/TensorFlowLiteMicro.cpp
    ${APP_THIRDPARTY_MLEVK_DIR}/source/math/PlatformMath.cpp
    ${APP_THIRDPARTY_MLEVK_DIR}/source/profiler/Profiler.cpp
)

target_sources(app PRIVATE
    ${ZEPHYR_NUVOTON_MODULE_DIR}/m55m1x/StdDriver/src/spim_hyper.c
    ${ZEPHYR_NUVOTON_MODULE_DIR}/m55m1x/StdDriver/src/sdh.c
    ${ZEPHYR_NUVOTON_MODULE_DIR}/m55m1x/StdDriver/src/ccap.c
    ${ZEPHYR_NUVOTON_MODULE_DIR}/m55m1x/StdDriver/src/hsusbd.c
)

target_sources(app PRIVATE
    ${ZEPHYR_FATFS_MODULE_DIR}/ff.c
    ${ZEPHYR_FATFS_MODULE_DIR}/option/ffunicode.c
)

target_link_libraries(app PRIVATE 
    ${PROJECT_SOURCE_DIR}/${APP_THIRDPARTY_OMV}/omv/Lib/libomv.a
)

# Add platform specific linker source snippet
zephyr_linker_sources(SECTIONS sram_hyperram_section.ld)
zephyr_linker_sources(SECTIONS sram2_section.ld)
