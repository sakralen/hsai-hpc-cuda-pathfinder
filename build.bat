@echo off
setlocal

set CUDA_ARCH=sm_50

set SRC_DIR=.\src
set BUILD_DIR=.\build

if not exist %BUILD_DIR% mkdir %BUILD_DIR%

nvcc -arch=%CUDA_ARCH% -I .\inc -o %BUILD_DIR%\cuda_test.exe %SRC_DIR%\main.cu %SRC_DIR%\utility.cu %SRC_DIR%\fieldgenerator.cu %SRC_DIR%\lee.cu

endlocal
::@pause