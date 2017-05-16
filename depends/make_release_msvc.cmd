@echo off
set ver=%1
set vs_year=%2
if "%vs_year%"=="2015" (
  set vs_ver=14
) else (
  if "%vs_year%"=="2013" (
    set vs_ver=12
  ) else (
    echo Unsupported MSVC version. Usage: %~nx0 version vs_year >&2
    exit /b
  )
)
@echo on

pushd .
call install_libusb_vs%vs_year%.cmd
popd

rmdir /s /q build
mkdir build
cd build
cmake ..\.. -G "Visual Studio %vs_ver% %vs_year% Win64" -DENABLE_OPENCL=OFF -DENABLE_CUDA=OFF
cmake --build . --config Release --target install

rmdir /s /q install\lib\cmake
rmdir /s /q install\lib\pkgconfig
copy ..\LICENSES.txt install
copy ..\INSTALL-windows.txt install
copy ..\..\CONTRIB install
cd ..
rmdir /s /q libfreenect2-%ver%-vs%vs_year%-x64
move build\install libfreenect2-%ver%-vs%vs_year%-x64
rmdir /s /q build
