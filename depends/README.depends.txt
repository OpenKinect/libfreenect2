For now to keep binaries out of git, download dependencies separately:

Dependencies:
---

= Windows

CMake
=====
1. Download cmake-3.2.2-win32-x86.exe (or newer) from http://www.cmake.org/download/
2. Install it to system

OpenCV
======
1. Download opencv-2.4.11.exe (or newer) from http://sourceforge.net/projects/opencvlibrary/files/opencv-win/
2. Extract it to `depends/opencv`

TurboJPEG
=========
1. Download libjpeg-turbo-1.4.0-vc64.exe (or newer) from http://sourceforge.net/projects/libjpeg-turbo/files/
2. Extract it as `depends/libjpeg_turbo`

Intel OpenCL SDK (for Intel HD Graphics)
========================================
1. Download intel_sdk_for_ocl_applications_2014_x64_setup.msi from http://www.softpedia.com/get/Programming/SDK-DDK/Intel-SDK-for-OpenCL-Applications.shtml
2. Install it to system
3. Verify `INTELOCLSDKROOT` is a environment variable

libusbx
=======
1. Download libusbx-winiso.zip from https://github.com/JoshBlake/libusbx/archive/winiso.zip
2. Extract it to `depends/libusb_src`
3. Open `depends/libusb_src/libusbx_2012.sln` with Visual Studio 2012 or newer
4. Build "libusb-1.0 (dll)"; you may set the profile to "Release x64"
5. Copy `depends/libusb_src/x64/Release/dll/*` to `depends/libusb/lib/`
6. Copy `depends/libusb_src/libusb/libusb.h` to `depends/libusb/include/libusb-1.0/libusb.h`

Alternatively
1. Download pre-built binaries libusbx-winiso.zip from https://www.dropbox.com/s/madoye1ayaoaj
et/libusbx-winiso.zip
2. Extract the content to places as above.

GLFW
====
1. Download glfw-3.0.4.zip from https://github.com/glfw/glfw/archive/3.0.4.zip
2. Extract it to `depends/glfw_src`
3. `cd depends/glfw_src; mkdir build && cd build`
4. `cmake .. -G "Visual Studio 12 2013 Win64" -DCMAKE_INSTALL_PREFIX=%cd%\..\..\glfw -DBUILD_SHARED_LIBS=TRUE`
5. `cmake --build . --config Release --target install`

Protonect
=========
1. `cd example/protonect`
2. `mkdir build && cd build`
3. `cmake .. -G "Visual Studio 12 2013 Win64" -DOpenCV_DIR=%cd%\..\..\..\depends\opencv\build -DCMAKE_INSTALL_PREFIX=%cd%\..\..\..\install`
4. `cmake --build . --config Release --target install`


````
cd ..\..\..\install
set PATH=%PATH%;%cd%\lib;%cd%\..\depends\libusb\lib;%cd%\..\depends\opencv\build\x64\vc12\bin
.\bin\Protonect.exe
````

= Ubuntu
kernel 3.10+
libusbx 1.0.17 
  with superspeed patch by Joshua Blake <joshblake@gmail.com> see https://github.com/JoshBlake/libusbx/tree/superspeed
  with MAX_ISO_BUFFER_LENGTH increased to 49152 in libusb/os/libusb_usbfs.h
turbojpeg
opencv

run ./depends/install_ubuntu.sh

= Mac OSX
Same as Ubuntu

run ./depends/install_mac.sh

= Non-Windows
libusbx 1.0.17 or later
http://libusbx.org/

Not tested.
Please contribute if you get things setup.
