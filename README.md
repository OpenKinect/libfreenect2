# libfreenect2

## Table of Contents

* [Description](README.md#description)
* [Requirements](README.md#requirements)
* [Troubleshooting](README.md#troubleshooting-and-reporting-bugs)
* [Maintainers](README.md#maintainers)
* [Installation](README.md#installation)
  * [Windows / Visual Studio](README.md#windows--visual-studio)
  * [MacOS](README.md#macos)
  * [Linux](README.md#linux)
* [API Documentation (external)](https://openkinect.github.io/libfreenect2/)

## Description

Driver for Kinect for Windows v2 (K4W2) devices (release and developer preview).

Note: libfreenect2 does not do anything for either Kinect for Windows v1 or Kinect for Xbox 360 sensors. Use libfreenect1 for those sensors.

If you are using libfreenect2 in an academic context, please cite our work using the following DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5167182.svg)](https://doi.org/10.5281/zenodo.5167182)



If you use the KDE depth unwrapping algorithm implemented in the library, please also cite this ECCV 2016 [paper](http://users.isy.liu.se/cvl/perfo/abstracts/jaremo16.html).

This driver supports:
* RGB image transfer
* IR and depth image transfer
* registration of RGB and depth images

Missing features:
* firmware updates (see [issue #460](https://github.com/OpenKinect/libfreenect2/issues/460) for WiP)

Watch the OpenKinect wiki at www.openkinect.org and the mailing list at https://groups.google.com/forum/#!forum/openkinect for the latest developments and more information about the K4W2 USB protocol.

The API reference documentation is provided here https://openkinect.github.io/libfreenect2/.

## Requirements

### Hardware requirements

* USB 3.0 controller. USB 2 is not supported.

Intel and NEC USB 3.0 host controllers are known to work. ASMedia controllers are known to not work.

Virtual machines likely do not work, because USB 3.0 isochronous transfer is quite delicate.

##### Requirements for multiple Kinects

It has been reported to work for up to 5 devices on a high-end PC using multiple separate PCI Express USB3 expansion cards (with NEC controller chip). If you're using Linux, you may have to [increase USBFS memory buffers](https://github.com/OpenKinect/libfreenect2/wiki/Troubleshooting#multiple-kinects-try-increasing-usbfs-buffer-size). Depending on the number of Kinects, you may need to use an even larger buffer size. If you're using an expansion card, make sure it's not plugged into an PCI-E x1 slot. A single lane doesn't have enough bandwidth. x8 or x16 slots usually work.

### Operating system requirements

* Windows 7 (buggy), Windows 8, Windows 8.1, and probably Windows 10
* Debian, Ubuntu 14.04 or newer, probably other Linux distros. Recommend kernel 3.16+ or as new as possible.
* Mac OS X

### Requirements for optional features

* OpenGL depth processing: OpenGL 3.1 (Windows, Linux, Mac OS X). OpenGL ES is not supported at the moment.
* OpenCL depth processing: OpenCL 1.1
* CUDA depth processing: CUDA (6.5 and 7.5 are tested; The minimum version is not clear.)
* VAAPI JPEG decoding: Intel (minimum Ivy Bridge or newer) and Linux only
* VideoToolbox JPEG decoding: Mac OS X only
* OpenNI2 integration: OpenNI2 2.2.0.33
* Jetson TK1: Linux4Tegra 21.3 or later. Check [Jetson TK1 issues](https://github.com/OpenKinect/libfreenect2/wiki/Troubleshooting#jetson-tk1-issues) before installation. Jetson TX1 is not yet supported as the developers don't have one, but it may be easy to add the support.

## Troubleshooting and reporting bugs

First, check https://github.com/OpenKinect/libfreenect2/wiki/Troubleshooting for known issues.

When you report USB issues, please attach relevant debug log from running the program with environment variable `LIBUSB_DEBUG=3`, and relevant log from `dmesg`. Also include relevant hardware information `lspci` and `lsusb -t`.

## Maintainers

* Joshua Blake <joshblake@gmail.com>
* Florian Echtler
* Christian Kerl
* Lingzhu Xiang (development/master branch)

## Installation

### Windows / Visual Studio

* Install UsbDk driver

    1. (Windows 7) You must first install Microsoft Security Advisory 3033929 otherwise your USB keyboards and mice will stop working!
    2. Download the latest x64 installer from https://github.com/daynix/UsbDk/releases, install it.
    3. If UsbDk somehow does not work, uninstall UsbDk and follow the libusbK instructions.

    This doesn't interfere with the Microsoft SDK. Do not install both UsbDK and libusbK drivers
* (Alternatively) Install libusbK driver

    You don't need the Kinect for Windows v2 SDK to build and install libfreenect2, though it doesn't hurt to have it too. You don't need to uninstall the SDK or the driver before doing this procedure.

    Install the libusbK backend driver for libusb. Please follow the steps exactly:

    1. Download Zadig from http://zadig.akeo.ie/.
    2. Run Zadig and in options, check "List All Devices" and uncheck "Ignore Hubs or Composite Parents"
    3. Select the "Xbox NUI Sensor (composite parent)" from the drop-down box. (Important: Ignore the "NuiSensor Adaptor" varieties, which are the adapter, NOT the Kinect) The current driver will list usbccgp. USB ID is VID 045E, PID 02C4 or 02D8.
    4. Select libusbK (v3.0.7.0 or newer) from the replacement driver list.
    5. Click the "Replace Driver" button. Click yes on the warning about replacing a system driver. (This is because it is a composite parent.)

    To uninstall the libusbK driver (and get back the official SDK driver, if installed):

    1. Open "Device Manager"
    2. Under "libusbK USB Devices" tree, right click the "Xbox NUI Sensor (Composite Parent)" device and select uninstall.
    3. Important: Check the "Delete the driver software for this device." checkbox, then click OK.

    If you already had the official SDK driver installed and you want to use it:

    4. In Device Manager, in the Action menu, click "Scan for hardware changes."

    This will enumerate the Kinect sensor again and it will pick up the K4W2 SDK driver, and you should be ready to run KinectService.exe again immediately.

    You can go back and forth between the SDK driver and the libusbK driver very quickly and easily with these steps.

* Install libusb

    Download the latest build (.7z file) from https://github.com/libusb/libusb/releases, and extract as `depends/libusb` (rename folder `libusb-1.x.y` to `libusb` if any).
* Install TurboJPEG

    Download the `-vc64.exe` installer from http://sourceforge.net/projects/libjpeg-turbo/files, extract it to `c:\libjpeg-turbo64` (the installer's default) or `depends/libjpeg-turbo64`, or anywhere as specified by the environment variable `TurboJPEG_ROOT`.
* Install GLFW

    Download from http://www.glfw.org/download.html (64-bit), extract as `depends/glfw` (rename `glfw-3.x.x.bin.WIN64` to `glfw`), or anywhere as specified by the environment variable `GLFW_ROOT`.
* Install OpenCL (optional)
    1. Intel GPU: Download "Intel® SDK for OpenCL™ Applications 2016" from https://software.intel.com/en-us/intel-opencl (requires free registration) and install it.
* Install CUDA (optional, Nvidia only)
    1. Download CUDA Toolkit and install it. You MUST install the samples too.
* Install OpenNI2 (optional)

    Download OpenNI 2.2.0.33 (x64) from http://structure.io/openni, install it to default locations (`C:\Program Files...`).
* Build

    The default installation path is `install`, you may change it by editing `CMAKE_INSTALL_PREFIX`.
    ```
    mkdir build && cd build
    cmake .. -G "Visual Studio 12 2013 Win64"
    cmake --build . --config RelWithDebInfo --target install
    ```
    Or `-G "Visual Studio 14 2015 Win64"`.
    Or `-G "Visual Studio 16 2019"`.
* Run the test program: `.\install\bin\Protonect.exe`, or start debugging in Visual Studio.
* Test OpenNI2 (optional)

    Copy freenect2-openni2.dll, and other dll files (libusb-1.0.dll, glfw.dll, etc.) in `install\bin` to `C:\Program Files\OpenNI2\Tools\OpenNI2\Drivers`. Then run `C:\Program Files\OpenNI\Tools\NiViewer.exe`. Environment variable `LIBFREENECT2_PIPELINE` can be set to `cl`, `cuda`, etc to specify the pipeline.

### Windows / vcpkg

You can download and install libfreenect2 using the [vcpkg](https://github.com/Microsoft/vcpkg) dependency manager:
```
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./vcpkg integrate install
vcpkg install libfreenect2
```
The libfreenect2 port in vcpkg is kept up to date by Microsoft team members and community contributors. If the version is out of date, please [create an issue or pull request](https://github.com/Microsoft/vcpkg) on the vcpkg repository.

### MacOS

Use your favorite package managers (brew, ports, etc.) to install most if not all dependencies:

* Make sure these build tools are available: wget, git, cmake, pkg-config. Xcode may provide some of them. Install the rest via package managers.
* Download libfreenect2 source
    ```
    git clone https://github.com/OpenKinect/libfreenect2.git
    cd libfreenect2
    ```
* Install dependencies: libusb, GLFW
    ```
    brew update
    brew install libusb
    brew install glfw3
    ```
* Install TurboJPEG (optional)
    ```
    brew install jpeg-turbo
    ```
* Install CUDA (optional): TODO
* Install OpenNI2 (optional)
    ```
    brew tap brewsci/science
    brew install openni2
    export OPENNI2_REDIST=/usr/local/lib/ni2
    export OPENNI2_INCLUDE=/usr/local/include/ni2
    ```
* Build
    ```
    mkdir build && cd build
    cmake ..
    make
    make install
    ```
* Run the test program: `./bin/Protonect`
* Test OpenNI2. `make install-openni2` (may need sudo), then run `NiViewer`. Environment variable `LIBFREENECT2_PIPELINE` can be set to `cl`, `cuda`, etc to specify the pipeline.

### Linux

Note: Ubuntu 12.04 is too old to support. Debian jessie may also be too old, and Debian stretch is implied in the following.

* Download libfreenect2 source
    ```
    git clone https://github.com/OpenKinect/libfreenect2.git
    cd libfreenect2
    ```
* (Ubuntu 14.04 only) Download upgrade deb files
    ```
    cd depends; ./download_debs_trusty.sh
    ```
* Install build tools
    ```
    sudo apt-get install build-essential cmake pkg-config
    ```
* Install libusb. The version must be >= 1.0.20.
    1. (Ubuntu 14.04 only) `sudo dpkg -i debs/libusb*deb`
    2. (Other) `sudo apt-get install libusb-1.0-0-dev`
* Install TurboJPEG
    1. (Ubuntu 14.04 to 16.04) `sudo apt-get install libturbojpeg libjpeg-turbo8-dev`
    2. (Debian/Ubuntu 17.10 and newer) `sudo apt-get install libturbojpeg0-dev`
* Install OpenGL
    1. (Ubuntu 14.04 only) `sudo dpkg -i debs/libglfw3*deb; sudo apt-get install -f`
    2. (Odroid XU4) OpenGL 3.1 is not supported on this platform. Use `cmake -DENABLE_OPENGL=OFF` later.
    3. (Other) `sudo apt-get install libglfw3-dev`
* Install OpenCL (optional)
    - Intel GPU
        1. (Ubuntu 14.04 only) `sudo apt-add-repository ppa:floe/beignet; sudo apt-get update; sudo apt-get install beignet-dev; sudo dpkg -i debs/ocl-icd*deb`
        2. (Other) `sudo apt-get install beignet-dev`
        3. For older kernels, `# echo 0 >/sys/module/i915/parameters/enable_cmd_parser` is needed. See more known issues at https://www.freedesktop.org/wiki/Software/Beignet/.
    - AMD GPU: Install the latest version of the AMD Catalyst drivers from https://support.amd.com and `apt-get install opencl-headers`.
    - Mali GPU (e.g. Odroid XU4): (with root) `mkdir -p /etc/OpenCL/vendors; echo /usr/lib/arm-linux-gnueabihf/mali-egl/libmali.so >/etc/OpenCL/vendors/mali.icd; apt-get install opencl-headers`.
    - Verify: You can install `clinfo` to verify if you have correctly set up the OpenCL stack.
* Install CUDA (optional, Nvidia only):
    - (Ubuntu 14.04 only) Download `cuda-repo-ubuntu1404...*.deb` ("deb (network)") from Nvidia website, follow their installation instructions, including `apt-get install cuda` which installs Nvidia graphics driver.
    - (Jetson TK1) It is preloaded.
    - (Nvidia/Intel dual GPUs) After `apt-get install cuda`, use `sudo prime-select intel` to use Intel GPU for desktop.
    - (Other) Follow Nvidia website's instructions. You must install the samples package.
* Install VAAPI (optional, Intel only)
    1. (Ubuntu 14.04 only) `sudo dpkg -i debs/{libva,i965}*deb; sudo apt-get install -f`
    2. (Other) `sudo apt-get install libva-dev libjpeg-dev`
    3. Linux kernels 4.1 to 4.3 have performance regression. Use 4.0 and earlier or 4.4 and later (Though Ubuntu kernel 4.2.0-28.33~14.04.1 has backported the fix).
* Install OpenNI2 (optional)
    1. (Ubuntu 14.04 only) `sudo apt-add-repository ppa:deb-rob/ros-trusty && sudo apt-get update` (You don't need this if you have ROS repos), then `sudo apt-get install libopenni2-dev`
    2. (Other) `sudo apt-get install libopenni2-dev`
* Build (if you have run `cd depends` previously, `cd ..` back to the libfreenect2 root directory first.)
    ```
    mkdir build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2
    make
    make install
    ```
    You need to specify `cmake -Dfreenect2_DIR=$HOME/freenect2/lib/cmake/freenect2` for CMake based third-party application to find libfreenect2.
* Set up udev rules for device access: `sudo cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/`, then replug the Kinect.
* Run the test program: `./bin/Protonect`
* Run OpenNI2 test (optional): `sudo apt-get install openni2-utils && sudo make install-openni2 && NiViewer2`. Environment variable `LIBFREENECT2_PIPELINE` can be set to `cl`, `cuda`, etc to specify the pipeline.
