# libfreenect2

## Description
Driver for Kinect for Windows v2 (K4W2) devices (release and developer preview).

Note: libfreenect2 does not do anything for either Kinect for Windows v1 or Kinect for Xbox 360 sensors. Use libfreenect1 for those sensors.

This driver supports:
* RGB image transfer
* IR and depth image transfer
* registration of RGB and depth images

Missing features:
* audio transfer
* firmware updates

Watch the OpenKinect wiki at www.openkinect.org and the mailing list at https://groups.google.com/forum/#!forum/openkinect for the latest developments and more information about the K4W2 USB protocol.

## FAQ

### Can I use the Kinect v2 without an USB3 port?

No. It's a pure USB3 device due to the high bandwidth requirements.

### Protonect complains about "no device connected" or "failure opening device".

Either your device is connected to an USB2-only port (see above), or you don't have permissions to access the device. On Linux, try running Protonect as root (e.g. using `sudo`). If that fixes things, place `rules/90-kinect2.rules` into `/etc/udev/rules.d/` and re-plug the device.

On Linux, also check `dmesg`. If there are warnings like `usb 4-1.1: Not enough bandwidth for new device state.` it means the hardware does not have the capacity for USB3 even if it claims so, or its capacity is not well supported.

On Mac OS X, open "System Information" from Spotlight, go to the USB section, and verify "Xbox NUI Sensor" is under "USB 3.0 SuperSpeed Bus" not "High-Speed Bus". If this is not the case, try unplugging the Kinect from power source with the USB cable connected, and plug the power again, then verify.

### I'm getting lots of USB transfer errors, and/or only blank windows.

USB3 as a whole is a flaky thing. If you're running Linux, try upgrading to a recent kernel (>= 3.16) first. If that doesn't work, try a different USB3 controller. The following ones are known to work on a 3.16 kernel:
* Intel Corporation 8 Series/C220 Series Chipset Family USB xHCI
* Intel Corporation 7 Series/C210 Series Chipset Family USB xHCI
* NEC Corporation uPD720200 USB 3.0 Host Controller

Probably not working:
* ASMedia Technology Inc. Device 1142
* ASMedia Technology Inc. ASM1042

Messages in `dmesg` like this means bugs in the USB driver. Updating kernel might help.
```
[  509.238571] xhci_hcd 0000:03:00.0: xHCI host not responding to stop endpoint command.
[  509.238580] xhci_hcd 0000:03:00.0: Assuming host is dying, halting host.
```

Finally, it's also possible that your executable is not actually using the patched libusb from the `depends/` folder which is required at the moment. Check this using `ldd ./Protonect | grep libusb` (shows `libusb-1.0` under `depends/`), and adjust your `LD_LIBRARY_PATH` if necessary.

### I'm seeing the color camera stream, but no depth/IR (black windows).

The depth packet processor runs on OpenGL by default. You can try alternatives, such as OpenCL (by running `Protonect cl`) or CPU (`Protonect cpu`). At least the CPU DPP should always produce some output, although slow.

For OpenCL on Intel/Linux, you can also try to set `/sys/module/i915/parameters/enable_cmd_parser` to 0.

### Can I use multiple Kinect v2 devices on one machine?

Yes - in fact, this has been reported to work for up to 5 devices on a high-end PC using multiple separate PCI Express USB3 expansion cards (with NEC controller chip). 

If you're using Linux, you may have to increase USBFS memory buffers by appending `usbcore.usbfs_memory_mb=64` to your kernel commandline. Depending on the number of Kinects, you may need to use an even larger buffer size.

If you're using an expansion card, make sure it's not plugged into an PCI-E x1 slot. A single lane doesn't have enough bandwidth. x8 or x16 slots usually work.

## Maintainers:

* Joshua Blake <joshblake@gmail.com>
* Florian Echtler
* Christian Kerl

## Installation

This project uses the libusbx drivers and API. Setting things up varies by platform.

### Windows / Visual Studio

#### libusbK driver

If you have the Kinect for Windows v2 SDK, install it first. You don't need to uninstall the SDK or the driver before doing this procedure.

Install the libusbK backend driver for libusbx:

1. Download Zadig from http://zadig.akeo.ie/.
2. Run Zadig and in options, check List All Devices and uncheck Ignore Hubs or Composite Parents
3. Select the Xbox NUI Sensor (composite parent) from the drop-down box. (Ignore the Interface 0 and Interface 2 varieties.) The current driver will list usbccgp. USB ID is VID 045E, PID 02C4.
4. Select libusbK (v3.0.6.0) from the replacement driver list.
5. Click the Replace Driver button. Click yes on the warning about replacing a system driver. (This is because it is a composite parent.)
6. Done. 

To uninstall the libusbK driver (and get back the official SDK driver, if installed):

1. Open Device Manager
2. Under libusbK USB Devices, right click the "Xbox NUI Sensor (Composite Parent)" device and select uninstall.
3. Important: Check the "Delete the driver software for this device." checkbox, then click OK.

If you already had the official SDK driver installed and you want to use it:

4. In Device Manager, in the Action menu, click "Scan for hardware changes."

This will enumerate the Kinect sensor again and it will pick up the K4W2 SDK driver, and you should be ready to run KinectService.exe again immediately.

You can go back and forth between the SDK driver and the libusbK driver very quickly and easily with these steps.

#### libusb

* Build from source (recommended)
```bash
cd depends/
git clone https://github.com/libusb/libusb.git
cd libusb
git remote add joshblake https://github.com/JoshBlake/libusbx.git
git fetch joshblake
git merge joshblake/winiso  # patches for libusbK backend
```
Open `libusb/msvc/libusb_2013.sln` with Visual Studio 2013 (or older version, accordingly), set configurations to "Release x64", and build "libusb-1.0 (dll)". You can clone the libusb repo to somewhere else, but you will need to set environment variable `LibUSB_ROOT` to that path. Building with "Win32" is not recommended as it results in lower performance.

* Pre-built binary

Joshua Blake provided a Debug version binary: https://www.dropbox.com/s/madoye1ayaoajet/libusbx-winiso.zip. Install it as `depends/libusbx`. This version was built in 2013.

#### TurboJPEG

* Download from http://sourceforge.net/projects/libjpeg-turbo/files
* Extract it to the default path (`c:\libjpeg-turbo64`), or as `depends/libjpeg-turbo64`, or anywhere as long as the environment variable `TurboJPEG_ROOT` is set to installed path.

#### GLFW

* Download 64-bit Windows binaries from http://www.glfw.org/download.html
* Extract it as `depends/glfw` (rename `glfw-3.x.x.bin.WIN64` to glfw), or anywhere as long as the environment variable `GLFW_ROOT` is set to the installed path.

#### OpenCL

* Intel GPU: Download `intel_sdk_for_ocl_applications_2014_x64_setup.msi` from http://www.softpedia.com/get/Programming/SDK-DDK/Intel-SDK-for-OpenCL-Applications.shtml (SDK official download is replaced by $$$ and no longer available) and install it. Then verify `INTELOCLSDKROOT` is set as an environment variable.

#### Build

```
mkdir build && cd build
cmake .. -G "Visual Studio 12 2013 Win64" -DCMAKE_INSTALL_PREFIX=.
cmake --build . --config Release --target install
```

Then you can run the program with `.\bin\Protonect.exe`. If DLLs are missing, you can copy them to the `bin` folder.

### Mac OSX

Use your favorite package managers (brew, ports, etc.)

1. ``cd`` into a directory where you want to keep libfreenect2 stuff in
1. Make sure these build tools are available: wget, git, cmake, pkg-config, automake, autoconf, libtool. Xcode may provide some of them. Install the rest via package managers.
1. Install dependencies: TurboJPEG, GLFW.

    ```
brew update
brew tap homebrew/science
brew install jpeg-turbo
brew tap homebrew/versions
brew install glfw3
```

    Do not install libusb via package managers for libfreenect2. libfreenect2 includes an unreleased local version of libusb with USB3 specific patches. libfreenect2's libusb should still work fine in presence of a global version libusb.

    It is not recommended to build TurboJPEG from source, which produces corrupted results on Mac OSX according to reports. Install TurboJPEG binary only from package managers.

1. Download the libfreenect2 repository

    ```
git clone git@github.com:OpenKinect/libfreenect2.git
```

1. Install a bunch of dependencies

    ```
cd ./libfreenect2
sh ./depends/install_mac.sh
```

1. Build the actual protonect executable

    ```
mkdir build && cd build
cmake ..
make
make install
```

1. Run the program

    ```
./bin/Protonect
```

### Debian/Ubuntu 14.04 (perhaps earlier)

1. Install libfreenect2

    ```
git clone https://github.com/OpenKinect/libfreenect2.git
```

1. Install a bunch of dependencies

    ```bash
sudo apt-get install build-essential libturbojpeg libjpeg-turbo8-dev libtool autoconf libudev-dev cmake mesa-common-dev freeglut3-dev libxrandr-dev doxygen libxi-dev automake
# sudo apt-get install libturbojpeg0-dev (Debian)

cd libfreenect2/depends
sh install_ubuntu.sh
sudo dpkg -i libglfw3*_3.0.4-1_*.deb  # Ubuntu 14.04 only
# sudo apt-get install libglfw3-dev (Debian/Ubuntu 14.10+:)
```

1. OpenCL dependency
  * AMD GPU: Install the latest version of the AMD Catalyst drivers from https://support.amd.com and `apt-get install opencl-headers`.
  * Nvidia GPU: Install the latest version of the Nvidia drivers, for example nvidia-346 from `ppa:xorg-edgers` and `apt-get install opencl-headers`.
  * Intel GPU (kernel 3.16+ recommended): Install beignet-dev 1.0+, `apt-get install beignet-dev`. If not available, use this ppa `sudo apt-add-repository ppa:pmjdebruijn/beignet-testing`.

1. Build the actual protonect executable

    ```
mkdir build && cd build
cmake ..
make
sudo make install
```

1. Run the program

    ```
./bin/Protonect
```

### Other operating systems

I'm not sure, but look for libusbx installation instructions for your OS. Figure out how to attach the driver to the Xbox NUI Sensor composite parent device, VID 045E PID 02C4, then contribute your procedure.

## Required notification

The K4W2 hardware is currently pre-release. Per the K4W2 developer program agreement, all public demonstrations and code should display this notice:

    "This is preliminary software and/or hardware and APIs are preliminary and subject to change."
