== libfreenect2

Maintainers:
* Joshua Blake <joshblake@gmail.com>
* Florian Echtler
* Christian Kerl

=== Description
Driver for Kinect for Windows v2 (K4W2) devices (release and developer preview).

Note: libfreenect2 does not do anything for either Kinect for Windows v1 or Kinect for Xbox 360 sensors. Use libfreenect1 for those sensors.

This driver supports:
* RGB image transfer
* IR and depth image transfer

Missing features:
* registration of RGB and depth images
* audio transfer
* firmware updates

Watch the OpenKinect wiki at www.openkinect.org and the mailing list at https://groups.google.com/forum/#!forum/openkinect for the latest developments and more information about the K4W2 USB protocol.

=== Installation

This project uses the libusbx drivers and API. Setting things up varies by platform.

==== Windows

If you have the Kinect for Windows v2 SDK, install it first. You don't need to uninstall the SDK or the driver before doing this procedure.

Install the libusbK backend driver for libusbx:

1. Download Zadig from http://zadig.akeo.ie/.
2. Run Zadig and in options, check List All Devices and uncheck Ignore Hubs or Composite Parents
3. Select the Xbox NUI Sensor (composite parent) from the drop-down box. (Ignore the Interface 0 and Interface 2 varieties.)
The current driver will list usbccgp. USB ID is VID 045E, PID 02C4.
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

==== Mac OSX

1. cd into a directory where you want to keep libfreenect2 stuff in
2. Install opencv and git via brew (or your own favorite package manager, ie ports)

brew update
brew tap homebrew/science
brew install opencv git nasm wget

3. Download the libfreenect2 repository

git clone git@github.com:OpenKinect/libfreenect2.git

4. Install a bunch of dependencies

cd ./libfreenect2
sh ./depends/install_mac.sh

5. Build the actual protonect executable

cd ./examples/protonect/
cmake CMakeLists.txt
make && make install

6. Run the program
./bin/Protonect

==== Other operating systems

I'm not sure, but look for libusbx installation instructions for your OS. Figure out how to attach the driver to the Xbox NUI Sensor composite parent device, VID 045E PID 02C4, then contribute your procedure.

=== Building

Make sure you install the driver as describe above first.

1. Follow directions in the ./depends/README.depends.txt to get the dependencies. (Process may be streamlined later.)

==== Windows / Visual Studio

2. Open the .sln file from ./build/msvc for your version of Visual Studio.
3. Confirm the platform configuration has x64 selected
4. Build and run.

==== Other platforms

2. ?
3. Build and run.
4. Contribute your solution for your platform back to the project please.

== Required notification

The K4W2 hardware is currently pre-release. Per the K4W2 developer program agreement, all public demonstrations and code should display this notice:

"This is preliminary software and/or hardware and APIs are preliminary and subject to change."
