== libfreenect2

This represents a fork of the original libfreenect2, enforced with CMake to find 3rd Party Modules and to make the development of new applications based on libfreenect2 easier than the original version.
Feel free to contribute to this code, but please, don't forget to cite my name.

Maintainer:
* Michele Adduci <info@micheleadduci.net>

libfreenect2 Official Maintainers:
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

=== Remarks

This software has been tested mainly on Linux/Ubuntu and Linux/Arch systems. I don't have the availability of a Windows or a MacOSX machine, with USB3 support.

=== Installation

You need to execute the script to install the dependencies:

`cd depends && sudo ./install_ubuntu.sh`

Once the script has completed the operations, just go to the root folder of the project and run:

`mkdir build && cd build && cmake .. && make`



"This is preliminary software and/or hardware and APIs are preliminary and subject to change."
