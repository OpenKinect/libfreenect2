OpenNI2-Freenect2Driver
======================

OpenNI2-Freenect2Driver is a bridge to libfreenect2 implemented as an OpenNI2 driver.
It allows OpenNI2 to use Kinect for Windows v2 (K4W2) devices on Mac OS X. (and on Linux?)
OpenNI2-Freenect2Driver is derived from OpenNI2-FreenectDriver (https://github.com/OpenKinect/libfreenect/tree/master/OpenNI2-FreenectDriver).

Install
-------
Please refer libfreenect2 install documentation at first. This description assumes that you are familiar with the libfreenect2 build instruction.

1. You need [OpenNI](http://structure.io/openni) 2.2.0.33 or higher installed on your system and set the environment variables properly. You can use homebrew to install OpenNI if you use Mac OS X. And you will need to make sure target systems have libusb and all other dependencies also.

        $ brew tap homebrew/science
        $ brew install openni2

        $ export OPENNI2_REDIST=/usr/local/lib/ni2
        $ export OPENNI2_INCLUDE=/usr/local/include/ni2

2. Go to the top libfreenect2 directory and build it. The build option which enables to build this driver is OFF by default. You must specify -DBUILD_OPENNI2_DRIVER=ON in cmake argument. And you may specify additional -DENABLE_OPENGL=NO in cmake argument to use OpenNI2's NiViewer.

        $ cd /some/where/libfreenect2
        $ mkdir build
        $ cd build
        $ cmake -DBUILD_OPENNI2_DRIVER=ON ..
        $ make

3. 'make install' copies the driver to your OpenNI2 driver repository (${OPENNI2_REDIST}/OpenNI2/Drivers)

        $ make install
