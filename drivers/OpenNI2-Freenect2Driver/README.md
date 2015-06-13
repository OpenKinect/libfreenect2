OpenNI2-Freenect2Driver
======================

OpenNI2-Freenect2Driver is a bridge to libfreenect2 implemented as an OpenNI2 driver.
It allows OpenNI2 to use Kinect for Windows v2 (K4W2) devices on OSX. (and on Linux?)
OpenNI2-Freenect2Driver is derived from OpenNI2-FreenectDriver (https://github.com/OpenKinect/libfreenect/tree/master/OpenNI2-FreenectDriver).

Install
-------
1. Download and unpack [OpenNI](http://structure.io/openni) 2.2.0.33 or higher.
2. Go to the top OpenNI2-Freenect2Driver directory and build it.

        cd /some/where/libfreenect2/OpenNI2-Freenect2Driver
        cmake .
        make

3. Copy the driver to your OpenNI2 driver repository. You must first change `Repository` to match your project layout.

        Repository="/example/path/to/Samples/Bin/OpenNI2/Drivers/"
        cp -L lib/OpenNI2-Freenect2Driver/libFreenect2Driver.{so,dylib} ${Repository}
        
        # you could instead make a symlink to avoid copying after every build
        # ln -s lib/OpenNI2-Freenect2Driver/libFreenect2Driver.{so,dylib} ${Repository}

OpenNI2-Freenect2Driver is currently not built with a static libfreenect2, so you might need to include libfreenect2 when deploying.
You will need to make sure target systems have libusb and all other dependencies also.
