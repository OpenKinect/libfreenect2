For now to keep binaries out of git, download dependencies separately:

Dependencies:
---

= Windows

This guide is x64 only.

1. Install libusb as one of the two following ways:
    a.Download libusbx with patch from https://www.dropbox.com/s/madoye1ayaoajet/libusbx-winiso.zip
    and install it in /depends/libusbx (this only has x64 release version)
    
    b.Or clone master libusb from https://github.com/libusb/libusb.git
        i. Add joshblakes ISO patch. 
            Add jblake https://github.com/JoshBlake/libusbx.git 
            merge his winiso branch
            
        ii. Build the solution with the shared dll.
            Add an environment variable with name LibUSB_ROOT and set it to the root of libusb build folder.

2. Install turboJPEG:
    a. Download and install turboJPEG http://sourceforge.net/projects/libjpeg-turbo/files
    b. If not installed in default path - Add an environment variable with name TurboJPEG_ROOT and set it to the root of the installed folder.

3. Install GLFW3:
    a. Get latest source release or clone GLFW from here https://github.com/glfw/glfw.git.
    b. Cmake a solution and run the install project
    c. If not installed in default path - Add an environment variable with name GLFW_ROOT and set it to the root of the installed folder.

4. Install opencv
    a. Download opencv 
    b. Cmake a solution
    c. Build & run install project.

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
