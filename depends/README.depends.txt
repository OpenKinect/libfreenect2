For now to keep binaries out of git, download dependencies separately:

Dependencies:
---

= Windows
libusbx-winiso 
libusbx post-1.0.17 + winiso
winiso = Windows isochronous modifications by Joshua Blake <joshblake@gmail.com>

1. Download pre-built binaries libusbx-winiso.zip from https://www.dropbox.com/s/madoye1ayaoajet/libusbx-winiso.zip
2. Unzip into the depends folder such that the resulting folder structure ends up looking like:
./depends/libusbx/
./depends/libusbx/include/
./depends/libusbx/MS64/
3. Done

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
