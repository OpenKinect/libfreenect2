#/bin/sh
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root" 1>&2
   exit 1
fi

cd `dirname $0`
DEPENDS_DIR=`pwd`

# libusb with superspeed patch
git clone https://github.com/libusb/libusb.git libusb

cd libusb
git checkout v1.0.19
git apply $DEPENDS_DIR/linux_usbfs_increase_max_iso_buffer_length.patch
./bootstrap.sh
./configure
make && make install

cd $DEPENDS_DIR

# glfw
git clone https://github.com/glfw/glfw.git glfw
cd glfw
git checkout 3.0.4
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=TRUE ..
make && make install

cd $DEPENDS_DIR

apt-get install libglewmx-dev libglew-dev 

cd $DEPENDS_DIR
