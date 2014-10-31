#/bin/sh

cd `dirname $0`
DEPENDS_DIR=`pwd`

# Ubuntu debian dependencies
sudo apt-get install libusb-dev libturbojpeg opencl-headers

sh ./install_deps.sh
