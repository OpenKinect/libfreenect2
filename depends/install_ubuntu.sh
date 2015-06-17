#!/bin/sh

cd `dirname $0`
DEPENDS_DIR=`pwd`

ARCH="$(uname -m | grep -q 64 && echo 'amd64' || echo 'i386')"

# download standalone packages for 14.04 LTS
wget -N http://archive.ubuntu.com/ubuntu/pool/universe/g/glfw3/libglfw3_3.0.4-1_${ARCH}.deb
wget -N http://archive.ubuntu.com/ubuntu/pool/universe/g/glfw3/libglfw3-dev_3.0.4-1_${ARCH}.deb

sh ./install_libusb.sh

cat <<-EOT

	Execute the following commands to install the remaining dependencies (if you have not already done so):

	sudo dpkg -i libglfw3*_3.0.4-1_*.deb
	sudo apt-get install build-essential libturbojpeg libjpeg-turbo8-dev libtool autoconf libudev-dev cmake mesa-common-dev freeglut3-dev libxrandr-dev doxygen libxi-dev libopencv-dev automake
	sudo apt-get install libturbojpeg0-dev # (Debian)

EOT

