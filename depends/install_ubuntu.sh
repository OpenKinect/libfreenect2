#!/bin/sh

cd `dirname $0`
DEPENDS_DIR=`pwd`

ARCH="$(uname -m | grep -q 64 && echo 'amd64' || echo 'i386')"

# download standalone packages for 14.04 LTS
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/glfw3/libglfw3_3.0.4-1_${ARCH}.deb
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/glfw3/libglfw3-dev_3.0.4-1_${ARCH}.deb
wget http://mirrors.kernel.org/ubuntu/pool/universe/g/glfw3/libglfw3-doc_3.0.4-1_all.deb

sudo dpkg -i libglfw3*_3.0.4-1_*.deb

sudo apt-get install -y build-essential libjpeg-turbo8-dev libtool autoconf libudev-dev cmake mesa-common-dev freeglut3-dev libxrandr-dev doxygen libxi-dev libopencv-dev automake

# bugfix for broken libjpeg-turbo8-dev package
TURBOLIB="/usr/lib/$(uname -m)-linux-gnu/libturbojpeg.so"
[ -e $TURBOLIB ] || sudo ln -s ${TURBOLIB}.0.0.0 ${TURBOLIB}

sh ./install_libusb.sh
