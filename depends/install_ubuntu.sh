#!/bin/sh

cd `dirname $0`
ARCH=`/usr/bin/dpkg --print-architecture`

# download standalone packages for 14.04 LTS
if [ "$ARCH" = amd64 -o "$ARCH" = i386 ]; then
  REPO=http://archive.ubuntu.com/ubuntu
else
  REPO=http://ports.ubuntu.com/ubuntu-ports
fi
wget -N $REPO/pool/universe/g/glfw3/libglfw3_3.0.4-1_${ARCH}.deb
wget -N $REPO/pool/universe/g/glfw3/libglfw3-dev_3.0.4-1_${ARCH}.deb

cat <<-EOT

	Execute the following commands to install the remaining dependencies (if you have not already done so):

	sudo dpkg -i libglfw3*_3.0.4-1_*.deb
	sudo apt-get install build-essential cmake pkg-config libturbojpeg libjpeg-turbo8-dev mesa-common-dev freeglut3-dev libxrandr-dev libxi-dev
	sudo apt-get install libturbojpeg0-dev # (Debian)

EOT

