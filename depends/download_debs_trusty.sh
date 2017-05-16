#!/bin/sh

set -e

cd `dirname $0`
ARCH=`/usr/bin/dpkg --print-architecture`

# download standalone packages for 14.04 LTS
if [ "$ARCH" = amd64 -o "$ARCH" = i386 ]; then
  REPO=http://archive.ubuntu.com/ubuntu
else
  REPO=http://ports.ubuntu.com/ubuntu-ports
fi

download() {
  path=$1
  ver=$2
  mkdir -p debs
  shift 2
  for pkg in "$@"; do
    wget -nv -N -P debs -nv $REPO/pool/$path/${pkg}_${ver}_${ARCH}.deb
  done
}

download main/libu/libusb-1.0 1.0.20-1 libusb-1.0-0-dev libusb-1.0-0
download universe/g/glfw3 3.1.2-3 libglfw3-dev libglfw3
download main/o/ocl-icd 2.2.8-1 ocl-icd-libopencl1 ocl-icd-opencl-dev
if [ "$ARCH" = amd64 -o "$ARCH" = i386 ]; then
  download universe/libv/libva 1.7.0-1 libva-dev libva-drm1 libva-egl1 libva-glx1 libva-tpi1 libva-wayland1 libva-x11-1 libva1 vainfo
  download main/i/intel-vaapi-driver 1.7.0-1 i965-va-driver
fi
