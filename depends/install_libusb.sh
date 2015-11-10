#!/bin/sh
set -e

cd `dirname $0`
DEPENDS_DIR=`pwd`

# libusbx with superspeed patch
LIBUSB_SOURCE_DIR=$DEPENDS_DIR/libusb_src
LIBUSB_INSTALL_DIR=$DEPENDS_DIR/libusb

rm -rf $LIBUSB_SOURCE_DIR $LIBUSB_INSTALL_DIR

git clone https://github.com/libusb/libusb.git $LIBUSB_SOURCE_DIR

cd $LIBUSB_SOURCE_DIR
git checkout v1.0.20
./bootstrap.sh
./configure --prefix=$LIBUSB_INSTALL_DIR
make && make install

cd $DEPENDS_DIR

