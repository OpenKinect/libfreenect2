#/bin/sh
cd `dirname $0`
DEPENDS_DIR=`pwd`

#libusbx with superspeed patch
LIBUSBX_SOURCE_DIR=$DEPENDS_DIR/libusbx_src
LIBUSBX_INSTALL_DIR=$DEPENDS_DIR/libusbx

rm -rf $LIBUSBX_SOURCE_DIR $LIBUSBX_INSTALL_DIR

git clone -b superspeed https://github.com/JoshBlake/libusbx.git $LIBUSBX_SOURCE_DIR

cd $LIBUSBX_SOURCE_DIR
git apply $DEPENDS_DIR/linux_usbfs_increase_max_iso_buffer_length.patch
./bootstrap.sh
./configure --prefix=$LIBUSBX_INSTALL_DIR
make && make install

cd $DEPENDS_DIR



