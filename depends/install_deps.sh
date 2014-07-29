#/bin/sh
cd `dirname $0`
DEPENDS_DIR=`pwd`

# libusbx with superspeed patch
LIBUSB_SOURCE_DIR=$DEPENDS_DIR/libusb_src
LIBUSB_INSTALL_DIR=$DEPENDS_DIR/libusb

rm -rf $LIBUSB_SOURCE_DIR $LIBUSB_INSTALL_DIR

git clone https://github.com/libusb/libusb.git $LIBUSB_SOURCE_DIR

cd $LIBUSB_SOURCE_DIR
git checkout v1.0.19
git apply $DEPENDS_DIR/linux_usbfs_increase_max_iso_buffer_length.patch
./bootstrap.sh
./configure --prefix=$LIBUSB_INSTALL_DIR
make && make install

cd $DEPENDS_DIR

# glfw
GLFW_SOURCE_DIR=$DEPENDS_DIR/glfw_src
GLFW_INSTALL_DIR=$DEPENDS_DIR/glfw

rm -rf $GLFW_SOURCE_DIR $GLFW_INSTALL_DIR

git clone https://github.com/glfw/glfw.git $GLFW_SOURCE_DIR
cd $GLFW_SOURCE_DIR
git checkout 3.0.4
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$GLFW_INSTALL_DIR -DBUILD_SHARED_LIBS=TRUE ..
make && make install

cd $DEPENDS_DIR

# glew
GLEW_SOURCE_DIR=$DEPENDS_DIR/glew_src
GLEW_INSTALL_DIR=$DEPENDS_DIR/glew

rm -rf $GLEW_SOURCE_DIR $GLEW_INSTALL_DIR

git clone https://github.com/nigels-com/glew.git $GLEW_SOURCE_DIR
cd $GLEW_SOURCE_DIR
export GLEW_DEST=$GLEW_INSTALL_DIR
make extensions && make all && make install.all

cd $DEPENDS_DIR
