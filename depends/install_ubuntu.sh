#/bin/sh
cd `dirname $0`
DEPENDS_DIR=`pwd`

# libusbx with superspeed patch
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

# libjpeg
LIBJPEG_SOURCE_DIR=$DEPENDS_DIR/libjpeg_turbo_src
LIBJPEG_INSTALL_DIR=$DEPENDS_DIR/libjpeg_turbo
LIBJPEG_VERSION=1.3.1

wget https://downloads.sourceforge.net/project/libjpeg-turbo/$LIBJPEG_VERSION/libjpeg-turbo-$LIBJPEG_VERSION.tar.gz
tar xvf libjpeg-turbo-$LIBJPEG_VERSION.tar.gz
mv libjpeg-turbo-$LIBJPEG_VERSION $LIBJPEG_SOURCE_DIR
cd $LIBJPEG_SOURCE_DIR

# libjpeg-turbo is missing some files config files (config.guess and config.sub)
cp $LIBUSBX_SOURCE_DIR/config.guess ./
cp $LIBUSBX_SOURCE_DIR/config.sub ./
./configure --disable-dependency-tracking --with-jpeg8 --prefix=$LIBJPEG_INSTALL_DIR 
make && make install

cd $DEPENDS_DIR