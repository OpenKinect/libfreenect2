#/bin/sh

cd `dirname $0`
DEPENDS_DIR=`pwd`

sh ./install_deps.sh


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
./configure --disable-dependency-tracking --host x86_64-apple-darwin --with-jpeg8 --prefix=$LIBJPEG_INSTALL_DIR 
make && make install

#get the missing cl.hpp from Khronos.org
cd /System/Library/Frameworks/OpenCL.framework/Versions/A/Headers/ && sudo curl -O http://www.khronos.org/registry/cl/api/1.2/cl.hpp

cd $DEPENDS_DIR
