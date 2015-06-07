#!/bin/sh
set -e

cd `dirname $0`
DEPENDS_DIR=`pwd`

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
