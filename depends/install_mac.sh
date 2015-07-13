#!/bin/sh

cd `dirname $0`
DEPENDS_DIR=`pwd`
OS_VER=`sw_vers -productVersion | sed 's/^\([0-9]*\)\.\([0-9]*\).*$/\1.\2/'`

sh ./install_libusb.sh
sh ./install_glfw.sh

#get the missing cl.hpp from Khronos.org
cd /System/Library/Frameworks/OpenCL.framework/Versions/A/Headers/ ||
  cd /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX${OS_VER}.sdk/System/Library/Frameworks/OpenCL.framework/Versions/A/Headers
[ -f cl.hpp ] || sudo wget http://www.khronos.org/registry/cl/api/1.2/cl.hpp

cd $DEPENDS_DIR
