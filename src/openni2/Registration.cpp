/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2015 individual OpenKinect contributors. See the CONTRIB file
 * for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

#include <iostream>
#include <Driver/OniDriverAPI.h>
#include "Registration.hpp"

using namespace Freenect2Driver;
    
Registration::Registration(libfreenect2::Freenect2Device* dev) :
  dev(dev),
  reg(NULL),
  enabled(false)
{
}

Registration::~Registration() {
  delete reg;
}

void Registration::depthFrame(libfreenect2::Frame* frame) {
  lastDepthFrame = frame;
}

void Registration::colorFrameRGB888(libfreenect2::Frame* colorFrame, libfreenect2::Frame* registeredFrame) 
{
  if (!reg) {
    libfreenect2::Freenect2Device::ColorCameraParams colCamParams = dev->getColorCameraParams();
    libfreenect2::Freenect2Device::IrCameraParams irCamParams = dev->getIrCameraParams();
	{
		libfreenect2::Freenect2Device::ColorCameraParams cp = colCamParams;
		std::cout << "fx=" << cp.fx << ",fy=" << cp.fy <<
			",cx=" << cp.cx << ",cy=" << cp.cy << std::endl;
		libfreenect2::Freenect2Device::IrCameraParams ip = irCamParams;
		std::cout << "fx=" << ip.fx << ",fy=" << ip.fy <<
			",ix=" << ip.cx << ",iy=" << ip.cy <<
			",k1=" << ip.k1 << ",k2=" << ip.k2 << ",k3=" << ip.k3 <<
			",p1=" << ip.p1 << ",p2=" << ip.p2 << std::endl;
	}
    reg = new libfreenect2::Registration(irCamParams, colCamParams);
  }
  
  libfreenect2::Frame undistorted(lastDepthFrame->width, lastDepthFrame->height, lastDepthFrame->bytes_per_pixel);

  reg->apply(colorFrame, lastDepthFrame, &undistorted, registeredFrame);
}

void Registration::setEnable(bool enable) { enabled = enable; }

bool Registration::isEnabled() { return enabled; }
