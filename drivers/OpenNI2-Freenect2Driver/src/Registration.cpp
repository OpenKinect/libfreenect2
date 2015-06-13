#include <iostream>
#include "Driver/OniDriverAPI.h"
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
