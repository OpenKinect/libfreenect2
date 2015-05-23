#include <iostream>
#include "Driver/OniDriverAPI.h"
#include "Registration.hpp"

using namespace FreenectDriver;
    
const int Registration::depthWidth;
const int Registration::depthHeight;

Registration::Registration(libfreenect2::Freenect2Device* dev) :
  dev(dev),
  reg(NULL)
{
  for (int i = 0; i < depthWidth * depthHeight; i++) {
    depth[i] = invalidDepth;
  }
}

Registration::~Registration() {
}

void Registration::depthFrame(OniFrame* frame) {
  uint16_t* src = static_cast<uint16_t*>(frame->data);
  float* dst = depth;
  for (int y = 0; y < std::max(depthHeight, frame->height); y++) {
    for (int x = 0; x < std::max(depthWidth, frame->width); x++) {
      *dst++ = *src++;
    }
  }
}

void Registration::colorFrameRGB888(uint8_t* data, OniFrame* frame) {
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
    reg = new libfreenect2::Registration(&irCamParams, &colCamParams);
  }
  
  float cx, cy;
  uint8_t* dst = static_cast<uint8_t*>(frame->data);
  for (int y = 0; y < depthHeight; y++) {
    for (int x = 0; x < depthWidth; x++) {
      reg->apply(x, y, depth[y * depthWidth + x], cx, cy);
      if (cx < 0.0 || 1920 <= cx || cy < 0.0 || 1080 <= cy) {
        // dark green
        *dst++ = 0x00;
        *dst++ = 0x80;
        *dst++ = 0x00;
      } else {
        uint8_t* src = &data[(int(cy) * 1920 + int(cx)) * 3];
        *dst++ = src[2];
        *dst++ = src[1];
        *dst++ = src[0];
      }
    }
  }
}
