#include <iostream>
#include "Driver/OniDriverAPI.h"
#include "Registration.hpp"

using namespace Freenect2Driver;
    
const int Registration::depthWidth;
const int Registration::depthHeight;
const float Registration::invalidDepth = 0.0;
const float Registration::infiniteDepth = 65536.0;

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

void Registration::depthFrame(libfreenect2::Frame* frame) {
    for (int y = 0; y < std::min(depthHeight, (int)frame->height); y++) {
    float* src = static_cast<float*>((void*)frame->data) + y * frame->width;
    float* dst = depth + y * depthWidth;
    for (int x = 0; x < std::min(depthWidth, (int)frame->width); x++) {
      *dst++ = *src++;
    }
  }
}

void Registration::colorFrameRGB888(libfreenect2::Frame* srcFrame, OniFrame* dstFrame) {
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
  
  float cx, cy;
  uint8_t* dst = static_cast<uint8_t*>(dstFrame->data);
  uint8_t* data = static_cast<uint8_t*>(srcFrame->data);

  for (int i = 0; i < colorWidth * colorHeight; i++) {
    colorDepth[i] = infiniteDepth;
  }
#if 1
  const int delta = 0;
  const int scaleX = 4;
  const int scaleY = 4;
#else
  const int delta = 2;
  const int scaleX = 1;
  const int scaleY = 1;
#endif
  for (int y = 0; y < depthHeight; y++) {
    for (int x = 0; x < depthWidth; x++) {
      float &z = depth[y * depthWidth + x];
      reg->apply(x, y, z, cx, cy);
      for (int dy = -delta; dy <= delta; dy ++) {
        for (int dx = -delta; dx <= delta; dx ++) {
          if (cx < delta || colorWidth-delta <= cx || cy < delta || colorHeight-delta <= cy)
            continue;
          float &cz = colorDepth[(int(cy) + dy)/scaleY * colorWidth/scaleX + (int(cx) + dx)/scaleX];
          if (z < cz)
            cz = z;
        }
      }
    }
  }

  for (int y = 0; y < depthHeight; y++) {
    for (int x = 0; x < depthWidth; x++) {
      float &z = depth[y * depthWidth + x];
      reg->apply(x, y, z, cx, cy);
      float &cz = colorDepth[int(cy)/scaleY * colorWidth/scaleX + int(cx)/scaleX];
      if (cx < 0.0 || srcFrame->width <= cx || cy < 0.0 || srcFrame->height <= cy || 0.2 < (z - cz)/z) {
#if 0
        // dark green
        *dst++ = 0x00;
        *dst++ = 0x80;
        *dst++ = 0x00;
#else
        // black
        *dst++ = 0x00;
        *dst++ = 0x00;
        *dst++ = 0x00;
#endif
      } else {
        uint8_t* src = &data[(int(cy) * srcFrame->width + int(cx)) * 3];
        *dst++ = src[2];
        *dst++ = src[1];
        *dst++ = src[0];
      }
    }
  }
}
