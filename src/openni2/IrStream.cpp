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

#define _USE_MATH_DEFINES
#include <cmath> // for M_PI
#include "IrStream.hpp"

using namespace Freenect2Driver;

// from NUI library and converted to radians
const float IrStream::HORIZONTAL_FOV = 58.5 * (M_PI / 180);
const float IrStream::VERTICAL_FOV = 45.6 * (M_PI / 180);

IrStream::IrStream(Device* driver_dev, libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg) : VideoStream(driver_dev, pDevice, reg)
{
  video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_GRAY16, 512, 424, 30);
  setVideoMode(video_mode);
}

// Add video modes here as you implement them
IrStream::VideoModeMap IrStream::getSupportedVideoModes() const
{
  VideoModeMap modes;
  // pixelFormat, resolutionX, resolutionY, fps
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_GRAY16, 512, 424, 30)] = 0;

  return modes;
}

void IrStream::populateFrame(libfreenect2::Frame* srcFrame, int srcX, int srcY, OniFrame* dstFrame, int dstX, int dstY, int width, int height) const
{
  dstFrame->sensorType = getSensorType();
  dstFrame->stride = dstFrame->width * sizeof(uint16_t);

  // copy stream buffer from freenect
  copyFrame(static_cast<float*>((void*)srcFrame->data), srcX, srcY, srcFrame->width,
            static_cast<uint16_t*>(dstFrame->data), dstX, dstY, dstFrame->width,
            width, height, mirroring);
}

OniSensorType IrStream::getSensorType() const { return ONI_SENSOR_IR; }

// from StreamBase
OniBool IrStream::isPropertySupported(int propertyId)
{
  switch(propertyId)
  {
    default:
      return VideoStream::isPropertySupported(propertyId);
    case ONI_STREAM_PROPERTY_HORIZONTAL_FOV:
    case ONI_STREAM_PROPERTY_VERTICAL_FOV:
      return true;
  }
}

OniStatus IrStream::getProperty(int propertyId, void* data, int* pDataSize)
{
  switch (propertyId)
  {
    default:
      return VideoStream::getProperty(propertyId, data, pDataSize);

    case ONI_STREAM_PROPERTY_HORIZONTAL_FOV:        // float (radians)
      if (*pDataSize != sizeof(float))
      {
        LogError("Unexpected size for ONI_STREAM_PROPERTY_HORIZONTAL_FOV");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<float*>(data)) = HORIZONTAL_FOV;
      return ONI_STATUS_OK;
    case ONI_STREAM_PROPERTY_VERTICAL_FOV:          // float (radians)
      if (*pDataSize != sizeof(float))
      {
        LogError("Unexpected size for ONI_STREAM_PROPERTY_VERTICAL_FOV");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<float*>(data)) = VERTICAL_FOV;
      return ONI_STATUS_OK;
  }
}
