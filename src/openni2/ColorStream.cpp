/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2014 Benn Snyder, 2015 individual OpenKinect contributors.
 * See the CONTRIB file for details.
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
#include "ColorStream.hpp"

using namespace Freenect2Driver;

// from NUI library & converted to radians
const float ColorStream::DIAGONAL_FOV = 73.9 * (M_PI / 180);
const float ColorStream::HORIZONTAL_FOV = 62 * (M_PI / 180);
const float ColorStream::VERTICAL_FOV = 48.6 * (M_PI / 180);

ColorStream::ColorStream(Device* driver_dev, libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg) : VideoStream(driver_dev, pDevice, reg)
{
  video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 1920, 1080, 30);
  setVideoMode(video_mode);
}

// Add video modes here as you implement them
ColorStream::FreenectVideoModeMap ColorStream::getSupportedVideoModes() const
{
  FreenectVideoModeMap modes;
  //                    pixelFormat, resolutionX, resolutionY, fps    freenect_video_format, freenect_resolution
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 512, 424, 30)] = 0;
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 1920, 1080, 30)] = 1;

  return modes;
 
  /* working format possiblities
  FREENECT_VIDEO_RGB
  FREENECT_VIDEO_YUV_RGB
  FREENECT_VIDEO_YUV_RAW
  */
}

void ColorStream::populateFrame(libfreenect2::Frame* srcFrame, int srcX, int srcY, OniFrame* dstFrame, int dstX, int dstY, int width, int height) const
{
  dstFrame->sensorType = getSensorType();
  dstFrame->stride = dstFrame->width * 3;

  // copy stream buffer from freenect
  switch (video_mode.pixelFormat)
  {
    default:
      LogError("Pixel format " + to_string(video_mode.pixelFormat) + " not supported by populateFrame()");
      return;

    case ONI_PIXEL_FORMAT_RGB888:
      if (reg->isEnabled()) {
        libfreenect2::Frame registered(512, 424, 4);

        reg->colorFrameRGB888(srcFrame, &registered);

        copyFrame(static_cast<uint8_t*>(registered.data), srcX, srcY, registered.width * registered.bytes_per_pixel, 
                  static_cast<uint8_t*>(dstFrame->data), dstX, dstY, dstFrame->stride, 
                  width, height, mirroring);
      } else {
        copyFrame(static_cast<uint8_t*>(srcFrame->data), srcX, srcY, srcFrame->width * srcFrame->bytes_per_pixel, 
                  static_cast<uint8_t*>(dstFrame->data), dstX, dstY, dstFrame->stride, 
                  width, height, mirroring);
      }
      return;
  }
}

void ColorStream::copyFrame(uint8_t* srcPix, int srcX, int srcY, int srcStride, uint8_t* dstPix, int dstX, int dstY, int dstStride, int width, int height, bool mirroring)
{
  srcPix += srcX + srcY * srcStride;
  dstPix += dstX + dstY * dstStride;

  for (int y = 0; y < height; y++) {
    uint8_t* dst = dstPix + y * dstStride;
    uint8_t* src = srcPix + y * srcStride;
    if (mirroring) {
      dst += dstStride - 1;
      for (int x = 0; x < srcStride; ++x)
      {
        if (x % 4 != 3)
        {
          *dst-- = *src++;
        }
        else
        {
          ++src;
        }
      }
    } else {
      for (int x = 0; x < dstStride-2; x += 3)
      {
        *dst++ = src[2];
        *dst++ = src[1];
        *dst++ = src[0];
        src += 4;
      }
    }
  }
}

OniSensorType ColorStream::getSensorType() const { return ONI_SENSOR_COLOR; }

OniStatus ColorStream::setImageRegistrationMode(OniImageRegistrationMode mode)
{
  if (mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR) {
    // XXX, switch color resolution to 512x424 for registrarion here
    OniVideoMode video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 512, 424, 30);
    setProperty(ONI_STREAM_PROPERTY_VIDEO_MODE, &video_mode, sizeof(video_mode));
  }
  return ONI_STATUS_OK;
}

// from StreamBase
OniBool ColorStream::isPropertySupported(int propertyId)
{
  switch(propertyId)
  {
    default:
      return VideoStream::isPropertySupported(propertyId);

    case ONI_STREAM_PROPERTY_HORIZONTAL_FOV:
    case ONI_STREAM_PROPERTY_VERTICAL_FOV:
    case ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE:
    case ONI_STREAM_PROPERTY_AUTO_EXPOSURE:
      return true;
  }
}

OniStatus ColorStream::getProperty(int propertyId, void* data, int* pDataSize)
{
  switch (propertyId)
  {
    default:
      return VideoStream::getProperty(propertyId, data, pDataSize);

    case ONI_STREAM_PROPERTY_HORIZONTAL_FOV:     // float (radians)
    {
      if (*pDataSize != sizeof(float))
      {
        LogError("Unexpected size for ONI_STREAM_PROPERTY_HORIZONTAL_FOV");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<float*>(data)) = HORIZONTAL_FOV;
      return ONI_STATUS_OK;
    }
    case ONI_STREAM_PROPERTY_VERTICAL_FOV:       // float (radians)
    {
      if (*pDataSize != sizeof(float))
      {
        LogError("Unexpected size for ONI_STREAM_PROPERTY_VERTICAL_FOV");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<float*>(data)) = VERTICAL_FOV;
      return ONI_STATUS_OK;
    }

    // camera
    case ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE: // OniBool
    {
      if (*pDataSize != sizeof(OniBool))
      {
        LogError("Unexpected size for ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<OniBool*>(data)) = auto_white_balance;
      return ONI_STATUS_OK;
    }
    case ONI_STREAM_PROPERTY_AUTO_EXPOSURE:      // OniBool
    {
      if (*pDataSize != sizeof(OniBool))
      {
        LogError("Unexpected size for ONI_STREAM_PROPERTY_AUTO_EXPOSURE");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<OniBool*>(data)) = auto_exposure;
      return ONI_STATUS_OK;
    }
  }
}

OniStatus ColorStream::setProperty(int propertyId, const void* data, int dataSize)
{
  switch (propertyId)
  {
    case 0:
    default:
      return VideoStream::setProperty(propertyId, data, dataSize);
  }
}
