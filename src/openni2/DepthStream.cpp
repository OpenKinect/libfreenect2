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
#include "DepthStream.hpp"

using namespace Freenect2Driver;

// from NUI library and converted to radians
const float DepthStream::DIAGONAL_FOV = 70 * (M_PI / 180);
const float DepthStream::HORIZONTAL_FOV = 58.5 * (M_PI / 180);
const float DepthStream::VERTICAL_FOV = 45.6 * (M_PI / 180);
// from DepthKinectStream.cpp
const int DepthStream::MAX_VALUE;
const unsigned long long DepthStream::GAIN_VAL;
const unsigned long long DepthStream::CONST_SHIFT_VAL;
const unsigned long long DepthStream::MAX_SHIFT_VAL;
const unsigned long long DepthStream::PARAM_COEFF_VAL;
const unsigned long long DepthStream::SHIFT_SCALE_VAL;
const unsigned long long DepthStream::ZERO_PLANE_DISTANCE_VAL;
const double DepthStream::ZERO_PLANE_PIXEL_SIZE_VAL = 0.10520000010728836;
const double DepthStream::EMITTER_DCMOS_DISTANCE_VAL = 7.5;

DepthStream::DepthStream(Device* driver_dev, libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg) : VideoStream(driver_dev, pDevice, reg)
{
  //video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 512, 424, 30);
  video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 640, 480, 30);
  setVideoMode(video_mode);
  setImageRegistrationMode(ONI_IMAGE_REGISTRATION_OFF);
}

// Add video modes here as you implement them
// Note: if image_registration_mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR,
// setVideoFormat() will try FREENECT_DEPTH_REGISTERED first then fall back on what is set here.
DepthStream::VideoModeMap DepthStream::getSupportedVideoModes() const
{
  VideoModeMap modes;
  //                      pixelFormat, resolutionX, resolutionY, fps
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 640, 480, 30)] = 0;
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 512, 424, 30)] = 1;

  return modes;
}

void DepthStream::populateFrame(libfreenect2::Frame* srcFrame, int srcX, int srcY, OniFrame* dstFrame, int dstX, int dstY, int width, int height) const
{
  dstFrame->sensorType = getSensorType();
  dstFrame->stride = dstFrame->width * sizeof(uint16_t);

  // XXX, save depth map for registration
  if (reg->isEnabled())
    reg->depthFrame(srcFrame);

  if (srcFrame->width < (size_t)dstFrame->width || srcFrame->height < (size_t)dstFrame->height)
    memset(dstFrame->data, 0x00, dstFrame->width * dstFrame->height * 2);

  // copy stream buffer from freenect
  copyFrame(static_cast<float*>((void*)srcFrame->data), srcX, srcY, srcFrame->width,
            static_cast<uint16_t*>(dstFrame->data), dstX, dstY, dstFrame->width,
            width, height, mirroring);
}

OniSensorType DepthStream::getSensorType() const { return ONI_SENSOR_DEPTH; }

OniImageRegistrationMode DepthStream::getImageRegistrationMode() const { return image_registration_mode; }

OniStatus DepthStream::setImageRegistrationMode(OniImageRegistrationMode mode)
{
  if (!isImageRegistrationModeSupported(mode))
    return ONI_STATUS_NOT_SUPPORTED;
  image_registration_mode = mode;
  reg->setEnable(image_registration_mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR);
  return setVideoMode(video_mode);
}

// from StreamBase
OniBool DepthStream::isImageRegistrationModeSupported(OniImageRegistrationMode mode) { return (mode == ONI_IMAGE_REGISTRATION_OFF || mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR); }

OniBool DepthStream::isPropertySupported(int propertyId)
{
  switch(propertyId)
  {
    default:
      return VideoStream::isPropertySupported(propertyId);
    case ONI_STREAM_PROPERTY_HORIZONTAL_FOV:
    case ONI_STREAM_PROPERTY_VERTICAL_FOV:
    case ONI_STREAM_PROPERTY_MAX_VALUE:
    case XN_STREAM_PROPERTY_GAIN:
    case XN_STREAM_PROPERTY_CONST_SHIFT:
    case XN_STREAM_PROPERTY_MAX_SHIFT:
    case XN_STREAM_PROPERTY_PARAM_COEFF:
    case XN_STREAM_PROPERTY_SHIFT_SCALE:
    case XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE:
    case XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE:
    case XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE:
    case XN_STREAM_PROPERTY_S2D_TABLE:
    case XN_STREAM_PROPERTY_D2S_TABLE:
      return true;
  }
}

OniStatus DepthStream::getProperty(int propertyId, void* data, int* pDataSize)
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
    case ONI_STREAM_PROPERTY_MAX_VALUE:             // int
      if (*pDataSize != sizeof(int))
      {
        LogError("Unexpected size for ONI_STREAM_PROPERTY_MAX_VALUE");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<int*>(data)) = MAX_VALUE;
      return ONI_STATUS_OK;

    case XN_STREAM_PROPERTY_PIXEL_REGISTRATION:     // XnPixelRegistration (get only)
    case XN_STREAM_PROPERTY_WHITE_BALANCE_ENABLED:  // unsigned long long
    case XN_STREAM_PROPERTY_HOLE_FILTER:            // unsigned long long
    case XN_STREAM_PROPERTY_REGISTRATION_TYPE:      // XnProcessingType
    case XN_STREAM_PROPERTY_AGC_BIN:                // XnDepthAGCBin*
    case XN_STREAM_PROPERTY_PIXEL_SIZE_FACTOR:      // unsigned long long
    case XN_STREAM_PROPERTY_DCMOS_RCMOS_DISTANCE:   // double
    case XN_STREAM_PROPERTY_CLOSE_RANGE:            // unsigned long long
      return ONI_STATUS_NOT_SUPPORTED;

    case XN_STREAM_PROPERTY_GAIN:                   // unsigned long long
      if (*pDataSize != sizeof(unsigned long long))
      {
        LogError("Unexpected size for XN_STREAM_PROPERTY_GAIN");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<unsigned long long*>(data)) = GAIN_VAL;
      return ONI_STATUS_OK;
    case XN_STREAM_PROPERTY_CONST_SHIFT:            // unsigned long long
      if (*pDataSize != sizeof(unsigned long long))
      {
        LogError("Unexpected size for XN_STREAM_PROPERTY_CONST_SHIFT");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<unsigned long long*>(data)) = CONST_SHIFT_VAL;
      return ONI_STATUS_OK;
    case XN_STREAM_PROPERTY_MAX_SHIFT:              // unsigned long long
      if (*pDataSize != sizeof(unsigned long long))
      {
        LogError("Unexpected size for XN_STREAM_PROPERTY_MAX_SHIFT");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<unsigned long long*>(data)) = MAX_SHIFT_VAL;
      return ONI_STATUS_OK;
    case XN_STREAM_PROPERTY_PARAM_COEFF:            // unsigned long long
      if (*pDataSize != sizeof(unsigned long long))
      {
        LogError("Unexpected size for XN_STREAM_PROPERTY_PARAM_COEFF");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<unsigned long long*>(data)) = PARAM_COEFF_VAL;
      return ONI_STATUS_OK;
    case XN_STREAM_PROPERTY_SHIFT_SCALE:            // unsigned long long
      if (*pDataSize != sizeof(unsigned long long))
      {
        LogError("Unexpected size for XN_STREAM_PROPERTY_SHIFT_SCALE");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<unsigned long long*>(data)) = SHIFT_SCALE_VAL;
      return ONI_STATUS_OK;
    case XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE:    // unsigned long long
      if (*pDataSize != sizeof(unsigned long long))
      {
        LogError("Unexpected size for XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<unsigned long long*>(data)) = ZERO_PLANE_DISTANCE_VAL;
      return ONI_STATUS_OK;
    case XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE:  // double
      if (*pDataSize != sizeof(double))
      {
        LogError("Unexpected size for XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<double*>(data)) = ZERO_PLANE_PIXEL_SIZE_VAL;
      return ONI_STATUS_OK;
    case XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE: // double
      if (*pDataSize != sizeof(double))
      {
        LogError("Unexpected size for XN_STREAM_PROPERTY_EMITTER_DCMOS_DISTANCE");
        return ONI_STATUS_ERROR;
      }
      *(static_cast<double*>(data)) = EMITTER_DCMOS_DISTANCE_VAL;
      return ONI_STATUS_OK;
    case XN_STREAM_PROPERTY_S2D_TABLE:              // OniDepthPixel[]
      {
        uint16_t *s2d = (uint16_t *)data;
        *pDataSize = sizeof(*s2d) * 2048;
        memset(data, 0, *pDataSize);
        for (int i = 1; i <= 1052; i++)
          s2d[i] = 342205.0/(1086.671 - i);
      }
      return ONI_STATUS_OK;
    case XN_STREAM_PROPERTY_D2S_TABLE:              // unsigned short[]
      {
        uint16_t *d2s = (uint16_t *)data;
        *pDataSize = sizeof(*d2s) * 10001;
        memset(data, 0, *pDataSize);
        for (int i = 315; i <= 10000; i++)
          d2s[i] = 1086.671 - 342205.0/(i + 1);
      }
      return ONI_STATUS_OK;
  }
}
