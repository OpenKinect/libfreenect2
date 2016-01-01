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

#pragma once

#include <algorithm> // for transform()
#include <cmath> // for M_PI
#include <cstdio> // for memcpy
#include <libfreenect2/libfreenect2.hpp>
#include <Driver/OniDriverAPI.h>
#include "PS1080.h"
#include "VideoStream.hpp"
#include "S2D.h"
#include "D2S.h"


namespace Freenect2Driver
{
  class DepthStream : public VideoStream
  {
  public:
    // from NUI library and converted to radians
    static const float DIAGONAL_FOV;
    static const float HORIZONTAL_FOV;
    static const float VERTICAL_FOV;
    // from DepthKinectStream.cpp
    static const int MAX_VALUE = 10000;
    static const unsigned long long GAIN_VAL = 42;
    static const unsigned long long CONST_SHIFT_VAL = 200;
    static const unsigned long long MAX_SHIFT_VAL = 2047;
    static const unsigned long long PARAM_COEFF_VAL = 4;
    static const unsigned long long SHIFT_SCALE_VAL = 10;
    static const unsigned long long ZERO_PLANE_DISTANCE_VAL = 120;
    static const double ZERO_PLANE_PIXEL_SIZE_VAL;
    static const double EMITTER_DCMOS_DISTANCE_VAL;

  private:
    OniSensorType getSensorType() const { return ONI_SENSOR_DEPTH; }
    OniImageRegistrationMode image_registration_mode;
    VideoModeMap getSupportedVideoModes() const;
    void populateFrame(libfreenect2::Frame* srcFrame, int srcX, int srcY, OniFrame* dstFrame, int dstX, int dstY, int width, int height) const;

  public:
    DepthStream(libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg);
    //~DepthStream() { }

    OniImageRegistrationMode getImageRegistrationMode() const { return image_registration_mode; }
    OniStatus setImageRegistrationMode(OniImageRegistrationMode mode)
    {
      if (!isImageRegistrationModeSupported(mode))
        return ONI_STATUS_NOT_SUPPORTED;
      image_registration_mode = mode;
      reg->setEnable(image_registration_mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR);
      return setVideoMode(video_mode);
    }

    // from StreamBase
    OniBool isImageRegistrationModeSupported(OniImageRegistrationMode mode) { return (mode == ONI_IMAGE_REGISTRATION_OFF || mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR); }

    OniBool isPropertySupported(int propertyId)
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

    OniStatus getProperty(int propertyId, void* data, int* pDataSize)
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
          *pDataSize = sizeof(S2D);
          //std::copy(S2D, S2D+1, static_cast<OniDepthPixel*>(data));
          memcpy(data, S2D, *pDataSize);
          return ONI_STATUS_OK;
        case XN_STREAM_PROPERTY_D2S_TABLE:              // unsigned short[]
          *pDataSize = sizeof(D2S);
          //std::copy(D2S, D2S+1, static_cast<unsigned short*>(data));
          memcpy(data, D2S, *pDataSize);
          return ONI_STATUS_OK;
      }
    }
  };
}
