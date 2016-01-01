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
#include <map>
#include <libfreenect2/libfreenect2.hpp>
#include <Driver/OniDriverAPI.h>
#include "VideoStream.hpp"


namespace Freenect2Driver
{
  class ColorStream : public VideoStream
  {
  public:
    // from NUI library & converted to radians
    static const float DIAGONAL_FOV;
    static const float HORIZONTAL_FOV;
    static const float VERTICAL_FOV;

  private:
    typedef std::map< OniVideoMode, int > FreenectVideoModeMap;
    OniSensorType getSensorType() const { return ONI_SENSOR_COLOR; }
    VideoModeMap getSupportedVideoModes() const;
    void populateFrame(libfreenect2::Frame* srcFrame, int srcX, int srcY, OniFrame* dstFrame, int dstX, int dstY, int width, int height) const;
    
    static void copyFrame(uint8_t* srcPix, int srcX, int srcY, int srcStride, uint8_t* dstPix, int dstX, int dstY, int dstStride, int width, int height, bool mirroring);

    bool auto_white_balance;
    bool auto_exposure;

  public:
    ColorStream(libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg);
    //~ColorStream() { }

    OniStatus setImageRegistrationMode(OniImageRegistrationMode mode)
    {
      if (mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR) {
        // XXX, switch color resolution to 512x424 for registrarion here
        OniVideoMode video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 512, 424, 30);
        setProperty(ONI_STREAM_PROPERTY_VIDEO_MODE, &video_mode, sizeof(video_mode));
      }
      return ONI_STATUS_OK;
    }

    // from StreamBase
    OniBool isPropertySupported(int propertyId)
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

    OniStatus getProperty(int propertyId, void* data, int* pDataSize)
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
    
    OniStatus setProperty(int propertyId, const void* data, int dataSize)
    {
      switch (propertyId)
      {
        default:
          return VideoStream::setProperty(propertyId, data, dataSize);      
      }
    }
  };
}
