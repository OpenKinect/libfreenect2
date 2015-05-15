#pragma once

#include <algorithm> // for transform()
#include <cmath> // for M_PI
#include <cstdio> // for memcpy
#include <libfreenect2/libfreenect2.hpp>
#include <Driver/OniDriverAPI.h>
#include "PS1080.h"
#include "VideoStream.hpp"

namespace Freenect2Driver
{
  class IrStream : public VideoStream
  {
  public:
    // from NUI library and converted to radians
    static const float HORIZONTAL_FOV = 58.5 * (M_PI / 180);
    static const float VERTICAL_FOV = 45.6 * (M_PI / 180);

  private:
    typedef std::map< OniVideoMode, int > FreenectIrModeMap;
    static const OniSensorType sensor_type = ONI_SENSOR_IR;
    OniImageRegistrationMode image_registration_mode;

    static FreenectIrModeMap getSupportedVideoModes();
    OniStatus setVideoMode(OniVideoMode requested_mode);
    void populateFrame(void* data, OniFrame* frame) const;

  public:
    IrStream(libfreenect2::Freenect2Device* pDevice);
    //~IrStream() { }

    static OniSensorInfo getSensorInfo()
    {
      FreenectIrModeMap supported_modes = getSupportedVideoModes();
      OniVideoMode* modes = new OniVideoMode[supported_modes.size()];
      std::transform(supported_modes.begin(), supported_modes.end(), modes, ExtractKey());
      OniSensorInfo sensors = { sensor_type, static_cast<int>(supported_modes.size()), modes };
      return sensors;
    }

    OniImageRegistrationMode getImageRegistrationMode() const { return image_registration_mode; }
    OniStatus setImageRegistrationMode(OniImageRegistrationMode mode)
    {
      if (!isImageRegistrationModeSupported(mode))
        return ONI_STATUS_NOT_SUPPORTED;
      image_registration_mode = mode;
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
      }
    }
  };
}
