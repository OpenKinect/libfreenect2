#pragma once

#include <algorithm> // for transform()
#include <cmath> // for M_PI
#include <map>
#include "libfreenect2.h"
#include "libfreenect2/libfreenect2.hpp"
#include "Driver/OniDriverAPI.h"
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
    typedef std::map< OniVideoMode, std::pair<freenect2_video_format, freenect2_resolution> > FreenectVideoModeMap;
    static const OniSensorType sensor_type = ONI_SENSOR_COLOR;

    static FreenectVideoModeMap getSupportedVideoModes();
    OniStatus setVideoMode(OniVideoMode requested_mode);
    void populateFrame(libfreenect2::Frame* srcFrame, int srcX, int srcY, OniFrame* dstFrame, int dstX, int dstY, int width, int height) const;
    
    static void copyFrame(uint8_t* srcPix, int srcX, int srcY, int srcStride, uint8_t* dstPix, int dstX, int dstY, int dstStride, int width, int height, bool mirroring);

    bool auto_white_balance;
    bool auto_exposure;

  public:
    ColorStream(libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg);
    //~ColorStream() { }

    static OniSensorInfo getSensorInfo()
    {
      FreenectVideoModeMap supported_modes = getSupportedVideoModes();
      OniVideoMode* modes = new OniVideoMode[supported_modes.size()];
      std::transform(supported_modes.begin(), supported_modes.end(), modes, ExtractKey());
      OniSensorInfo sensors = { sensor_type, static_cast<int>(supported_modes.size()), modes };
      return sensors;
    }

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
      
#if 0      
        // camera
        case ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE: // OniBool
        {
          if (dataSize != sizeof(OniBool))
          {
            LogError("Unexpected size for ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE");
            return ONI_STATUS_ERROR;
          }
          auto_white_balance = *(static_cast<const OniBool*>(data));
          int ret = device->setFlag(FREENECT_AUTO_WHITE_BALANCE, auto_white_balance);
          return (ret == 0) ? ONI_STATUS_OK : ONI_STATUS_ERROR;
        }
        case ONI_STREAM_PROPERTY_AUTO_EXPOSURE:      // OniBool
        {
          if (dataSize != sizeof(OniBool))
          {
            LogError("Unexpected size for ONI_STREAM_PROPERTY_AUTO_EXPOSURE");
            return ONI_STATUS_ERROR;
          }
          auto_exposure = *(static_cast<const OniBool*>(data));
          int ret = device->setFlag(FREENECT_AUTO_WHITE_BALANCE, auto_exposure);
          return (ret == 0) ? ONI_STATUS_OK : ONI_STATUS_ERROR;
        }
        case ONI_STREAM_PROPERTY_MIRRORING:          // OniBool
        {
          if (dataSize != sizeof(OniBool))
          {
            LogError("Unexpected size for ONI_STREAM_PROPERTY_MIRRORING");
            return ONI_STATUS_ERROR;
          }
          mirroring = *(static_cast<const OniBool*>(data));
          int ret = device->setFlag(FREENECT_MIRROR_VIDEO, mirroring);
          return (ret == 0) ? ONI_STATUS_OK : ONI_STATUS_ERROR;
        }
#endif
      }
    }
  };
}
