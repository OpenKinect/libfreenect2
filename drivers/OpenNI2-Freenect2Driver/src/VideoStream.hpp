#pragma once

#include <sstream>
#include <map>
#include "libfreenect2.h"
#include "libfreenect2/libfreenect2.hpp"
#include "PS1080.h"
#include "Utility.hpp"
#include "Registration.hpp"

namespace Freenect2Driver
{
  class VideoStream : public oni::driver::StreamBase
  {
  private:
    unsigned int frame_id; // number each frame

    virtual OniStatus setVideoMode(OniVideoMode requested_mode) = 0;
    virtual void populateFrame(libfreenect2::Frame* lf2Frame, int srcX, int srcY, OniFrame* oniFrame, int tgtX, int tgtY, int width, int height) const = 0;

  protected:
    static const OniSensorType sensor_type;
    libfreenect2::Freenect2Device* device;
    bool running; // buildFrame() does something iff true
    OniVideoMode video_mode;
    OniCropping cropping;
    bool mirroring;
    Freenect2Driver::Registration* reg;
    bool callPropertyChangedCallback;

    static void copyFrame(float* srcPix, int srcX, int srcY, int srcStride, uint16_t* dstPix, int dstX, int dstY, int dstStride, int width, int height, bool mirroring)
    {
      srcPix += srcX + srcY * srcStride;
      dstPix += dstX + dstY * dstStride;

      for (int y = 0; y < height; y++) {
        uint16_t* dst = dstPix + y * dstStride;
        float* src = srcPix + y * srcStride;
        if (mirroring) {
          dst += width;
          for (int x = 0; x < width; x++)
            *dst-- = *src++;
        } else {
          for (int x = 0; x < width; x++)
            *dst++ = *src++;
        }
      }
    }
    void raisePropertyChanged(int propertyId, const void* data, int dataSize) {
      if (callPropertyChangedCallback)
        StreamBase::raisePropertyChanged(propertyId, data, dataSize);
    }

  public:
    VideoStream(libfreenect2::Freenect2Device* device, Freenect2Driver::Registration* reg) :
      frame_id(1),
      device(device),
      reg(reg),
      callPropertyChangedCallback(false),
      mirroring(false)
      {
        // joy of structs
        memset(&cropping, 0, sizeof(cropping));
        memset(&video_mode, 0, sizeof(video_mode));
      }
    //~VideoStream() { stop();  }

    void setPropertyChangedCallback(oni::driver::PropertyChangedCallback handler, void* pCookie) {
      callPropertyChangedCallback = true;
      StreamBase::setPropertyChangedCallback(handler, pCookie);
    }

    bool buildFrame(libfreenect2::Frame* lf2Frame)
    {
      if (!running)
        return false;

      OniFrame* oniFrame = getServices().acquireFrame();
      oniFrame->frameIndex = frame_id++;
      oniFrame->timestamp = lf2Frame->sequence*33369;
      oniFrame->videoMode = video_mode;
      oniFrame->width = video_mode.resolutionX;
      oniFrame->height = video_mode.resolutionY;

      if (cropping.enabled)
      {
        oniFrame->height = cropping.height;
        oniFrame->width = cropping.width;
        oniFrame->cropOriginX = cropping.originX;
        oniFrame->cropOriginY = cropping.originY;
        oniFrame->croppingEnabled = true;
      }
      else
      {
        oniFrame->cropOriginX = 0;
        oniFrame->cropOriginY = 0;
        oniFrame->croppingEnabled = false;
      }
      int width = std::min(oniFrame->width, (int)lf2Frame->width);
      int height = std::min(oniFrame->height, (int)lf2Frame->height);

      populateFrame(lf2Frame, oniFrame->cropOriginX, oniFrame->cropOriginY, oniFrame, 0, 0, width, height);
      raiseNewFrame(oniFrame);
      getServices().releaseFrame(oniFrame);

      return false;
    }

    // from StreamBase

    OniStatus start()
    {
      running = true;
      return ONI_STATUS_OK;
    }
    void stop() { running = false; }

    // only add to property handlers if the property is generic to all children
    // otherwise, implement in child and call these in default case
    OniBool isPropertySupported(int propertyId)
    {
      switch(propertyId)
      {
        case ONI_STREAM_PROPERTY_VIDEO_MODE:
        case ONI_STREAM_PROPERTY_CROPPING:
        case ONI_STREAM_PROPERTY_MIRRORING:
          return true;
        default:
          return false;
      }
    }

    virtual OniStatus getProperty(int propertyId, void* data, int* pDataSize)
    {
      switch (propertyId)
      {
        default:
        case ONI_STREAM_PROPERTY_HORIZONTAL_FOV:      // float: radians
        case ONI_STREAM_PROPERTY_VERTICAL_FOV:        // float: radians
        case ONI_STREAM_PROPERTY_MAX_VALUE:           // int
        case ONI_STREAM_PROPERTY_MIN_VALUE:           // int
        case ONI_STREAM_PROPERTY_STRIDE:              // int
        case ONI_STREAM_PROPERTY_NUMBER_OF_FRAMES:    // int
        // camera
        case ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE:  // OniBool
        case ONI_STREAM_PROPERTY_AUTO_EXPOSURE:       // OniBool
        // xn
        case XN_STREAM_PROPERTY_INPUT_FORMAT:         // unsigned long long
        case XN_STREAM_PROPERTY_CROPPING_MODE:        // XnCroppingMode
          return ONI_STATUS_NOT_SUPPORTED;

        case ONI_STREAM_PROPERTY_VIDEO_MODE:          // OniVideoMode*
          if (*pDataSize != sizeof(OniVideoMode))
          {
            LogError("Unexpected size for ONI_STREAM_PROPERTY_VIDEO_MODE");
            return ONI_STATUS_ERROR;
          }
          *(static_cast<OniVideoMode*>(data)) = video_mode;
          return ONI_STATUS_OK;

        case ONI_STREAM_PROPERTY_CROPPING:            // OniCropping*
          if (*pDataSize != sizeof(OniCropping))
          {
            LogError("Unexpected size for ONI_STREAM_PROPERTY_CROPPING");
            return ONI_STATUS_ERROR;
          }
          *(static_cast<OniCropping*>(data)) = cropping;
          return ONI_STATUS_OK;

        case ONI_STREAM_PROPERTY_MIRRORING:           // OniBool
          if (*pDataSize != sizeof(OniBool))
          {
            LogError("Unexpected size for ONI_STREAM_PROPERTY_MIRRORING");
            return ONI_STATUS_ERROR;
          }
          *(static_cast<OniBool*>(data)) = mirroring;
          return ONI_STATUS_OK;
      }
    }
    virtual OniStatus setProperty(int propertyId, const void* data, int dataSize)
    {
      switch (propertyId)
      {
        default:
        case ONI_STREAM_PROPERTY_HORIZONTAL_FOV:      // float: radians
        case ONI_STREAM_PROPERTY_VERTICAL_FOV:        // float: radians
        case ONI_STREAM_PROPERTY_MAX_VALUE:           // int
        case ONI_STREAM_PROPERTY_MIN_VALUE:           // int
        case ONI_STREAM_PROPERTY_STRIDE:              // int
        case ONI_STREAM_PROPERTY_NUMBER_OF_FRAMES:    // int
        // camera
        case ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE:  // OniBool
        case ONI_STREAM_PROPERTY_AUTO_EXPOSURE:       // OniBool
        // xn
        case XN_STREAM_PROPERTY_INPUT_FORMAT:         // unsigned long long
        case XN_STREAM_PROPERTY_CROPPING_MODE:        // XnCroppingMode
          return ONI_STATUS_NOT_SUPPORTED;

        case ONI_STREAM_PROPERTY_VIDEO_MODE:          // OniVideoMode*
          if (dataSize != sizeof(OniVideoMode))
          {
            LogError("Unexpected size for ONI_STREAM_PROPERTY_VIDEO_MODE");
            return ONI_STATUS_ERROR;
          }
          if (ONI_STATUS_OK != setVideoMode(*(static_cast<const OniVideoMode*>(data))))
            return ONI_STATUS_NOT_SUPPORTED;
          raisePropertyChanged(propertyId, data, dataSize);
          return ONI_STATUS_OK;

        case ONI_STREAM_PROPERTY_CROPPING:            // OniCropping*
          if (dataSize != sizeof(OniCropping))
          {
            LogError("Unexpected size for ONI_STREAM_PROPERTY_CROPPING");
            return ONI_STATUS_ERROR;
          }
          cropping = *(static_cast<const OniCropping*>(data));
          raisePropertyChanged(propertyId, data, dataSize);
          return ONI_STATUS_OK;

        case ONI_STREAM_PROPERTY_MIRRORING:           // OniBool
          if (dataSize != sizeof(OniBool))
          {
            LogError("Unexpected size for ONI_STREAM_PROPERTY_MIRRORING");
            return ONI_STATUS_ERROR;
          }
          mirroring = *(static_cast<const OniBool*>(data));
          raisePropertyChanged(propertyId, data, dataSize);
          return ONI_STATUS_OK;
      }
    }


    /* todo : from StreamBase
    virtual OniStatus convertDepthToColorCoordinates(StreamBase* colorStream, int depthX, int depthY, OniDepthPixel depthZ, int* pColorX, int* pColorY) { return ONI_STATUS_NOT_SUPPORTED; }
    */
  };
}


/* image video modes reference

FREENECT_VIDEO_RGB             = 0, //< Decompressed RGB mode (demosaicing done by libfreenect)
FREENECT_VIDEO_BAYER           = 1, //< Bayer compressed mode (raw information from camera)
FREENECT_VIDEO_IR_8BIT         = 2, //< 8-bit IR mode
FREENECT_VIDEO_IR_10BIT        = 3, //< 10-bit IR mode
FREENECT_VIDEO_IR_10BIT_PACKED = 4, //< 10-bit packed IR mode
FREENECT_VIDEO_YUV_RGB         = 5, //< YUV RGB mode
FREENECT_VIDEO_YUV_RAW         = 6, //< YUV Raw mode
FREENECT_VIDEO_DUMMY           = 2147483647, //< Dummy value to force enum to be 32 bits wide

ONI_PIXEL_FORMAT_RGB888 = 200,
ONI_PIXEL_FORMAT_YUV422 = 201,
ONI_PIXEL_FORMAT_GRAY8 = 202,
ONI_PIXEL_FORMAT_GRAY16 = 203,
ONI_PIXEL_FORMAT_JPEG = 204,
*/
