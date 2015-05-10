#include <string>
#include "ColorStream.hpp"

using namespace Freenect2Driver;


ColorStream::ColorStream(libfreenect2::Freenect2Device* pDevice) : VideoStream(pDevice)
{
  video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 1920, 1080, 30);
  setVideoMode(video_mode);
  pDevice->start();
}

// Add video modes here as you implement them
ColorStream::FreenectVideoModeMap ColorStream::getSupportedVideoModes()
{
  FreenectVideoModeMap modes;
  //                    pixelFormat, resolutionX, resolutionY, fps    freenect_video_format, freenect_resolution
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 1920, 1080, 30)] = 1;

  return modes;
 
  /* working format possiblities
  FREENECT_VIDEO_RGB
  FREENECT_VIDEO_YUV_RGB
  FREENECT_VIDEO_YUV_RAW
  */
}

OniStatus ColorStream::setVideoMode(OniVideoMode requested_mode)
{
  FreenectVideoModeMap supported_video_modes = getSupportedVideoModes();
  FreenectVideoModeMap::const_iterator matched_mode_iter = supported_video_modes.find(requested_mode);
  if (matched_mode_iter == supported_video_modes.end())
    return ONI_STATUS_NOT_SUPPORTED;

  video_mode = requested_mode;
  return ONI_STATUS_OK;
}

void ColorStream::populateFrame(void* data, OniFrame* frame) const
{
  frame->sensorType = sensor_type;
  frame->stride = video_mode.resolutionX * 3;
  frame->cropOriginX = 0;
  frame->cropOriginY = 0;
  frame->croppingEnabled = false;

  // copy stream buffer from freenect
  switch (video_mode.pixelFormat)
  {
    default:
      LogError("Pixel format " + to_string(video_mode.pixelFormat) + " not supported by populateFrame()");
      return;

    case ONI_PIXEL_FORMAT_RGB888:
      uint8_t* source = static_cast<uint8_t*>(data);
      uint8_t* target = static_cast<uint8_t*>(frame->data);
      for (uint8_t* p = source; p < source + frame->dataSize; p+=3) {
          *target++ = p[2];
          *target++ = p[1];
          *target++ = p[0];
      }
      return;
  }
}
