#include <string>
#include "ColorStream.hpp"

using namespace Freenect2Driver;


ColorStream::ColorStream(libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg) : VideoStream(pDevice, reg)
{
  //video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 1920, 1080, 30);
  video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 512, 424, 30);
  setVideoMode(video_mode);
  pDevice->start();
}

// Add video modes here as you implement them
ColorStream::FreenectVideoModeMap ColorStream::getSupportedVideoModes()
{
  FreenectVideoModeMap modes;
  //                    pixelFormat, resolutionX, resolutionY, fps    freenect_video_format, freenect_resolution
  //modes[makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 1920, 1080, 30)] = std::pair<freenect2_video_format, freenect2_resolution>(FREENECT2_VIDEO_RGB, FREENECT2_RESOLUTION_1920x1080);
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 512, 424, 30)] = std::pair<freenect2_video_format, freenect2_resolution>(FREENECT2_VIDEO_RGB, FREENECT2_RESOLUTION_1920x1080);


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

  freenect2_video_format format = matched_mode_iter->second.first;
  freenect2_resolution resolution = matched_mode_iter->second.second;

#if 0
  try { device->setVideoFormat(format, resolution); }
  catch (std::runtime_error e)
  {
    LogError("Format " + to_string(format) + " and resolution " + to_string(resolution) + " combination not supported by libfreenect");
    return ONI_STATUS_NOT_SUPPORTED;
  }
#endif // 0
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
      reg->colorFrameRGB888(static_cast<uint8_t*>(data), frame);
      return;
  }
}

/* color video modes reference

FREENECT_VIDEO_RGB             = 0, //< Decompressed RGB mode (demosaicing done by libfreenect)
FREENECT_VIDEO_BAYER           = 1, //< Bayer compressed mode (raw information from camera)
FREENECT_VIDEO_YUV_RGB         = 5, //< YUV RGB mode
FREENECT_VIDEO_YUV_RAW         = 6, //< YUV Raw mode

ONI_PIXEL_FORMAT_RGB888 = 200,
ONI_PIXEL_FORMAT_YUV422 = 201,
ONI_PIXEL_FORMAT_JPEG = 204,
*/
