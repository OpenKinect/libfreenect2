#include <string>
#include "IrStream.hpp"

using namespace FreenectDriver;


IrStream::IrStream(libfreenect2::Freenect2Device* pDevice) : VideoStream(pDevice)
{
  video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_GRAY16, 512, 424, 30);
  image_registration_mode = ONI_IMAGE_REGISTRATION_OFF;
  setVideoMode(video_mode);
  pDevice->start();
}

// Add video modes here as you implement them
// Note: if image_registration_mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR,
// setVideoFormat() will try FREENECT_DEPTH_REGISTERED first then fall back on what is set here.
IrStream::FreenectIrModeMap IrStream::getSupportedVideoModes()
{
  FreenectIrModeMap modes;
  // pixelFormat, resolutionX, resolutionY, fps
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_GRAY16, 512, 424, 30)] = std::pair<freenect2_ir_format, freenect2_resolution>(FREENECT2_IR_RAW, FREENECT2_RESOLUTION_512x424);


  return modes;
}

OniStatus IrStream::setVideoMode(OniVideoMode requested_mode)
{
  FreenectIrModeMap supported_video_modes = getSupportedVideoModes();
  FreenectIrModeMap::const_iterator matched_mode_iter = supported_video_modes.find(requested_mode);
  if (matched_mode_iter == supported_video_modes.end())
    return ONI_STATUS_NOT_SUPPORTED;

  video_mode = requested_mode;
  return ONI_STATUS_OK;
}

void IrStream::populateFrame(void* data, OniFrame* frame) const
{
  frame->sensorType = sensor_type;
  frame->stride = video_mode.resolutionX * sizeof(uint16_t);

  if (cropping.enabled)
  {
    frame->height = cropping.height;
    frame->width = cropping.width;
    frame->cropOriginX = cropping.originX;
    frame->cropOriginY = cropping.originY;
    frame->croppingEnabled = true;
  }
  else
  {
    frame->cropOriginX = 0;
    frame->cropOriginY = 0;
    frame->croppingEnabled = false;
  }


  // copy stream buffer from freenect

  float* source = static_cast<float*>(data) + frame->cropOriginX + frame->cropOriginY * video_mode.resolutionX;
  uint16_t* target = static_cast<uint16_t*>(frame->data);
  const unsigned int skipWidth = video_mode.resolutionX - frame->width;

  if (mirroring)
  {
    target += frame->width;

    for (int y = 0; y < frame->height; y++)
    {
      for (int x = 0; x < frame->width; x++)
      {
        *target-- = *source++;
      }

      source += skipWidth;
      target += 2 * frame->width;
    }
  }
  else
  {
    for (int y = 0; y < frame->height; y++)
    {
      for (int x = 0; x < frame->width; x++)
      {
        *target++ = *source++;
      }

      source += skipWidth;
    }
  }
}
