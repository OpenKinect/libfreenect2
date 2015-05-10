#include <string>
#include "DepthStream.hpp"

using namespace Freenect2Driver;


DepthStream::DepthStream(libfreenect2::Freenect2Device* pDevice) : VideoStream(pDevice)
{
  video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 512, 424, 30);
  image_registration_mode = ONI_IMAGE_REGISTRATION_OFF;
  setVideoMode(video_mode);
  pDevice->start();
}

// Add video modes here as you implement them
// Note: if image_registration_mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR,
// setVideoFormat() will try FREENECT_DEPTH_REGISTERED first then fall back on what is set here.
DepthStream::FreenectDepthModeMap DepthStream::getSupportedVideoModes()
{
  FreenectDepthModeMap modes;
  //                      pixelFormat, resolutionX, resolutionY, fps
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 512, 424, 30)] = 1;

  return modes;
}

OniStatus DepthStream::setVideoMode(OniVideoMode requested_mode)
{
  FreenectDepthModeMap supported_video_modes = getSupportedVideoModes();
  FreenectDepthModeMap::const_iterator matched_mode_iter = supported_video_modes.find(requested_mode);
  if (matched_mode_iter == supported_video_modes.end())
    return ONI_STATUS_NOT_SUPPORTED;

  video_mode = requested_mode;
  return ONI_STATUS_OK;
}

void DepthStream::populateFrame(void* data, OniFrame* frame) const
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

  /*
  uint16_t* data_ptr = static_cast<uint16_t*>(data);
  uint16_t* frame_data = static_cast<uint16_t*>(frame->data);
  if (mirroring)
  {
    for (unsigned int i = 0; i < frame->dataSize / 2; i++)
    {
      // find corresponding mirrored pixel
      unsigned int row = i / video_mode.resolutionX;
      unsigned int col = video_mode.resolutionX - (i % video_mode.resolutionX);
      unsigned int target = (row * video_mode.resolutionX) + col;
      // copy it to this pixel
      frame_data[i] = data_ptr[target];
    }
  }
  else
    std::copy(data_ptr, data_ptr+frame->dataSize / 2, frame_data);
  */
}
