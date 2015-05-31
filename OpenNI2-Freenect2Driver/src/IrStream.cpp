#include <string>
#include "IrStream.hpp"

using namespace Freenect2Driver;


IrStream::IrStream(libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg) : VideoStream(pDevice, reg)
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

void IrStream::populateFrame(libfreenect2::Frame* srcFrame, int srcX, int srcY, OniFrame* dstFrame, int dstX, int dstY, int width, int height) const
{
  dstFrame->sensorType = sensor_type;
  dstFrame->stride = dstFrame->width * sizeof(uint16_t);

  // copy stream buffer from freenect
  copyFrame(static_cast<float*>((void*)srcFrame->data), srcX, srcY, srcFrame->width,
            static_cast<uint16_t*>(dstFrame->data), dstX, dstY, dstFrame->width,
            width, height, mirroring);
}
