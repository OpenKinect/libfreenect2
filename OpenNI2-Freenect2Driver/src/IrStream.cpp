#include <string>
#include "IrStream.hpp"

using namespace Freenect2Driver;

// from NUI library and converted to radians
const float IrStream::DIAGONAL_FOV = 70 * (M_PI / 180);
const float IrStream::HORIZONTAL_FOV = 58.5 * (M_PI / 180);
const float IrStream::VERTICAL_FOV = 45.6 * (M_PI / 180);
// from DepthKinectStream.cpp
const int IrStream::MAX_VALUE;
const unsigned long long IrStream::GAIN_VAL;
const unsigned long long IrStream::CONST_SHIFT_VAL;
const unsigned long long IrStream::MAX_SHIFT_VAL;
const unsigned long long IrStream::PARAM_COEFF_VAL;
const unsigned long long IrStream::SHIFT_SCALE_VAL;
const unsigned long long IrStream::ZERO_PLANE_DISTANCE_VAL;
const double IrStream::ZERO_PLANE_PIXEL_SIZE_VAL = 0.10520000010728836;
const double IrStream::EMITTER_DCMOS_DISTANCE_VAL = 7.5;

IrStream::IrStream(libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg) : VideoStream(pDevice, reg)
{
  video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_GRAY16, 512, 424, 30);
  setVideoMode(video_mode);
  pDevice->start();
}

// Add video modes here as you implement them
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
