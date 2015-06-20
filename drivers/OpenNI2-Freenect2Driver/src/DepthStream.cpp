#include <string>
#include "DepthStream.hpp"

using namespace Freenect2Driver;

// from NUI library and converted to radians
const float DepthStream::DIAGONAL_FOV = 70 * (M_PI / 180);
const float DepthStream::HORIZONTAL_FOV = 58.5 * (M_PI / 180);
const float DepthStream::VERTICAL_FOV = 45.6 * (M_PI / 180);
// from DepthKinectStream.cpp
const int DepthStream::MAX_VALUE;
const unsigned long long DepthStream::GAIN_VAL;
const unsigned long long DepthStream::CONST_SHIFT_VAL;
const unsigned long long DepthStream::MAX_SHIFT_VAL;
const unsigned long long DepthStream::PARAM_COEFF_VAL;
const unsigned long long DepthStream::SHIFT_SCALE_VAL;
const unsigned long long DepthStream::ZERO_PLANE_DISTANCE_VAL;
const double DepthStream::ZERO_PLANE_PIXEL_SIZE_VAL = 0.10520000010728836;
const double DepthStream::EMITTER_DCMOS_DISTANCE_VAL = 7.5;

DepthStream::DepthStream(libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg) : VideoStream(pDevice, reg)
{
  //video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 512, 424, 30);
  video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 640, 480, 30);
  setVideoMode(video_mode);
  setImageRegistrationMode(ONI_IMAGE_REGISTRATION_OFF);
  pDevice->start();
}

// Add video modes here as you implement them
// Note: if image_registration_mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR,
// setVideoFormat() will try FREENECT_DEPTH_REGISTERED first then fall back on what is set here.
DepthStream::FreenectDepthModeMap DepthStream::getSupportedVideoModes()
{
  FreenectDepthModeMap modes;
  //                      pixelFormat, resolutionX, resolutionY, fps
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 640, 480, 30)] = std::pair<freenect2_depth_format, freenect2_resolution>(FREENECT2_DEPTH_MM, FREENECT2_RESOLUTION_512x424);
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 512, 424, 30)] = std::pair<freenect2_depth_format, freenect2_resolution>(FREENECT2_DEPTH_MM, FREENECT2_RESOLUTION_512x424);

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

void DepthStream::populateFrame(libfreenect2::Frame* srcFrame, int srcX, int srcY, OniFrame* dstFrame, int dstX, int dstY, int width, int height) const
{
  dstFrame->sensorType = sensor_type;
  dstFrame->stride = dstFrame->width * sizeof(uint16_t);

  // XXX, save depth map for registration
  if (reg->isEnabled())
    reg->depthFrame(srcFrame);

  if (srcFrame->width < dstFrame->width || srcFrame->height < dstFrame->height)
    memset(dstFrame->data, 0x00, dstFrame->width * dstFrame->height * 2);

  // copy stream buffer from freenect
  copyFrame(static_cast<float*>((void*)srcFrame->data), srcX, srcY, srcFrame->width,
            static_cast<uint16_t*>(dstFrame->data), dstX, dstY, dstFrame->width,
            width, height, mirroring);
}
