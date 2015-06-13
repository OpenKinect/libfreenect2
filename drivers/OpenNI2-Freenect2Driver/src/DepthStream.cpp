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

#if 0
  freenect_depth_format format = matched_mode_iter->second.first;
  freenect_resolution resolution = matched_mode_iter->second.second;
  if (image_registration_mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR) // try forcing registration mode
    format = FREENECT_DEPTH_REGISTERED;

  try { device->setDepthFormat(format, resolution); }
  catch (std::runtime_error e)
  {
    LogError("Format " + to_string(format) + " and resolution " + to_string(resolution) + " combination not supported by libfreenect");
    if (image_registration_mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR)
    {
      LogError("Could not enable image registration format; falling back to format defined in getSupportedVideoModes()");
      image_registration_mode = ONI_IMAGE_REGISTRATION_OFF;
      return setVideoMode(requested_mode);
    }
    return ONI_STATUS_NOT_SUPPORTED;
  }
#endif // 0
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


/* depth video modes reference

FREENECT_DEPTH_11BIT        = 0, //< 11 bit depth information in one uint16_t/pixel
FREENECT_DEPTH_10BIT        = 1, //< 10 bit depth information in one uint16_t/pixel
FREENECT_DEPTH_11BIT_PACKED = 2, //< 11 bit packed depth information
FREENECT_DEPTH_10BIT_PACKED = 3, //< 10 bit packed depth information
FREENECT_DEPTH_REGISTERED   = 4, //< processed depth data in mm, aligned to 640x480 RGB
FREENECT_DEPTH_MM           = 5, //< depth to each pixel in mm, but left unaligned to RGB image
FREENECT_DEPTH_DUMMY        = 2147483647, //< Dummy value to force enum to be 32 bits wide

ONI_PIXEL_FORMAT_DEPTH_1_MM = 100,
ONI_PIXEL_FORMAT_DEPTH_100_UM = 101,
ONI_PIXEL_FORMAT_SHIFT_9_2 = 102,
ONI_PIXEL_FORMAT_SHIFT_9_3 = 103,
*/
