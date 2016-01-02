#include <string>
#include "ColorStream.hpp"

using namespace Freenect2Driver;

// from NUI library & converted to radians
const float ColorStream::DIAGONAL_FOV = 73.9 * (M_PI / 180);
const float ColorStream::HORIZONTAL_FOV = 62 * (M_PI / 180);
const float ColorStream::VERTICAL_FOV = 48.6 * (M_PI / 180);

ColorStream::ColorStream(libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg) : VideoStream(pDevice, reg)
{
  video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 1920, 1080, 30);
  setVideoMode(video_mode);
  pDevice->start();
}

// Add video modes here as you implement them
ColorStream::FreenectVideoModeMap ColorStream::getSupportedVideoModes()
{
  FreenectVideoModeMap modes;
  //                    pixelFormat, resolutionX, resolutionY, fps
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 512, 424, 30)] = 0;
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_RGB888, 1920, 1080, 30)] = 1;

  return modes;
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

void ColorStream::populateFrame(libfreenect2::Frame* srcFrame, int srcX, int srcY, OniFrame* dstFrame, int dstX, int dstY, int width, int height) const
{
  dstFrame->sensorType = sensor_type;
  dstFrame->stride = dstFrame->width * 3;

  // copy stream buffer from freenect
  switch (video_mode.pixelFormat)
  {
    default:
      LogError("Pixel format " + to_string(video_mode.pixelFormat) + " not supported by populateFrame()");
      return;

    case ONI_PIXEL_FORMAT_RGB888:
      if (reg->isEnabled()) {
        libfreenect2::Frame registered(512, 424, 4);

        reg->colorFrameRGB888(srcFrame, &registered);

        copyFrame(static_cast<uint8_t*>(registered.data), srcX, srcY, registered.width * registered.bytes_per_pixel, 
                  static_cast<uint8_t*>(dstFrame->data), dstX, dstY, dstFrame->stride, 
                  width, height, mirroring);
      } else {
        copyFrame(static_cast<uint8_t*>(srcFrame->data), srcX, srcY, srcFrame->width * srcFrame->bytes_per_pixel, 
                  static_cast<uint8_t*>(dstFrame->data), dstX, dstY, dstFrame->stride, 
                  width, height, mirroring);
      }
      return;
  }
}

void ColorStream::copyFrame(uint8_t* srcPix, int srcX, int srcY, int srcStride, uint8_t* dstPix, int dstX, int dstY, int dstStride, int width, int height, bool mirroring)
{
  srcPix += srcX + srcY * srcStride;
  dstPix += dstX + dstY * dstStride;

  for (int y = 0; y < height; y++) {
    uint8_t* dst = dstPix + y * dstStride;
    uint8_t* src = srcPix + y * srcStride;
    if (mirroring) {
      dst += dstStride - 1;
      for (int x = 0; x < srcStride; ++x)
      {
        if (x % 4 != 3)
        {
          *dst-- = *src++;
        }
        else
        {
          ++src;
        }
      }
    } else {
      for (int x = 0; x < dstStride-2; x += 3)
      {
        *dst++ = src[2];
        *dst++ = src[1];
        *dst++ = src[0];
        src += 4;
      }
    }
  }
}
