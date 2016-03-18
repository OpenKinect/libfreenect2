/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2014 Benn Snyder, 2015 individual OpenKinect contributors.
 * See the CONTRIB file for details.
 *
 * This code is licensed to you under the terms of the Apache License, version
 * 2.0, or, at your option, the terms of the GNU General Public License,
 * version 2.0. See the APACHE20 and GPL2 files for the text of the licenses,
 * or the following URLs:
 * http://www.apache.org/licenses/LICENSE-2.0
 * http://www.gnu.org/licenses/gpl-2.0.txt
 *
 * If you redistribute this file in source form, modified or unmodified, you
 * may:
 *   1) Leave this header intact and distribute it under the same terms,
 *      accompanying it with the APACHE20 and GPL20 files, or
 *   2) Delete the Apache 2.0 clause and accompany it with the GPL2 file, or
 *   3) Delete the GPL v2 clause and accompany it with the APACHE20 file
 * In all cases you must keep the copyright notice intact and include a copy
 * of the CONTRIB file.
 *
 * Binary distributions must follow the binary distribution requirements of
 * either License.
 */

#include <algorithm>
#include <libfreenect2/libfreenect2.hpp>
#include "PS1080.h"
#include "VideoStream.hpp"
#include "Utility.hpp"
#include "Registration.hpp"

struct ExtractKey
{
  template <typename T> typename T::first_type
  operator()(T pair) const
  {
    return pair.first;
  }
};

namespace Freenect2Driver
{
OniStatus VideoStream::setVideoMode(OniVideoMode requested_mode)
{
  VideoModeMap supported_video_modes = getSupportedVideoModes();
  VideoModeMap::const_iterator matched_mode_iter = supported_video_modes.find(requested_mode);
  if (matched_mode_iter == supported_video_modes.end())
    return ONI_STATUS_NOT_SUPPORTED;

  video_mode = requested_mode;
  return ONI_STATUS_OK;
}

void VideoStream::copyFrame(float* srcPix, int srcX, int srcY, int srcStride, uint16_t* dstPix, int dstX, int dstY, int dstStride, int width, int height, bool mirroring)
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
void VideoStream::raisePropertyChanged(int propertyId, const void* data, int dataSize) {
  if (callPropertyChangedCallback)
    StreamBase::raisePropertyChanged(propertyId, data, dataSize);
}

VideoStream::VideoStream(Device* drvdev, libfreenect2::Freenect2Device* device, Freenect2Driver::Registration *reg) :
  frame_id(1),
  device(device),
  driver_dev(drvdev),
  running(false),
  mirroring(false),
  reg(reg),
  callPropertyChangedCallback(false)
  {
    // joy of structs
    memset(&cropping, 0, sizeof(cropping));
    memset(&video_mode, 0, sizeof(video_mode));
  }
//~VideoStream() { stop();  }


OniSensorInfo VideoStream::getSensorInfo()
{
  VideoModeMap supported_modes = getSupportedVideoModes();
  OniVideoMode* modes = new OniVideoMode[supported_modes.size()];
  std::transform(supported_modes.begin(), supported_modes.end(), modes, ExtractKey());
  OniSensorInfo sensors = { getSensorType(), static_cast<int>(supported_modes.size()), modes };
  return sensors;
}

void VideoStream::setPropertyChangedCallback(oni::driver::PropertyChangedCallback handler, void* pCookie) {
  callPropertyChangedCallback = true;
  StreamBase::setPropertyChangedCallback(handler, pCookie);
}

bool VideoStream::buildFrame(libfreenect2::Frame* lf2Frame)
{
  if (!running)
    return false;

  OniFrame* oniFrame = getServices().acquireFrame();
  oniFrame->frameIndex = frame_id++;
  oniFrame->timestamp = lf2Frame->timestamp;
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
  //min is a macro with MSVC
  #undef min
  int width = std::min(oniFrame->width, (int)lf2Frame->width);
  int height = std::min(oniFrame->height, (int)lf2Frame->height);

  populateFrame(lf2Frame, oniFrame->cropOriginX, oniFrame->cropOriginY, oniFrame, 0, 0, width, height);
  raiseNewFrame(oniFrame);
  getServices().releaseFrame(oniFrame);

  return false;
}

// from StreamBase

OniStatus VideoStream::start()
{
  driver_dev->start();
  running = true;
  return ONI_STATUS_OK;
}
void VideoStream::stop()
{
  driver_dev->stop();
  running = false;
}

// only add to property handlers if the property is generic to all children
// otherwise, implement in child and call these in default case
OniBool VideoStream::isPropertySupported(int propertyId)
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

OniStatus VideoStream::getProperty(int propertyId, void* data, int* pDataSize)
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
OniStatus VideoStream::setProperty(int propertyId, const void* data, int dataSize)
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
      mirroring = !!*(static_cast<const OniBool*>(data));
      raisePropertyChanged(propertyId, data, dataSize);
      return ONI_STATUS_OK;
  }
}


/* todo : from StreamBase
virtual OniStatus convertDepthToColorCoordinates(StreamBase* colorStream, int depthX, int depthY, OniDepthPixel depthZ, int* pColorX, int* pColorY) { return ONI_STATUS_NOT_SUPPORTED; }
*/
}
