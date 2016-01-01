/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2015 individual OpenKinect contributors. See the CONTRIB file
 * for details.
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
}

// Add video modes here as you implement them
// Note: if image_registration_mode == ONI_IMAGE_REGISTRATION_DEPTH_TO_COLOR,
// setVideoFormat() will try FREENECT_DEPTH_REGISTERED first then fall back on what is set here.
DepthStream::VideoModeMap DepthStream::getSupportedVideoModes() const
{
  VideoModeMap modes;
  //                      pixelFormat, resolutionX, resolutionY, fps
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 640, 480, 30)] = 0;
  modes[makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, 512, 424, 30)] = 1;

  return modes;
}

void DepthStream::populateFrame(libfreenect2::Frame* srcFrame, int srcX, int srcY, OniFrame* dstFrame, int dstX, int dstY, int width, int height) const
{
  dstFrame->sensorType = getSensorType();
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
