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

#pragma once

#include <libfreenect2/libfreenect2.hpp>
#include <Driver/OniDriverAPI.h>
#include "VideoStream.hpp"


namespace Freenect2Driver
{
  class DepthStream : public VideoStream
  {
  public:
    // from NUI library and converted to radians
    static const float DIAGONAL_FOV;
    static const float HORIZONTAL_FOV;
    static const float VERTICAL_FOV;
    // from DepthKinectStream.cpp
    static const int MAX_VALUE = 10000;
    static const unsigned long long GAIN_VAL = 42;
    static const unsigned long long CONST_SHIFT_VAL = 200;
    static const unsigned long long MAX_SHIFT_VAL = 2047;
    static const unsigned long long PARAM_COEFF_VAL = 4;
    static const unsigned long long SHIFT_SCALE_VAL = 10;
    static const unsigned long long ZERO_PLANE_DISTANCE_VAL = 120;
    static const double ZERO_PLANE_PIXEL_SIZE_VAL;
    static const double EMITTER_DCMOS_DISTANCE_VAL;

  private:
    OniSensorType getSensorType() const;
    OniImageRegistrationMode image_registration_mode;
    VideoModeMap getSupportedVideoModes() const;
    void populateFrame(libfreenect2::Frame* srcFrame, int srcX, int srcY, OniFrame* dstFrame, int dstX, int dstY, int width, int height) const;

  public:
    DepthStream(Device* driver_dev, libfreenect2::Freenect2Device* pDevice, Freenect2Driver::Registration *reg);
    //~DepthStream() { }

    OniImageRegistrationMode getImageRegistrationMode() const;
    OniStatus setImageRegistrationMode(OniImageRegistrationMode mode);

    // from StreamBase
    OniBool isImageRegistrationModeSupported(OniImageRegistrationMode mode);
    OniBool isPropertySupported(int propertyId);
    OniStatus getProperty(int propertyId, void* data, int* pDataSize);
  };
}
