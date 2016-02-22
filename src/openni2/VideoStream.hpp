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

#include <map>
#include <libfreenect2/libfreenect2.hpp>
#include <PS1080.h>
#include "Utility.hpp"
#include "Registration.hpp"

namespace Freenect2Driver
{
  class Device : public oni::driver::DeviceBase
  {
  public:
    virtual void start() = 0;
    virtual void stop() = 0;
  };

  class VideoStream : public oni::driver::StreamBase
  {
  private:
    unsigned int frame_id; // number each frame
    virtual void populateFrame(libfreenect2::Frame* lf2Frame, int srcX, int srcY, OniFrame* oniFrame, int tgtX, int tgtY, int width, int height) const = 0;

  protected:
    virtual OniSensorType getSensorType() const = 0;
    libfreenect2::Freenect2Device* device;
    Device* driver_dev;
    bool running; // buildFrame() does something iff true
    OniVideoMode video_mode;
    OniCropping cropping;
    bool mirroring;
    Freenect2Driver::Registration *reg;
    bool callPropertyChangedCallback;
    typedef std::map< OniVideoMode, int > VideoModeMap;
    virtual VideoModeMap getSupportedVideoModes() const = 0;

    OniStatus setVideoMode(OniVideoMode requested_mode);

    static void copyFrame(float* srcPix, int srcX, int srcY, int srcStride, uint16_t* dstPix, int dstX, int dstY, int dstStride, int width, int height, bool mirroring);
    void raisePropertyChanged(int propertyId, const void* data, int dataSize);

  public:
    VideoStream(Device* driver_dev, libfreenect2::Freenect2Device* device, Freenect2Driver::Registration *reg);

    OniSensorInfo getSensorInfo();

    void setPropertyChangedCallback(oni::driver::PropertyChangedCallback handler, void* pCookie);

    bool buildFrame(libfreenect2::Frame* lf2Frame);

    OniStatus start();
    void stop();

    // only add to property handlers if the property is generic to all children
    // otherwise, implement in child and call these in default case
    OniBool isPropertySupported(int propertyId);

    virtual OniStatus getProperty(int propertyId, void* data, int* pDataSize);
    virtual OniStatus setProperty(int propertyId, const void* data, int dataSize);

    /* todo : from StreamBase
    virtual OniStatus convertDepthToColorCoordinates(StreamBase* colorStream, int depthX, int depthY, OniDepthPixel depthZ, int* pColorX, int* pColorY) { return ONI_STATUS_NOT_SUPPORTED; }
    */
  };
}
