/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2014 individual OpenKinect contributors. See the CONTRIB file
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

#ifndef LIBFREENECT2_HPP_
#define LIBFREENECT2_HPP_

#include <libfreenect2/frame_listener.hpp>

namespace libfreenect2
{

class Freenect2Device
{
public:
  static const unsigned int VendorId = 0x045E;
  static const unsigned int ProductId = 0x02D8;
  static const unsigned int ProductIdPreview = 0x02C4;

  struct ColorCameraParams
  {
    float fx, fy, cx, cy;
  };

  struct IrCameraParams
  {
    float fx, fy, cx, cy, k1, k2, k3, p1, p2;
  };

  virtual ~Freenect2Device();

  virtual std::string getSerialNumber() = 0;
  virtual std::string getFirmwareVersion() = 0;

  virtual Freenect2Device::ColorCameraParams getColorCameraParams() = 0;
  virtual Freenect2Device::IrCameraParams getIrCameraParams() = 0;


  virtual void setColorFrameListener(libfreenect2::FrameListener* rgb_frame_listener) = 0;
  virtual void setIrAndDepthFrameListener(libfreenect2::FrameListener* ir_frame_listener) = 0;

  virtual void start() = 0;
  virtual void stop() = 0;
  virtual void close() = 0;
};

class Freenect2Impl;

class Freenect2
{
public:
  Freenect2(void *usb_context = 0);
  virtual ~Freenect2();

  int enumerateDevices();

  std::string getDeviceSerialNumber(int idx);
  std::string getDefaultDeviceSerialNumber();

  Freenect2Device *openDevice(int idx);
  Freenect2Device *openDevice(const std::string &serial);

  Freenect2Device *openDefaultDevice();
protected:
  Freenect2Device *openDevice(int idx, bool attempting_reset);
private:
  Freenect2Impl *impl_;
};

} /* namespace libfreenect2 */
#endif /* LIBFREENECT2_HPP_ */
