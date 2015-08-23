/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2011 individual OpenKinect contributors. See the CONTRIB file
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

#ifndef USB_CONTROL_H_
#define USB_CONTROL_H_

#include <libusb.h>

namespace libfreenect2
{
namespace protocol
{
/**
 * The Kinect2 device defines 2 USB interface associations (=device functions)
 *
 * 1. Interface Association: video transfer
 * 2. Interface Association: audio transfer
 *
 * The first interface association is enabled/disabled by setting the feature FUNCTION_SUSPEND.
 *
 * The following describes the interfaces in the video transfer interface association.
 *
 * It contains 2 USB interfaces
 *
 * 1. Interface: control communication, RGB transfer
 * 2. Interface: IR transfer
 *
 * Each interface has different endpoints
 *
 * 1. Interface
 *  - 0x81 (bulk) control communication IN
 *  - 0x02 (bulk) control communication OUT
 *  - 0x83 (bulk) RGB transfer IN
 *  - 0x82 (interrupt) unknown IN
 *
 * 2. Interface
 *  - 0x84 (isochronous) IR transfer IN
 *
 * The 2. interface can be enabled/disabled by changing its alternate setting to 1/0
 */
class UsbControl
{
public:
  UsbControl(libusb_device_handle *handle);
  virtual ~UsbControl();

  enum State
  {
    Enabled,
    Disabled
  };

  enum ResultCode
  {
    Success,
    Error
  };

  ResultCode getIrMaxIsoPacketSize(int &size);

  ResultCode setConfiguration();
  ResultCode claimInterfaces();
  ResultCode releaseInterfaces();

  ResultCode setIsochronousDelay();
  ResultCode setPowerStateLatencies();
  ResultCode enablePowerStates();

  // enable/suspend 1. Interface Association
  ResultCode setVideoTransferFunctionState(State state);

  // enable/disable 2. Interface using alternate setting
  ResultCode setIrInterfaceState(State state);

private:
  static const int ControlAndRgbInterfaceId = 0;
  static const int IrInterfaceId = 1;

  libusb_device_handle* handle_;
  int timeout_;
};

} /* namespace protocol */
} /* namespace libfreenect2 */
#endif /* USB_CONTROL_H_ */
