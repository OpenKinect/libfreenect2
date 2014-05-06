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

#include <libfreenect2/protocol/usb_control.h>

#include <stdint.h>
#include <iostream>

namespace libfreenect2
{
namespace protocol
{

namespace libusb_ext
{
  enum DeviceFeatureSelector
  {
    U1_ENABLE = 48,
    U2_ENABLE = 49
  };

  enum InterfaceFeatureSelector
  {
    FUNCTION_SUSPEND = 0,
  };

  template<typename T>
  struct FeatureSelectorRecipient
  {
    static uint8_t get();
  };

  template<>
  struct FeatureSelectorRecipient<DeviceFeatureSelector>
  {
    static uint8_t get() { return LIBUSB_RECIPIENT_DEVICE; };
  };

  template<>
  struct FeatureSelectorRecipient<InterfaceFeatureSelector>
  {
    static uint8_t get() { return LIBUSB_RECIPIENT_INTERFACE; };
  };

  int set_isochronous_delay(libusb_device_handle *handle, int timeout)
  {
    // for details see USB 3.1 r1 spec section 9.4.11

    uint8_t bmRequestType = LIBUSB_RECIPIENT_DEVICE;
    uint8_t bRequest = LIBUSB_SET_ISOCH_DELAY;
    // if no super speed hubs in between, then it is equal to tTPTransmissionDelay(=40ns)
    uint16_t wValue  = 40; // 40 nanoseconds
    uint16_t wIndex  = 0;
    uint16_t wLength = 0;
    uint8_t *data    = 0;

    return libusb_control_transfer(handle, bmRequestType, bRequest, wValue, wIndex, data, wLength, timeout);
  }

  int set_sel(libusb_device_handle *handle, int timeout, uint8_t u1sel, uint8_t u1pel, uint8_t u2sel, uint8_t u2pel)
  {
    // for details see USB 3.1 r1 spec section 9.4.12

    uint8_t bmRequestType = LIBUSB_RECIPIENT_DEVICE;
    uint8_t bRequest = LIBUSB_REQUEST_SET_SEL;
    uint16_t wValue  = 0;
    uint16_t wIndex  = 0;
    uint16_t wLength = 6;
    unsigned char data[6]   = { 0x55, 0, 0x55, 0, 0, 0 };

    return libusb_control_transfer(handle, bmRequestType, bRequest, wValue, wIndex, data, wLength, timeout);
  }

  template<typename TFeatureSelector>
  int set_feature(libusb_device_handle *handle, int timeout, TFeatureSelector feature_selector)
  {
    // for details see USB 3.1 r1 spec section 9.4.9

    uint8_t bmRequestType = FeatureSelectorRecipient<TFeatureSelector>::get();
    uint8_t bRequest = LIBUSB_REQUEST_SET_FEATURE;
    uint16_t wValue  = static_cast<uint16_t>(feature_selector);
    uint16_t wIndex  = 0;
    uint16_t wLength = 0;
    uint8_t *data    = 0;

    return libusb_control_transfer(handle, bmRequestType, bRequest, wValue, wIndex, data, wLength, timeout);
  }

  int set_feature_function_suspend(libusb_device_handle *handle, int timeout, bool low_power_suspend, bool function_remote_wake)
  {
    uint8_t suspend_options = 0;
    suspend_options |= low_power_suspend ? 1 : 0;
    suspend_options |= function_remote_wake ? 2 : 0;

    // for details see USB 3.1 r1 spec section 9.4.9

    InterfaceFeatureSelector feature_selector = FUNCTION_SUSPEND;
    uint8_t bmRequestType = FeatureSelectorRecipient<InterfaceFeatureSelector>::get();
    uint8_t bRequest = LIBUSB_REQUEST_SET_FEATURE;
    uint16_t wValue  = static_cast<uint16_t>(feature_selector);
    uint16_t wIndex  = suspend_options << 8 | 0;
    uint16_t wLength = 0;
    uint8_t *data    = 0;

    return libusb_control_transfer(handle, bmRequestType, bRequest, wValue, wIndex, data, wLength, timeout);
  }
}

UsbControl::UsbControl(libusb_device_handle *handle) :
    handle_(handle),
    timeout_(1000)
{
}

UsbControl::~UsbControl()
{
}

static UsbControl::ResultCode checkLibusbResult(const char* method, int r)
{
  if(r != LIBUSB_SUCCESS)
  {
    std::cerr << "[UsbControl::" << method << "] failed! libusb error " << r << ": " << libusb_error_name(r) << std::endl;
    return UsbControl::Error;
  }
  else
  {
    return UsbControl::Success;
  }
}

UsbControl::ResultCode UsbControl::setIsochronousDelay()
{
  int r = libusb_ext::set_isochronous_delay(handle_, timeout_);

  return checkLibusbResult("setIsochronousDelay", r);
}

UsbControl::ResultCode UsbControl::setPowerStateLatencies()
{
  int r = libusb_ext::set_sel(handle_, timeout_, 0x55, 0, 0x55, 0);

  return checkLibusbResult("setPowerStateLatencies", r);
}

UsbControl::ResultCode UsbControl::enablePowerStates()
{
  UsbControl::ResultCode code;
  int r;

  r = libusb_ext::set_feature(handle_, timeout_, libusb_ext::U1_ENABLE);
  code = checkLibusbResult("enablePowerStates(U1)", r);

  if(code == Success)
  {
    r = libusb_ext::set_feature(handle_, timeout_, libusb_ext::U2_ENABLE);
    code = checkLibusbResult("enablePowerStates(U2)", r);
  }

  return code;
}

UsbControl::ResultCode UsbControl::setVideoTransferFunctionState(UsbControl::State state)
{
  bool suspend = state == Enabled ? false : true;
  int r = libusb_ext::set_feature_function_suspend(handle_, timeout_, suspend, suspend);

  return checkLibusbResult("setVideoTransferFunctionState", r);
}

UsbControl::ResultCode UsbControl::setIrInterfaceState(UsbControl::State state)
{
  int alternate_setting = state == Enabled ? 1 : 0;
  int r = libusb_set_interface_alt_setting(handle_, IrInterfaceId, alternate_setting);

  return checkLibusbResult("setIrInterfaceState", r);
}

} /* namespace protocol */
} /* namespace libfreenect2 */
