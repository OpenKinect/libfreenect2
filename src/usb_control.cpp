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

/** @file usb_control.cpp USB control using libusb. */

#include <libfreenect2/protocol/usb_control.h>
#include <libfreenect2/logging.h>

#include <stdint.h>

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

  int get_max_iso_packet_size(libusb_device *device, int configuration, int alternate_setting, int endpoint)
  {
    libusb_config_descriptor *config_desc;
    int r = LIBUSB_ERROR_NOT_FOUND;

    r = libusb_get_config_descriptor_by_value(device, configuration, &config_desc);

    if(r == LIBUSB_SUCCESS)
    {
      for(int interface_idx = 0; interface_idx < config_desc->bNumInterfaces; ++interface_idx)
      {
        const libusb_interface &interface = config_desc->interface[interface_idx];

        if(interface.num_altsetting > alternate_setting)
        {
          const libusb_interface_descriptor &interface_desc = interface.altsetting[alternate_setting];
          const libusb_endpoint_descriptor *endpoint_desc = 0;

          for(int endpoint_idx = 0; endpoint_idx < interface_desc.bNumEndpoints; ++endpoint_idx)
          {
            if(interface_desc.endpoint[endpoint_idx].bEndpointAddress == endpoint && (interface_desc.endpoint[endpoint_idx].bmAttributes & 0x3) == LIBUSB_TRANSFER_TYPE_ISOCHRONOUS)
            {
              endpoint_desc = interface_desc.endpoint + endpoint_idx;
              break;
            }
          }

          if(endpoint_desc != 0)
          {
            libusb_ss_endpoint_companion_descriptor *companion_desc;
            // ctx is only used for error reporting, libusb should better ask for a libusb_device anyway...
            r = libusb_get_ss_endpoint_companion_descriptor(NULL /* ctx */, endpoint_desc, &companion_desc);

            if(r != LIBUSB_SUCCESS) continue;

            r = companion_desc->wBytesPerInterval;

            libusb_free_ss_endpoint_companion_descriptor(companion_desc);
            break;
          }
        }
      }
    }
    libusb_free_config_descriptor(config_desc);

    return r;
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

#define CHECK_LIBUSB_RESULT(__CODE, __RESULT) if((__CODE = (__RESULT == LIBUSB_SUCCESS ? Success : Error)) == Error) LOG_ERROR
#define WRITE_LIBUSB_ERROR(__RESULT) libusb_error_name(__RESULT) << " " << libusb_strerror((libusb_error)__RESULT) << ". Try debugging with environment variable: export LIBUSB_DEBUG=3 ."

UsbControl::ResultCode UsbControl::setConfiguration()
{
  UsbControl::ResultCode code = Success;
  int desired_config_id = 1;
  int current_config_id = -1;
  int r;

  r = libusb_get_configuration(handle_, &current_config_id);
  CHECK_LIBUSB_RESULT(code, r) << "failed to get configuration! " << WRITE_LIBUSB_ERROR(r);

  if(code == Success)
  {
    if(current_config_id != desired_config_id)
    {
      r = libusb_set_configuration(handle_, desired_config_id);
      CHECK_LIBUSB_RESULT(code, r) << "failed to set configuration! " << WRITE_LIBUSB_ERROR(r);
    }
  }

  return code;
}

UsbControl::ResultCode UsbControl::claimInterfaces()
{
  UsbControl::ResultCode code = Success;
  int r;

  r = libusb_claim_interface(handle_, ControlAndRgbInterfaceId);
  CHECK_LIBUSB_RESULT(code, r) << "failed to claim interface with ControlAndRgbInterfaceId(="<< ControlAndRgbInterfaceId << ")! " << WRITE_LIBUSB_ERROR(r);

  if(code == Success)
  {
    r = libusb_claim_interface(handle_, IrInterfaceId);
    CHECK_LIBUSB_RESULT(code, r) << "failed to claim interface with IrInterfaceId(="<< IrInterfaceId << ")! " << WRITE_LIBUSB_ERROR(r);
  }

  return code;
}

UsbControl::ResultCode UsbControl::releaseInterfaces()
{
  UsbControl::ResultCode code = Success;
  int r;

  r = libusb_release_interface(handle_, ControlAndRgbInterfaceId);
  CHECK_LIBUSB_RESULT(code, r) << "failed to release interface with ControlAndRgbInterfaceId(="<< ControlAndRgbInterfaceId << ")! " << WRITE_LIBUSB_ERROR(r);

  if(code == Success)
  {
    r = libusb_release_interface(handle_, IrInterfaceId);
    CHECK_LIBUSB_RESULT(code, r) << "failed to release interface with IrInterfaceId(="<< IrInterfaceId << ")! " << WRITE_LIBUSB_ERROR(r);
  }

  return code;
}

UsbControl::ResultCode UsbControl::setIsochronousDelay()
{
  int r = libusb_ext::set_isochronous_delay(handle_, timeout_);

  UsbControl::ResultCode code;
  CHECK_LIBUSB_RESULT(code, r) << "failed to set isochronous delay! " << WRITE_LIBUSB_ERROR(r);
  return code;
}

UsbControl::ResultCode UsbControl::setPowerStateLatencies()
{
  int r = libusb_ext::set_sel(handle_, timeout_, 0x55, 0, 0x55, 0);

  UsbControl::ResultCode code;
  CHECK_LIBUSB_RESULT(code, r) << "failed to set power state latencies! " << WRITE_LIBUSB_ERROR(r);
  return code;
}

UsbControl::ResultCode UsbControl::enablePowerStates()
{
  UsbControl::ResultCode code;
  int r;

  r = libusb_ext::set_feature(handle_, timeout_, libusb_ext::U1_ENABLE);
  CHECK_LIBUSB_RESULT(code, r) << "failed to enable power states U1! " << WRITE_LIBUSB_ERROR(r);

  if(code == Success)
  {
    r = libusb_ext::set_feature(handle_, timeout_, libusb_ext::U2_ENABLE);
    CHECK_LIBUSB_RESULT(code, r) << "failed to enable power states U2! " << WRITE_LIBUSB_ERROR(r);
  }

  return code;
}

UsbControl::ResultCode UsbControl::setVideoTransferFunctionState(UsbControl::State state)
{
  bool suspend = state == Enabled ? false : true;
  int r = libusb_ext::set_feature_function_suspend(handle_, timeout_, suspend, suspend);

  UsbControl::ResultCode code;
  CHECK_LIBUSB_RESULT(code, r) << "failed to set video transfer function state! " << WRITE_LIBUSB_ERROR(r);
  return code;
}

UsbControl::ResultCode UsbControl::setIrInterfaceState(UsbControl::State state)
{
  int alternate_setting = state == Enabled ? 1 : 0;
  int r = libusb_set_interface_alt_setting(handle_, IrInterfaceId, alternate_setting);

  UsbControl::ResultCode code;
  CHECK_LIBUSB_RESULT(code, r) << "failed to set ir interface state! " << WRITE_LIBUSB_ERROR(r);
  return code;
}


UsbControl::ResultCode UsbControl::getIrMaxIsoPacketSize(int &size)
{
  size = 0;
  libusb_device *dev = libusb_get_device(handle_);
  int r = libusb_ext::get_max_iso_packet_size(dev, 1, 1, 0x84);

  if(r > LIBUSB_SUCCESS)
  {
    size = r;
    r = LIBUSB_SUCCESS;
  }

  UsbControl::ResultCode code;
  CHECK_LIBUSB_RESULT(code, r) << "failed to get max iso packet size! " << WRITE_LIBUSB_ERROR(r);
  return code;
}

} /* namespace protocol */
} /* namespace libfreenect2 */
