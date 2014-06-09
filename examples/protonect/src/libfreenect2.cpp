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

#include <libfreenect2/libfreenect2.hpp>
#include <libusb.h>

#include <libfreenect2/protocol/usb_control.h>
#include <libfreenect2/protocol/command.h>
#include <libfreenect2/protocol/command_transaction.h>

namespace libfreenect2
{
using namespace libfreenect2;
using namespace libfreenect2::protocol;

class Freenect2Impl
{
private:
  bool managed_usb_context_;
  libusb_context *usb_context_;
public:
  typedef std::vector<libusb_device *> UsbDeviceVector;
  bool has_device_enumeration_;
  UsbDeviceVector devices_;

  Freenect2Impl(void *usb_context) :
    managed_usb_context_(usb_context != 0),
    usb_context_(reinterpret_cast<libusb_context *>(usb_context)),
    has_device_enumeration_(false)
  {
    if(managed_usb_context_)
    {
      int r = libusb_init(&usb_context_);
      // TODO: error handling
    }
  }

  ~Freenect2Impl()
  {
    clearDeviceEnumeration();

    if(managed_usb_context_ && usb_context_ != 0)
    {
      libusb_exit(usb_context_);
      usb_context_ = 0;
    }
  }

  void clearDeviceEnumeration()
  {
    // free enumerated device pointers, this should not affect opened devices
    for(UsbDeviceVector::iterator it = devices_.begin(); it != devices_.end(); ++it)
    {
      libusb_unref_device(*it);
    }

    devices_.clear();
    has_device_enumeration_ = false;
  }

  void enumerateDevices()
  {
    libusb_device **device_list;
    int num_devices = libusb_get_device_list(usb_context_, &device_list);

    if(num_devices > 0)
    {
      for(int idx = 0; idx < num_devices; ++idx)
      {
        libusb_device *dev = device_list[idx];
        libusb_device_descriptor *dev_desc;

        int r = libusb_get_device_descriptor(dev, dev_desc);
        // TODO: error handling

        if(dev_desc->idVendor == Freenect2Device::VendorId && dev_desc->iProduct == Freenect2Device::ProductId)
        {
          // valid Kinect v2
          devices_.push_back(dev);
        }
        else
        {
          libusb_unref_device(dev);
        }
      }
    }

    libusb_free_device_list(device_list, 0);
    has_device_enumeration_ = true;
  }

  int getNumDevices()
  {
    if(!has_device_enumeration_)
    {
      enumerateDevices();
    }
    return devices_.size();
  }
};

class Freenect2DeviceImpl : public Freenect2Device
{
private:
  bool closed_, has_usb_interfaces_;

  Freenect2Impl *context_;
  libusb_device *usb_device_;
  libusb_device_handle *usb_device_handle_;

  UsbControl usb_control_;
  CommandTransaction command_tx_;
  int command_seq_;
public:
  Freenect2DeviceImpl(Freenect2Impl *context, libusb_device *usb_device, libusb_device_handle *usb_device_handle) :
    closed_(false),
    has_usb_interfaces_(false),
    context_(context),
    usb_device_(usb_device),
    usb_device_handle_(usb_device_handle),
    usb_control_(usb_device_handle_),
    command_tx_(usb_device_handle_, 0x81, 0x02),
    command_seq_(0)
  {
  }

  virtual ~Freenect2DeviceImpl()
  {
    close();
  }

  int nextCommandSeq()
  {
    return command_seq_++;
  }

  virtual std::string getSerialNumber()
  {
    throw std::exception();
  }

  bool initialize()
  {
    if(closed_) return false;

    if(usb_control_.setConfiguration() != UsbControl::Success) return false;
    if(!has_usb_interfaces_ && usb_control_.claimInterfaces() != UsbControl::Success) return false;
    has_usb_interfaces_ = true;

    if(usb_control_.setIsochronousDelay() != UsbControl::Success) return false;
    // TODO: always fails right now with error 6 - TRANSFER_OVERFLOW!
    //if(usb_control_.setPowerStateLatencies() != UsbControl::Success) return false;
    if(usb_control_.setIrInterfaceState(UsbControl::Disabled) != UsbControl::Success) return false;
    if(usb_control_.enablePowerStates() != UsbControl::Success) return false;
    if(usb_control_.setVideoTransferFunctionState(UsbControl::Disabled) != UsbControl::Success) return false;

    return true;
  }

  virtual void close()
  {
    if(closed_) return;

    usb_control_.setIrInterfaceState(UsbControl::Disabled);

    CommandTransaction::Result result;
    command_tx_.execute(Unknown0x0ACommand(nextCommandSeq()), result);
    command_tx_.execute(SetStreamDisabledCommand(nextCommandSeq()), result);

    usb_control_.setVideoTransferFunctionState(UsbControl::Disabled);

    if(has_usb_interfaces_)
    {
      usb_control_.releaseInterfaces();
      has_usb_interfaces_ = false;
    }

    libusb_close(usb_device_handle_);
    usb_device_handle_ = 0;
    usb_device_ = 0;

    closed_ = true;
  }
};


Freenect2::Freenect2(void *usb_context) :
    impl_(new Freenect2Impl(usb_context))
{
}

Freenect2::~Freenect2()
{
  delete impl_;
}

int Freenect2::enumerateDevices()
{
  impl_->clearDeviceEnumeration();
  return impl_->getNumDevices();
}

std::string Freenect2::getDeviceSerialNumber(int idx)
{
  throw std::exception();
}

std::string Freenect2::getDefaultDeviceSerialNumber()
{
  return getDeviceSerialNumber(0);
}

Freenect2Device *Freenect2::openDevice(int idx)
{
  int num_devices = impl_->getNumDevices();

  if(idx < num_devices)
  {
    libusb_device *dev = impl_->devices_[idx];
    libusb_device_handle *dev_handle;

    int r = libusb_open(dev, &dev_handle);
    // TODO: error handling

    Freenect2DeviceImpl *device = new Freenect2DeviceImpl(impl_, dev, dev_handle);
    if(device->initialize())
    {
      return device;
    }
    else
    {
      delete device;

      // TODO: error handling
      return 0;
    }
  }
  else
  {
    // TODO: error handling
    return 0;
  }
}

Freenect2Device *Freenect2::openDevice(const std::string &serial)
{
  throw std::exception();
}

Freenect2Device *Freenect2::openDefaultDevice()
{
  return openDevice(0);
}

} /* namespace libfreenect2 */
