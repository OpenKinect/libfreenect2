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

#include <iostream>
#include <vector>
#include <algorithm>
#include <libusb.h>

#include <libfreenect2/libfreenect2.hpp>

#include <libfreenect2/usb/event_loop.h>
#include <libfreenect2/usb/transfer_pool.h>
#include <libfreenect2/rgb_packet_stream_parser.h>
#include <libfreenect2/rgb_packet_processor.h>
#include <libfreenect2/depth_packet_stream_parser.h>
#include <libfreenect2/protocol/usb_control.h>
#include <libfreenect2/protocol/command.h>
#include <libfreenect2/protocol/response.h>
#include <libfreenect2/protocol/command_transaction.h>

namespace libfreenect2
{
using namespace libfreenect2;
using namespace libfreenect2::usb;
using namespace libfreenect2::protocol;

class Freenect2DeviceImpl : public Freenect2Device
{
private:
  enum State
  {
    Created,
    Open,
    Streaming,
    Closed
  };

  State state_;
  bool has_usb_interfaces_;

  Freenect2Impl *context_;
  libusb_device *usb_device_;
  libusb_device_handle *usb_device_handle_;

  BulkTransferPool rgb_transfer_pool_;
  IsoTransferPool ir_transfer_pool_;

  UsbControl usb_control_;
  CommandTransaction command_tx_;
  int command_seq_;

  TurboJpegRgbPacketProcessor rgb_packet_processor_;
  OpenCLDepthPacketProcessor depth_packet_processor_;

  RgbPacketStreamParser rgb_packet_parser_;
  DepthPacketStreamParser depth_packet_parser_;

  std::string serial_, firmware_;
  Freenect2Device::IrCameraParams ir_camera_params_;
  Freenect2Device::ColorCameraParams rgb_camera_params_;
public:
  Freenect2DeviceImpl(Freenect2Impl *context, libusb_device *usb_device, libusb_device_handle *usb_device_handle, const std::string &serial);
  virtual ~Freenect2DeviceImpl();

  bool isSameUsbDevice(libusb_device* other);

  virtual std::string getSerialNumber();
  virtual std::string getFirmwareVersion();

  virtual Freenect2Device::ColorCameraParams getColorCameraParams();
  virtual Freenect2Device::IrCameraParams getIrCameraParams();

  int nextCommandSeq();

  bool open();

  virtual void setColorFrameListener(libfreenect2::FrameListener* rgb_frame_listener);
  virtual void setIrAndDepthFrameListener(libfreenect2::FrameListener* ir_frame_listener);
  virtual void start();
  virtual void stop();
  virtual void close();
};

struct PrintBusAndDevice
{
  libusb_device *dev_;

  PrintBusAndDevice(libusb_device *dev) : dev_(dev) {}
};

std::ostream &operator<<(std::ostream &out, const PrintBusAndDevice& dev)
{
  out << "@" << int(libusb_get_bus_number(dev.dev_)) << ":" << int(libusb_get_port_number(dev.dev_));
  return out;
}

class Freenect2Impl
{
private:
  bool managed_usb_context_;
  libusb_context *usb_context_;
  EventLoop usb_event_loop_;
public:
  struct UsbDeviceWithSerial
  {
    libusb_device *dev;
    std::string serial;
  };
  typedef std::vector<UsbDeviceWithSerial> UsbDeviceVector;
  typedef std::vector<Freenect2DeviceImpl *> DeviceVector;

  bool has_device_enumeration_;
  UsbDeviceVector enumerated_devices_;
  DeviceVector devices_;

  Freenect2Impl(void *usb_context) :
    managed_usb_context_(usb_context == 0),
    usb_context_(reinterpret_cast<libusb_context *>(usb_context)),
    has_device_enumeration_(false)
  {
    if(managed_usb_context_)
    {
      int r = libusb_init(&usb_context_);
      // TODO: error handling
      if(r != 0)
      {
        std::cout << "[Freenect2Impl] failed to create usb context!" << std::endl;
      }
    }

    usb_event_loop_.start(usb_context_);
  }

  ~Freenect2Impl()
  {
    clearDevices();
    clearDeviceEnumeration();

    usb_event_loop_.stop();

    if(managed_usb_context_ && usb_context_ != 0)
    {
      libusb_exit(usb_context_);
      usb_context_ = 0;
    }
  }

  void addDevice(Freenect2DeviceImpl *device)
  {
    devices_.push_back(device);
  }

  void removeDevice(Freenect2DeviceImpl *device)
  {
    DeviceVector::iterator it = std::find(devices_.begin(), devices_.end(), device);

    if(it != devices_.end())
    {
      devices_.erase(it);
    }
    else
    {
      std::cout << "[Freenect2Impl] tried to remove device, which is not in the internal device list!" << std::endl;
    }
  }

  bool tryGetDevice(libusb_device *usb_device, Freenect2DeviceImpl **device)
  {
    for(DeviceVector::iterator it = devices_.begin(); it != devices_.end(); ++it)
    {
      if((*it)->isSameUsbDevice(usb_device))
      {
        *device = *it;
        return true;
      }
    }

    return false;
  }

  void clearDevices()
  {
    DeviceVector devices(devices_.begin(), devices_.end());

    for(DeviceVector::iterator it = devices.begin(); it != devices.end(); ++it)
    {
      delete (*it);
    }

    if(!devices_.empty())
    {
      std::cout << "[Freenect2Impl] after deleting all devices the internal device list should be empty!" << std::endl;
    }
  }

  void clearDeviceEnumeration()
  {
    // free enumerated device pointers, this should not affect opened devices
    for(UsbDeviceVector::iterator it = enumerated_devices_.begin(); it != enumerated_devices_.end(); ++it)
    {
      libusb_unref_device(it->dev);
    }

    enumerated_devices_.clear();
    has_device_enumeration_ = false;
  }

  void enumerateDevices()
  {
    std::cout << "[Freenect2Impl] enumerating devices..." << std::endl;
    libusb_device **device_list;
    int num_devices = libusb_get_device_list(usb_context_, &device_list);

    std::cout << "[Freenect2Impl] " << num_devices << " usb devices connected" << std::endl;

    if(num_devices > 0)
    {
      for(int idx = 0; idx < num_devices; ++idx)
      {
        libusb_device *dev = device_list[idx];
        libusb_device_descriptor dev_desc;

        int r = libusb_get_device_descriptor(dev, &dev_desc); // this is always successful

        if(dev_desc.idVendor == Freenect2Device::VendorId && (dev_desc.idProduct == Freenect2Device::ProductId || dev_desc.idProduct == Freenect2Device::ProductIdPreview))
        {
          Freenect2DeviceImpl *freenect2_dev;

          // prevent error if device is already open
          if(tryGetDevice(dev, &freenect2_dev))
          {
            UsbDeviceWithSerial dev_with_serial;
            dev_with_serial.dev = dev;
            dev_with_serial.serial = freenect2_dev->getSerialNumber();

            enumerated_devices_.push_back(dev_with_serial);
            continue;
          }
          else
          {
            libusb_device_handle *dev_handle;
            r = libusb_open(dev, &dev_handle);

            if(r == LIBUSB_SUCCESS)
            {
              unsigned char buffer[1024];
              r = libusb_get_string_descriptor_ascii(dev_handle, dev_desc.iSerialNumber, buffer, sizeof(buffer));

              if(r > LIBUSB_SUCCESS)
              {
                UsbDeviceWithSerial dev_with_serial;
                dev_with_serial.dev = dev;
                dev_with_serial.serial = std::string(reinterpret_cast<char *>(buffer), size_t(r));

                std::cout << "[Freenect2Impl] found valid Kinect v2 " << PrintBusAndDevice(dev) << " with serial " << dev_with_serial.serial << std::endl;
                // valid Kinect v2
                enumerated_devices_.push_back(dev_with_serial);
                continue;
              }
              else
              {
                std::cout << "[Freenect2Impl] failed to get serial number of Kinect v2 " << PrintBusAndDevice(dev) << "!" << std::endl;
              }

              libusb_close(dev_handle);
            }
            else
            {
              std::cout << "[Freenect2Impl] failed to open Kinect v2 " << PrintBusAndDevice(dev) << "!" << std::endl;
            }
          }
        }
        libusb_unref_device(dev);
      }
    }

    libusb_free_device_list(device_list, 0);
    has_device_enumeration_ = true;

    std::cout << "[Freenect2Impl] found " << enumerated_devices_.size() << " devices" << std::endl;
  }

  int getNumDevices()
  {
    if(!has_device_enumeration_)
    {
      enumerateDevices();
    }
    return enumerated_devices_.size();
  }
};


Freenect2Device::~Freenect2Device()
{
}

Freenect2DeviceImpl::Freenect2DeviceImpl(Freenect2Impl *context, libusb_device *usb_device, libusb_device_handle *usb_device_handle, const std::string &serial) :
  state_(Created),
  has_usb_interfaces_(false),
  context_(context),
  usb_device_(usb_device),
  usb_device_handle_(usb_device_handle),
  rgb_transfer_pool_(usb_device_handle, 0x83),
  ir_transfer_pool_(usb_device_handle, 0x84),
  usb_control_(usb_device_handle_),
  command_tx_(usb_device_handle_, 0x81, 0x02),
  command_seq_(0),
  rgb_packet_processor_(),
  depth_packet_processor_(),
  rgb_packet_parser_(&rgb_packet_processor_),
  depth_packet_parser_(&depth_packet_processor_),
  serial_(serial),
  firmware_("<unknown>")
{
  rgb_transfer_pool_.setCallback(&rgb_packet_parser_);
  ir_transfer_pool_.setCallback(&depth_packet_parser_);

  depth_packet_processor_.load11To16LutFromFile("11to16.bin");
  depth_packet_processor_.loadXTableFromFile("xTable.bin");
  depth_packet_processor_.loadZTableFromFile("zTable.bin");
}

Freenect2DeviceImpl::~Freenect2DeviceImpl()
{
  close();
  context_->removeDevice(this);
}

int Freenect2DeviceImpl::nextCommandSeq()
{
  return command_seq_++;
}

bool Freenect2DeviceImpl::isSameUsbDevice(libusb_device* other)
{
  bool result = false;

  if(state_ != Closed && usb_device_ != 0)
  {
    unsigned char bus = libusb_get_bus_number(usb_device_);
    unsigned char port = libusb_get_port_number(usb_device_);

    unsigned char other_bus = libusb_get_bus_number(other);
    unsigned char other_port = libusb_get_port_number(other);

    result = (bus == other_bus) && (port == other_port);
  }

  return result;
}

std::string Freenect2DeviceImpl::getSerialNumber()
{
  return serial_;
}

std::string Freenect2DeviceImpl::getFirmwareVersion()
{
  return firmware_;
}

Freenect2Device::ColorCameraParams Freenect2DeviceImpl::getColorCameraParams()
{
  return rgb_camera_params_;
}


Freenect2Device::IrCameraParams Freenect2DeviceImpl::getIrCameraParams()
{
  return ir_camera_params_;
}
void Freenect2DeviceImpl::setColorFrameListener(libfreenect2::FrameListener* rgb_frame_listener)
{
  // TODO: should only be possible, if not started
  rgb_packet_processor_.setFrameListener(rgb_frame_listener);
}

void Freenect2DeviceImpl::setIrAndDepthFrameListener(libfreenect2::FrameListener* ir_frame_listener)
{
  // TODO: should only be possible, if not started
  depth_packet_processor_.setFrameListener(ir_frame_listener);
}

bool Freenect2DeviceImpl::open()
{
  std::cout << "[Freenect2DeviceImpl] opening..." << std::endl;

  if(state_ != Created) return false;

  if(usb_control_.setConfiguration() != UsbControl::Success) return false;
  if(!has_usb_interfaces_ && usb_control_.claimInterfaces() != UsbControl::Success) return false;
  has_usb_interfaces_ = true;

  if(usb_control_.setIsochronousDelay() != UsbControl::Success) return false;
  // TODO: always fails right now with error 6 - TRANSFER_OVERFLOW!
  //if(usb_control_.setPowerStateLatencies() != UsbControl::Success) return false;
  if(usb_control_.setIrInterfaceState(UsbControl::Disabled) != UsbControl::Success) return false;
  if(usb_control_.enablePowerStates() != UsbControl::Success) return false;
  if(usb_control_.setVideoTransferFunctionState(UsbControl::Disabled) != UsbControl::Success) return false;

  int max_iso_packet_size;
  if(usb_control_.getIrMaxIsoPacketSize(max_iso_packet_size) != UsbControl::Success) return false;

  if(max_iso_packet_size < 0x8400)
  {
    std::cout << "[Freenect2DeviceImpl] max iso packet size for endpoint 0x84 too small! (expected: " << 0x8400 << " got: " << max_iso_packet_size << ")" << std::endl;
    return false;
  }

  rgb_transfer_pool_.allocate(50, 0x4000);
  ir_transfer_pool_.allocate(80, 8, max_iso_packet_size);

  state_ = Open;

  std::cout << "[Freenect2DeviceImpl] opened" << std::endl;

  return true;
}

void Freenect2DeviceImpl::start()
{
  std::cout << "[Freenect2DeviceImpl] starting..." << std::endl;
  if(state_ != Open) return;

  CommandTransaction::Result serial_result, firmware_result, result;

  usb_control_.setVideoTransferFunctionState(UsbControl::Enabled);

  command_tx_.execute(ReadFirmwareVersionsCommand(nextCommandSeq()), firmware_result);
  firmware_ = FirmwareVersionResponse(firmware_result.data, firmware_result.length).toString();

  command_tx_.execute(ReadData0x14Command(nextCommandSeq()), result);
  std::cout << "[Freenect2DeviceImpl] ReadData0x14 response" << std::endl;
  std::cout << GenericResponse(result.data, result.length).toString() << std::endl;

  command_tx_.execute(ReadSerialNumberCommand(nextCommandSeq()), serial_result);
  std::string new_serial = SerialNumberResponse(serial_result.data, serial_result.length).toString();

  if(serial_ != new_serial)
  {
    std::cout << "[Freenect2DeviceImpl] serial number reported by libusb " << serial_ << " differs from serial number " << new_serial << " in device protocol! " << std::endl;
  }

  command_tx_.execute(ReadDepthCameraParametersCommand(nextCommandSeq()), result);
  DepthCameraParamsResponse *ir_p = reinterpret_cast<DepthCameraParamsResponse *>(result.data);

  ir_camera_params_.fx = ir_p->fx;
  ir_camera_params_.fy = ir_p->fy;
  ir_camera_params_.cx = ir_p->cx;
  ir_camera_params_.cy = ir_p->cy;
  ir_camera_params_.k1 = ir_p->k1;
  ir_camera_params_.k2 = ir_p->k2;
  ir_camera_params_.k3 = ir_p->k3;
  ir_camera_params_.p1 = ir_p->p1;
  ir_camera_params_.p2 = ir_p->p2;

  command_tx_.execute(ReadP0TablesCommand(nextCommandSeq()), result);
  depth_packet_processor_.loadP0TablesFromCommandResponse(result.data, result.length);

  command_tx_.execute(ReadRgbCameraParametersCommand(nextCommandSeq()), result);
  RgbCameraParamsResponse *rgb_p = reinterpret_cast<RgbCameraParamsResponse *>(result.data);

  rgb_camera_params_.fx = rgb_p->intrinsics[0];
  rgb_camera_params_.fy = rgb_p->intrinsics[0];
  rgb_camera_params_.cx = rgb_p->intrinsics[1];
  rgb_camera_params_.cy = rgb_p->intrinsics[2];

  command_tx_.execute(ReadStatus0x090000Command(nextCommandSeq()), result);
  std::cout << "[Freenect2DeviceImpl] ReadStatus0x090000 response" << std::endl;
  std::cout << GenericResponse(result.data, result.length).toString() << std::endl;

  command_tx_.execute(InitStreamsCommand(nextCommandSeq()), result);

  usb_control_.setIrInterfaceState(UsbControl::Enabled);

  command_tx_.execute(ReadStatus0x090000Command(nextCommandSeq()), result);
  std::cout << "[Freenect2DeviceImpl] ReadStatus0x090000 response" << std::endl;
  std::cout << GenericResponse(result.data, result.length).toString() << std::endl;

  command_tx_.execute(SetStreamEnabledCommand(nextCommandSeq()), result);

  //command_tx_.execute(Unknown0x47Command(nextCommandSeq()), result);
  //command_tx_.execute(Unknown0x46Command(nextCommandSeq()), result);
/*
  command_tx_.execute(SetModeEnabledCommand(nextCommandSeq()), result);
  command_tx_.execute(SetModeDisabledCommand(nextCommandSeq()), result);

  usb_control_.setIrInterfaceState(UsbControl::Enabled);

  command_tx_.execute(SetModeEnabledWith0x00640064Command(nextCommandSeq()), result);
  command_tx_.execute(ReadData0x26Command(nextCommandSeq()), result);
  command_tx_.execute(ReadStatus0x100007Command(nextCommandSeq()), result);
  command_tx_.execute(SetModeEnabledWith0x00500050Command(nextCommandSeq()), result);
  command_tx_.execute(ReadData0x26Command(nextCommandSeq()), result);
  command_tx_.execute(ReadStatus0x100007Command(nextCommandSeq()), result);
  command_tx_.execute(ReadData0x26Command(nextCommandSeq()), result);
  command_tx_.execute(ReadData0x26Command(nextCommandSeq()), result);
*/
  std::cout << "[Freenect2DeviceImpl] enabling usb transfer submission..." << std::endl;
  rgb_transfer_pool_.enableSubmission();
  ir_transfer_pool_.enableSubmission();

  std::cout << "[Freenect2DeviceImpl] submitting usb transfers..." << std::endl;
  rgb_transfer_pool_.submit(20);
  ir_transfer_pool_.submit(60);

  state_ = Streaming;
  std::cout << "[Freenect2DeviceImpl] started" << std::endl;
}

void Freenect2DeviceImpl::stop()
{
  std::cout << "[Freenect2DeviceImpl] stopping..." << std::endl;

  if(state_ != Streaming)
  {
    std::cout << "[Freenect2DeviceImpl] already stopped, doing nothing" << std::endl;
    return;
  }

  std::cout << "[Freenect2DeviceImpl] disabling usb transfer submission..." << std::endl;
  rgb_transfer_pool_.disableSubmission();
  ir_transfer_pool_.disableSubmission();

  std::cout << "[Freenect2DeviceImpl] canceling usb transfers..." << std::endl;
  rgb_transfer_pool_.cancel();
  ir_transfer_pool_.cancel();

  // wait for completion of transfer cancelation
  // TODO: better implementation
  libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(1500));

  usb_control_.setIrInterfaceState(UsbControl::Disabled);

  CommandTransaction::Result result;
  command_tx_.execute(Unknown0x0ACommand(nextCommandSeq()), result);
  command_tx_.execute(SetStreamDisabledCommand(nextCommandSeq()), result);

  usb_control_.setVideoTransferFunctionState(UsbControl::Disabled);

  state_ = Open;
  std::cout << "[Freenect2DeviceImpl] stopped" << std::endl;
}

void Freenect2DeviceImpl::close()
{
  std::cout << "[Freenect2DeviceImpl] closing..." << std::endl;

  if(state_ == Closed)
  {
    std::cout << "[Freenect2DeviceImpl] already closed, doing nothing" << std::endl;
    return;
  }

  if(state_ == Streaming)
  {
    stop();
  }

  rgb_packet_processor_.setFrameListener(0);
  depth_packet_processor_.setFrameListener(0);

  if(has_usb_interfaces_)
  {
    std::cout << "[Freenect2DeviceImpl] releasing usb interfaces..." << std::endl;

    usb_control_.releaseInterfaces();
    has_usb_interfaces_ = false;
  }

  std::cout << "[Freenect2DeviceImpl] deallocating usb transfer pools..." << std::endl;
  rgb_transfer_pool_.deallocate();
  ir_transfer_pool_.deallocate();

  std::cout << "[Freenect2DeviceImpl] closing usb device..." << std::endl;

  libusb_close(usb_device_handle_);
  usb_device_handle_ = 0;
  usb_device_ = 0;

  state_ = Closed;
  std::cout << "[Freenect2DeviceImpl] closed" << std::endl;
}

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
  return impl_->enumerated_devices_[idx].serial;
}

std::string Freenect2::getDefaultDeviceSerialNumber()
{
  return getDeviceSerialNumber(0);
}

Freenect2Device *Freenect2::openDevice(int idx)
{
  return openDevice(idx, true);
}

Freenect2Device *Freenect2::openDevice(int idx, bool attempting_reset)
{
  int num_devices = impl_->getNumDevices();
  Freenect2DeviceImpl *device = 0;

  if(idx >= num_devices)
  {
    std::cout << "[Freenect2Impl] requested device " << idx << " is not connected!" << std::endl;
    return device;
  }

  Freenect2Impl::UsbDeviceWithSerial &dev = impl_->enumerated_devices_[idx];
  libusb_device_handle *dev_handle;

  if(impl_->tryGetDevice(dev.dev, &device))
  {
    std::cout << "[Freenect2Impl] failed to get device " << PrintBusAndDevice(dev.dev)
        << " (the device may already be open)" << std::endl;
    return device;
  }

  int r = libusb_open(dev.dev, &dev_handle);

  if(r != LIBUSB_SUCCESS)
  {
    std::cout << "[Freenect2Impl] failed to open Kinect v2 " << PrintBusAndDevice(dev.dev) << "!" << std::endl;
    return device;
  }

  if(attempting_reset)
  {
    r = libusb_reset_device(dev_handle);

    if(r == LIBUSB_ERROR_NOT_FOUND) 
    {
      // From libusb documentation:
      // "If the reset fails, the descriptors change, or the previous state
      // cannot be restored, the device will appear to be disconnected and
      // reconnected. This means that the device handle is no longer valid (you
      // should close it) and rediscover the device. A return code of
      // LIBUSB_ERROR_NOT_FOUND indicates when this is the case."

      // be a good citizen
      libusb_close(dev_handle);

      // HACK: wait for the planets to align... (When the reset fails it may
      // take a short while for the device to show up on the bus again. In the
      // absence of hotplug support, we just wait a little. If this code path
      // is followed there will already be a delay opening the device fully so
      // adding a little more is tolerable.)
      libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(1000));

      // reenumerate devices
      std::cout << "[Freenect2Impl] re-enumerating devices after reset" << std::endl;
      impl_->clearDeviceEnumeration();
      impl_->enumerateDevices();

      // re-open without reset
      return openDevice(idx, false);
    }
    else if(r != LIBUSB_SUCCESS)
    {
      std::cout << "[Freenect2Impl] failed to reset Kinect v2 " << PrintBusAndDevice(dev.dev) << "!" << std::endl;
      return device;
    }
  }

  device = new Freenect2DeviceImpl(impl_, dev.dev, dev_handle, dev.serial);
  impl_->addDevice(device);

  if(!device->open())
  {
    delete device;
    device = 0;

    std::cout << "[Freenect2DeviceImpl] failed to open Kinect v2 " << PrintBusAndDevice(dev.dev) << "!" << std::endl;
  }

  return device;
}

Freenect2Device *Freenect2::openDevice(const std::string &serial)
{
  Freenect2Device *device = 0;
  int num_devices = impl_->getNumDevices();

  for(int idx = 0; idx < num_devices; ++idx)
  {
    if(impl_->enumerated_devices_[idx].serial == serial)
    {
      device = openDevice(idx);
      break;
    }
  }

  return device;
}

Freenect2Device *Freenect2::openDefaultDevice()
{
  return openDevice(0);
}

} /* namespace libfreenect2 */
