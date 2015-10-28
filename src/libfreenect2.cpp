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

/** @file libfreenect2.cpp Freenect2 devices and processing implementation. */

#include <string>
#include <vector>
#include <algorithm>
#include <libusb.h>
#include <limits>
#include <cmath>
#define WRITE_LIBUSB_ERROR(__RESULT) libusb_error_name(__RESULT) << " " << libusb_strerror((libusb_error)__RESULT)

#include <libfreenect2/libfreenect2.hpp>

#include <libfreenect2/usb/event_loop.h>
#include <libfreenect2/usb/transfer_pool.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/protocol/usb_control.h>
#include <libfreenect2/protocol/command.h>
#include <libfreenect2/protocol/response.h>
#include <libfreenect2/protocol/command_transaction.h>
#include <libfreenect2/logging.h>

#ifdef __APPLE__
  #define PKTS_PER_XFER 128
  #define NUM_XFERS 4
#else
  #define PKTS_PER_XFER 8
  #define NUM_XFERS 60
#endif

namespace libfreenect2
{
using namespace libfreenect2;
using namespace libfreenect2::usb;
using namespace libfreenect2::protocol;

/*
For detailed analysis see https://github.com/OpenKinect/libfreenect2/issues/144

The following discussion is in no way authoritative. It is the current best
explanation considering the hardcoded parameters and decompiled code.

p0 tables are the "initial shift" of phase values, as in US8587771 B2.

Three p0 tables are used for "disamgibuation" in the first half of stage 2
processing.

At the end of stage 2 processing:

phase_final is the phase shift used to compute the travel distance.

What is being measured is max_depth (d), the total travel distance of the
reflected ray.

But what we want is depth_fit (z), the distance from reflection to the XY
plane. There are two issues: the distance before reflection is not needed;
and the measured ray is not normal to the XY plane.

Suppose L is the distance between the light source and the focal point (a
fixed constant), and xu,yu is the undistorted and normalized coordinates for
each measured pixel at unit depth.

Through some derivation, we have

    z = (d*d - L*L)/(d*sqrt(xu*xu + yu*yu + 1) - xu*L)/2.

The expression in stage 2 processing is a variant of this, with the term
`-L*L` removed. Detailed derivation can be found in the above issue.

Here, the two terms `sqrt(xu*xu + yu*yu + 1)` and `xu` requires undistorted
coordinates, which is hard to compute in real-time because the inverse of
radial and tangential distortion has no analytical solutions and requires
numeric methods to solve. Thus these two terms are precomputed once and
their variants are stored as ztable and xtable respectively.

Even though x/ztable is derived with undistortion, they are only used to
correct the effect of distortion on the z value. Image warping is needed for
correcting distortion on x-y value, which happens in registration.cpp.
*/
struct IrCameraTables: Freenect2Device::IrCameraParams
{
  std::vector<float> xtable;
  std::vector<float> ztable;
  std::vector<short> lut;

  IrCameraTables(const Freenect2Device::IrCameraParams &parent):
    Freenect2Device::IrCameraParams(parent),
    xtable(DepthPacketProcessor::TABLE_SIZE),
    ztable(DepthPacketProcessor::TABLE_SIZE),
    lut(DepthPacketProcessor::LUT_SIZE)
  {
    const double scaling_factor = 8192;
    const double unambigious_dist = 6250.0/3;
    size_t divergence = 0;
    for (size_t i = 0; i < DepthPacketProcessor::TABLE_SIZE; i++)
    {
      size_t xi = i % 512;
      size_t yi = i / 512;
      double xd = (xi + 0.5 - cx)/fx;
      double yd = (yi + 0.5 - cy)/fy;
      double xu, yu;
      divergence += !undistort(xd, yd, xu, yu);
      xtable[i] = scaling_factor*xu;
      ztable[i] = unambigious_dist/sqrt(xu*xu + yu*yu + 1);
    }

    if (divergence > 0)
      LOG_ERROR << divergence << " pixels in x/ztable have incorrect undistortion.";

    short y = 0;
    for (int x = 0; x < 1024; x++)
    {
      unsigned inc = 1 << (x/128 - (x>=128));
      lut[x] = y;
      lut[1024 + x] = -y;
      y += inc;
    }
    lut[1024] = 32767;
  }

  //x,y: undistorted, normalized coordinates
  //xd,yd: distorted, normalized coordinates
  void distort(double x, double y, double &xd, double &yd) const
  {
    double x2 = x * x;
    double y2 = y * y;
    double r2 = x2 + y2;
    double xy = x * y;
    double kr = ((k3 * r2 + k2) * r2 + k1) * r2 + 1.0;
    xd = x*kr + p2*(r2 + 2*x2) + 2*p1*xy;
    yd = y*kr + p1*(r2 + 2*y2) + 2*p2*xy;
  }

  //The inverse of distort() using Newton's method
  //Return true if converged correctly
  //This function considers tangential distortion with double precision.
  bool undistort(double x, double y, double &xu, double &yu) const
  {
    double x0 = x;
    double y0 = y;

    double last_x = x;
    double last_y = y;
    const int max_iterations = 100;
    int iter;
    for (iter = 0; iter < max_iterations; iter++) {
      double x2 = x*x;
      double y2 = y*y;
      double x2y2 = x2 + y2;
      double x2y22 = x2y2*x2y2;
      double x2y23 = x2y2*x2y22;

      //Jacobian matrix
      double Ja = k3*x2y23 + (k2+6*k3*x2)*x2y22 + (k1+4*k2*x2)*x2y2 + 2*k1*x2 + 6*p2*x + 2*p1*y + 1;
      double Jb = 6*k3*x*y*x2y22 + 4*k2*x*y*x2y2 + 2*k1*x*y + 2*p1*x + 2*p2*y;
      double Jc = Jb;
      double Jd = k3*x2y23 + (k2+6*k3*y2)*x2y22 + (k1+4*k2*y2)*x2y2 + 2*k1*y2 + 2*p2*x + 6*p1*y + 1;

      //Inverse Jacobian
      double Jdet = 1/(Ja*Jd - Jb*Jc);
      double a = Jd*Jdet;
      double b = -Jb*Jdet;
      double c = -Jc*Jdet;
      double d = Ja*Jdet;

      double f, g;
      distort(x, y, f, g);
      f -= x0;
      g -= y0;

      x -= a*f + b*g;
      y -= c*f + d*g;
      const double eps = std::numeric_limits<double>::epsilon()*16;
      if (fabs(x - last_x) <= eps && fabs(y - last_y) <= eps)
        break;
      last_x = x;
      last_y = y;
    }
    xu = x;
    yu = y;
    return iter < max_iterations;
  }
};

/** Freenect2 device implementation. */
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

  const PacketPipeline *pipeline_;
  std::string serial_, firmware_;
  Freenect2Device::IrCameraParams ir_camera_params_;
  Freenect2Device::ColorCameraParams rgb_camera_params_;
public:
  Freenect2DeviceImpl(Freenect2Impl *context, const PacketPipeline *pipeline, libusb_device *usb_device, libusb_device_handle *usb_device_handle, const std::string &serial);
  virtual ~Freenect2DeviceImpl();

  bool isSameUsbDevice(libusb_device* other);

  virtual std::string getSerialNumber();
  virtual std::string getFirmwareVersion();

  virtual Freenect2Device::ColorCameraParams getColorCameraParams();
  virtual Freenect2Device::IrCameraParams getIrCameraParams();
  virtual void setColorCameraParams(const Freenect2Device::ColorCameraParams &params);
  virtual void setIrCameraParams(const Freenect2Device::IrCameraParams &params);

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
  int status_;

  PrintBusAndDevice(libusb_device *dev, int status = 0) : dev_(dev), status_(status) {}
};

std::ostream &operator<<(std::ostream &out, const PrintBusAndDevice& dev)
{
  out << "@" << int(libusb_get_bus_number(dev.dev_)) << ":" << int(libusb_get_device_address(dev.dev_));
  if (dev.status_)
    out << " " << WRITE_LIBUSB_ERROR(dev.status_);
  return out;
}

/** Freenect2 device storage and control. */
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

  bool initialized;

  Freenect2Impl(void *usb_context) :
    managed_usb_context_(usb_context == 0),
    usb_context_(reinterpret_cast<libusb_context *>(usb_context)),
    initialized(false),
    has_device_enumeration_(false)
  {
    if (libusb_get_version()->nano < 10952)
    {
      LOG_ERROR << "Your libusb does not support large iso buffer!";
      return;
    }

    if(managed_usb_context_)
    {
      int r = libusb_init(&usb_context_);
      if(r != 0)
      {
        LOG_ERROR << "failed to create usb context: " << WRITE_LIBUSB_ERROR(r);
        return;
      }
    }

    usb_event_loop_.start(usb_context_);
    initialized = true;
  }

  ~Freenect2Impl()
  {
    if (!initialized)
      return;

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
    if (!initialized)
      return;

    devices_.push_back(device);
  }

  void removeDevice(Freenect2DeviceImpl *device)
  {
    if (!initialized)
      return;

    DeviceVector::iterator it = std::find(devices_.begin(), devices_.end(), device);

    if(it != devices_.end())
    {
      devices_.erase(it);
    }
    else
    {
      LOG_WARNING << "tried to remove device, which is not in the internal device list!";
    }
  }

  bool tryGetDevice(libusb_device *usb_device, Freenect2DeviceImpl **device)
  {
    if (!initialized)
      return false;

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
    if (!initialized)
      return;

    DeviceVector devices(devices_.begin(), devices_.end());

    for(DeviceVector::iterator it = devices.begin(); it != devices.end(); ++it)
    {
      delete (*it);
    }

    if(!devices_.empty())
    {
      LOG_WARNING << "after deleting all devices the internal device list should be empty!";
    }
  }

  void clearDeviceEnumeration()
  {
    if (!initialized)
      return;

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
    if (!initialized)
      return;

    LOG_INFO << "enumerating devices...";
    libusb_device **device_list;
    int num_devices = libusb_get_device_list(usb_context_, &device_list);

    LOG_INFO << num_devices << " usb devices connected";

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

                LOG_INFO << "found valid Kinect v2 " << PrintBusAndDevice(dev) << " with serial " << dev_with_serial.serial;
                // valid Kinect v2
                enumerated_devices_.push_back(dev_with_serial);
                continue;
              }
              else
              {
                LOG_ERROR << "failed to get serial number of Kinect v2: " << PrintBusAndDevice(dev, r);
              }

              libusb_close(dev_handle);
            }
            else
            {
              LOG_ERROR << "failed to open Kinect v2: " << PrintBusAndDevice(dev, r);
            }
          }
        }
        libusb_unref_device(dev);
      }
    }

    libusb_free_device_list(device_list, 0);
    has_device_enumeration_ = true;

    LOG_INFO << "found " << enumerated_devices_.size() << " devices";
  }

  int getNumDevices()
  {
    if (!initialized)
      return 0;

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

Freenect2DeviceImpl::Freenect2DeviceImpl(Freenect2Impl *context, const PacketPipeline *pipeline, libusb_device *usb_device, libusb_device_handle *usb_device_handle, const std::string &serial) :
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
  pipeline_(pipeline),
  serial_(serial),
  firmware_("<unknown>")
{
  rgb_transfer_pool_.setCallback(pipeline_->getRgbPacketParser());
  ir_transfer_pool_.setCallback(pipeline_->getIrPacketParser());
}

Freenect2DeviceImpl::~Freenect2DeviceImpl()
{
  close();
  context_->removeDevice(this);

  delete pipeline_;
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
    unsigned char address = libusb_get_device_address(usb_device_);

    unsigned char other_bus = libusb_get_bus_number(other);
    unsigned char other_address = libusb_get_device_address(other);

    result = (bus == other_bus) && (address == other_address);
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

void Freenect2DeviceImpl::setColorCameraParams(const Freenect2Device::ColorCameraParams &params)
{
  rgb_camera_params_ = params;
}

void Freenect2DeviceImpl::setIrCameraParams(const Freenect2Device::IrCameraParams &params)
{
  ir_camera_params_ = params;
  DepthPacketProcessor *proc = pipeline_->getDepthPacketProcessor();
  if (proc != 0)
  {
    IrCameraTables tables(params);
    proc->loadXZTables(&tables.xtable[0], &tables.ztable[0]);
    proc->loadLookupTable(&tables.lut[0]);
  }
}

void Freenect2DeviceImpl::setColorFrameListener(libfreenect2::FrameListener* rgb_frame_listener)
{
  // TODO: should only be possible, if not started
  if(pipeline_->getRgbPacketProcessor() != 0)
    pipeline_->getRgbPacketProcessor()->setFrameListener(rgb_frame_listener);
}

void Freenect2DeviceImpl::setIrAndDepthFrameListener(libfreenect2::FrameListener* ir_frame_listener)
{
  // TODO: should only be possible, if not started
  if(pipeline_->getDepthPacketProcessor() != 0)
    pipeline_->getDepthPacketProcessor()->setFrameListener(ir_frame_listener);
}

bool Freenect2DeviceImpl::open()
{
  LOG_INFO << "opening...";

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
    LOG_ERROR << "max iso packet size for endpoint 0x84 too small! (expected: " << 0x8400 << " got: " << max_iso_packet_size << ")";
    return false;
  }

  rgb_transfer_pool_.allocate(20, 0x4000);
  ir_transfer_pool_.allocate(NUM_XFERS, PKTS_PER_XFER, max_iso_packet_size);

  state_ = Open;

  LOG_INFO << "opened";

  return true;
}

void Freenect2DeviceImpl::start()
{
  LOG_INFO << "starting...";
  if(state_ != Open) return;

  CommandTransaction::Result serial_result, firmware_result, result;

  usb_control_.setVideoTransferFunctionState(UsbControl::Enabled);

  command_tx_.execute(ReadFirmwareVersionsCommand(nextCommandSeq()), firmware_result);
  firmware_ = FirmwareVersionResponse(firmware_result.data, firmware_result.length).toString();

  command_tx_.execute(ReadData0x14Command(nextCommandSeq()), result);
  LOG_DEBUG << "ReadData0x14 response";
  LOG_DEBUG << GenericResponse(result.data, result.length).toString();

  command_tx_.execute(ReadSerialNumberCommand(nextCommandSeq()), serial_result);
  std::string new_serial = SerialNumberResponse(serial_result.data, serial_result.length).toString();

  if(serial_ != new_serial)
  {
    LOG_WARNING << "serial number reported by libusb " << serial_ << " differs from serial number " << new_serial << " in device protocol! ";
  }

  command_tx_.execute(ReadDepthCameraParametersCommand(nextCommandSeq()), result);
  DepthCameraParamsResponse *ir_p = reinterpret_cast<DepthCameraParamsResponse *>(result.data);

  IrCameraParams ir_camera_params_;
  ir_camera_params_.fx = ir_p->fx;
  ir_camera_params_.fy = ir_p->fy;
  ir_camera_params_.cx = ir_p->cx;
  ir_camera_params_.cy = ir_p->cy;
  ir_camera_params_.k1 = ir_p->k1;
  ir_camera_params_.k2 = ir_p->k2;
  ir_camera_params_.k3 = ir_p->k3;
  ir_camera_params_.p1 = ir_p->p1;
  ir_camera_params_.p2 = ir_p->p2;
  setIrCameraParams(ir_camera_params_);

  command_tx_.execute(ReadP0TablesCommand(nextCommandSeq()), result);
  if(pipeline_->getDepthPacketProcessor() != 0)
    pipeline_->getDepthPacketProcessor()->loadP0TablesFromCommandResponse(result.data, result.length);

  command_tx_.execute(ReadRgbCameraParametersCommand(nextCommandSeq()), result);
  RgbCameraParamsResponse *rgb_p = reinterpret_cast<RgbCameraParamsResponse *>(result.data);

  ColorCameraParams rgb_camera_params_;
  rgb_camera_params_.fx = rgb_p->color_f;
  rgb_camera_params_.fy = rgb_p->color_f;
  rgb_camera_params_.cx = rgb_p->color_cx;
  rgb_camera_params_.cy = rgb_p->color_cy;

  rgb_camera_params_.shift_d = rgb_p->shift_d;
  rgb_camera_params_.shift_m = rgb_p->shift_m;

  rgb_camera_params_.mx_x3y0 = rgb_p->mx_x3y0; // xxx
  rgb_camera_params_.mx_x0y3 = rgb_p->mx_x0y3; // yyy
  rgb_camera_params_.mx_x2y1 = rgb_p->mx_x2y1; // xxy
  rgb_camera_params_.mx_x1y2 = rgb_p->mx_x1y2; // yyx
  rgb_camera_params_.mx_x2y0 = rgb_p->mx_x2y0; // xx
  rgb_camera_params_.mx_x0y2 = rgb_p->mx_x0y2; // yy
  rgb_camera_params_.mx_x1y1 = rgb_p->mx_x1y1; // xy
  rgb_camera_params_.mx_x1y0 = rgb_p->mx_x1y0; // x
  rgb_camera_params_.mx_x0y1 = rgb_p->mx_x0y1; // y
  rgb_camera_params_.mx_x0y0 = rgb_p->mx_x0y0; // 1

  rgb_camera_params_.my_x3y0 = rgb_p->my_x3y0; // xxx
  rgb_camera_params_.my_x0y3 = rgb_p->my_x0y3; // yyy
  rgb_camera_params_.my_x2y1 = rgb_p->my_x2y1; // xxy
  rgb_camera_params_.my_x1y2 = rgb_p->my_x1y2; // yyx
  rgb_camera_params_.my_x2y0 = rgb_p->my_x2y0; // xx
  rgb_camera_params_.my_x0y2 = rgb_p->my_x0y2; // yy
  rgb_camera_params_.my_x1y1 = rgb_p->my_x1y1; // xy
  rgb_camera_params_.my_x1y0 = rgb_p->my_x1y0; // x
  rgb_camera_params_.my_x0y1 = rgb_p->my_x0y1; // y
  rgb_camera_params_.my_x0y0 = rgb_p->my_x0y0; // 1
  setColorCameraParams(rgb_camera_params_);

  command_tx_.execute(ReadStatus0x090000Command(nextCommandSeq()), result);
  LOG_DEBUG << "ReadStatus0x090000 response";
  LOG_DEBUG << GenericResponse(result.data, result.length).toString();

  command_tx_.execute(InitStreamsCommand(nextCommandSeq()), result);

  usb_control_.setIrInterfaceState(UsbControl::Enabled);

  command_tx_.execute(ReadStatus0x090000Command(nextCommandSeq()), result);
  LOG_DEBUG << "ReadStatus0x090000 response";
  LOG_DEBUG << GenericResponse(result.data, result.length).toString();

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
  LOG_INFO << "enabling usb transfer submission...";
  rgb_transfer_pool_.enableSubmission();
  ir_transfer_pool_.enableSubmission();

  LOG_INFO << "submitting usb transfers...";
  rgb_transfer_pool_.submit(20);
  ir_transfer_pool_.submit(NUM_XFERS);

  state_ = Streaming;
  LOG_INFO << "started";
}

void Freenect2DeviceImpl::stop()
{
  LOG_INFO << "stopping...";

  if(state_ != Streaming)
  {
    LOG_INFO << "already stopped, doing nothing";
    return;
  }

  LOG_INFO << "disabling usb transfer submission...";
  rgb_transfer_pool_.disableSubmission();
  ir_transfer_pool_.disableSubmission();

  LOG_INFO << "canceling usb transfers...";
  rgb_transfer_pool_.cancel();
  ir_transfer_pool_.cancel();

  usb_control_.setIrInterfaceState(UsbControl::Disabled);

  CommandTransaction::Result result;
  command_tx_.execute(Unknown0x0ACommand(nextCommandSeq()), result);
  command_tx_.execute(SetStreamDisabledCommand(nextCommandSeq()), result);

  usb_control_.setVideoTransferFunctionState(UsbControl::Disabled);

  state_ = Open;
  LOG_INFO << "stopped";
}

void Freenect2DeviceImpl::close()
{
  LOG_INFO << "closing...";

  if(state_ == Closed)
  {
    LOG_INFO << "already closed, doing nothing";
    return;
  }

  if(state_ == Streaming)
  {
    stop();
  }

  if(pipeline_->getRgbPacketProcessor() != 0)
    pipeline_->getRgbPacketProcessor()->setFrameListener(0);

  if(pipeline_->getDepthPacketProcessor() != 0)
    pipeline_->getDepthPacketProcessor()->setFrameListener(0);

  if(has_usb_interfaces_)
  {
    LOG_INFO << "releasing usb interfaces...";

    usb_control_.releaseInterfaces();
    has_usb_interfaces_ = false;
  }

  LOG_INFO << "deallocating usb transfer pools...";
  rgb_transfer_pool_.deallocate();
  ir_transfer_pool_.deallocate();

  LOG_INFO << "closing usb device...";

  libusb_close(usb_device_handle_);
  usb_device_handle_ = 0;
  usb_device_ = 0;

  state_ = Closed;
  LOG_INFO << "closed";
}

PacketPipeline *createDefaultPacketPipeline()
{
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
  return new OpenGLPacketPipeline();
#else
  #ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
    return new OpenCLPacketPipeline();
  #else
  return new CpuPacketPipeline();
  #endif
#endif
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
  if (!impl_->initialized)
    return std::string();

  return impl_->enumerated_devices_[idx].serial;
}

std::string Freenect2::getDefaultDeviceSerialNumber()
{
  return getDeviceSerialNumber(0);
}

Freenect2Device *Freenect2::openDevice(int idx)
{
  return openDevice(idx, createDefaultPacketPipeline());
}

Freenect2Device *Freenect2::openDevice(int idx, const PacketPipeline *pipeline)
{
  return openDevice(idx, pipeline, true);
}

Freenect2Device *Freenect2::openDevice(int idx, const PacketPipeline *pipeline, bool attempting_reset)
{
  int num_devices = impl_->getNumDevices();
  Freenect2DeviceImpl *device = 0;

  if(idx >= num_devices)
  {
    LOG_ERROR << "requested device " << idx << " is not connected!";
    delete pipeline;

    return device;
  }

  Freenect2Impl::UsbDeviceWithSerial &dev = impl_->enumerated_devices_[idx];
  libusb_device_handle *dev_handle;

  if(impl_->tryGetDevice(dev.dev, &device))
  {
    LOG_WARNING << "device " << PrintBusAndDevice(dev.dev)
        << " is already be open!";
    delete pipeline;

    return device;
  }

  int r = libusb_open(dev.dev, &dev_handle);

  if(r != LIBUSB_SUCCESS)
  {
    LOG_ERROR << "failed to open Kinect v2: " << PrintBusAndDevice(dev.dev, r);
    delete pipeline;

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
      LOG_INFO << "re-enumerating devices after reset";
      impl_->clearDeviceEnumeration();
      impl_->enumerateDevices();

      // re-open without reset
      return openDevice(idx, pipeline, false);
    }
    else if(r != LIBUSB_SUCCESS)
    {
      LOG_ERROR << "failed to reset Kinect v2: " << PrintBusAndDevice(dev.dev, r);
      delete pipeline;

      return device;
    }
  }

  device = new Freenect2DeviceImpl(impl_, pipeline, dev.dev, dev_handle, dev.serial);
  impl_->addDevice(device);

  if(!device->open())
  {
    delete device;
    device = 0;

    LOG_ERROR << "failed to open Kinect v2: " << PrintBusAndDevice(dev.dev);
  }

  return device;
}

Freenect2Device *Freenect2::openDevice(const std::string &serial)
{
  return openDevice(serial, createDefaultPacketPipeline());
}

Freenect2Device *Freenect2::openDevice(const std::string &serial, const PacketPipeline *pipeline)
{
  Freenect2Device *device = 0;
  int num_devices = impl_->getNumDevices();

  for(int idx = 0; idx < num_devices; ++idx)
  {
    if(impl_->enumerated_devices_[idx].serial == serial)
    {
      return openDevice(idx, pipeline);
    }
  }

  delete pipeline;
  return device;
}

Freenect2Device *Freenect2::openDefaultDevice()
{
  return openDevice(0);
}

Freenect2Device *Freenect2::openDefaultDevice(const PacketPipeline *pipeline)
{
  return openDevice(0, pipeline);
}

} /* namespace libfreenect2 */
