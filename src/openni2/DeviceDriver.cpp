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
/**
*  FreenectDriver
*  Copyright 2013 Benn Snyder <benn.snyder@gmail.com>
*/

#include <map>
#include <string>
#include <cstdlib>
#include <vector>
#include <Driver/OniDriverAPI.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>
#include "DepthStream.hpp"
#include "ColorStream.hpp"
#include "IrStream.hpp"


namespace Freenect2Driver
{
  typedef std::map<std::string, std::string> ConfigStrings;

  class DeviceImpl : public Device
  {
  private:
    libfreenect2::Freenect2Device *dev;
    ColorStream* color;
    DepthStream* depth;
    IrStream* ir;
    Registration *reg;
    ConfigStrings config;
    bool device_stop;
    bool device_used;
    libfreenect2::SyncMultiFrameListener listener;
    libfreenect2::thread* thread;

    static void static_run(void* cookie)
    {
      static_cast<DeviceImpl*>(cookie)->run();
    }

    VideoStream* getStream(libfreenect2::Frame::Type type)
    {
      if (type == libfreenect2::Frame::Depth)
        return depth;
      if (type == libfreenect2::Frame::Ir)
        return ir;
      if (type == libfreenect2::Frame::Color)
        return color;
      return NULL;
    }

    void run()
    {
      libfreenect2::FrameMap frames;
      uint32_t seqNum = 0;
      libfreenect2::Frame::Type seqType;

      struct streams {
        const char* name;
        libfreenect2::Frame::Type type;
      } streams[] = {
          { "Ir",    libfreenect2::Frame::Ir    },
          { "Depth", libfreenect2::Frame::Depth },
          { "Color", libfreenect2::Frame::Color }
      };
      while(!device_stop)
      {
        listener.waitForNewFrame(frames);

        for (unsigned i = 0; i < sizeof(streams)/sizeof(*streams); i++) {
          struct streams& s = streams[i];
          VideoStream* stream = getStream(s.type);
          libfreenect2::Frame *frame = frames[s.type];
          if (stream) {
            if (seqNum == 0)
              seqType = s.type;
            if (s.type == seqType)
              seqNum++;
            frame->timestamp = seqNum * 33369;
            stream->buildFrame(frame);
          }
        }

        listener.release(frames);
      }
    }

    OniStatus setStreamProperties(VideoStream* stream, std::string pfx)
    {
      pfx += '-';
      OniStatus res = ONI_STATUS_OK, tmp_res;
      if (config.find(pfx + "size") != config.end()) {
        WriteMessage("setStreamProperty: " + pfx + "size: " + config[pfx + "size"]);
        std::string size(config[pfx + "size"]);
        int i = size.find("x");
        OniVideoMode video_mode = makeOniVideoMode(ONI_PIXEL_FORMAT_DEPTH_1_MM, atoi(size.substr(0, i).c_str()), atoi(size.substr(i + 1).c_str()), 30);
        tmp_res = stream->setProperty(ONI_STREAM_PROPERTY_VIDEO_MODE, (void*)&video_mode, sizeof(video_mode));
        if (tmp_res != ONI_STATUS_OK)
          res = tmp_res;
      }

      return res;
    }

    void allocStream() { 
      if (! color) {
        color = new ColorStream(this, dev, reg);
        setStreamProperties(color, "color");
      }
      if (! depth) {
        depth = new DepthStream(this, dev, reg);
        setStreamProperties(depth, "depth");
      }
      if (! ir) {
        ir = new IrStream(this, dev, reg);
        setStreamProperties(ir, "ir");
      }
    }

    void deallocStream() { 
      if (color != NULL)
      {
        delete color;
        color = NULL;
      }
      if (depth != NULL)
      {
        delete depth;
        depth = NULL;
      }
      if (ir != NULL)
      {
        delete ir;
        ir = NULL;
      }
    }

  public:
    DeviceImpl(int index) : //libfreenect2::Freenect2Device(fn_ctx, index),
      dev(NULL),
      color(NULL),
      depth(NULL),
      ir(NULL),
      reg(NULL),
      device_stop(true),
      device_used(false),
      listener(libfreenect2::Frame::Depth | libfreenect2::Frame::Ir | libfreenect2::Frame::Color),
      thread(NULL)
    {
    }
    ~DeviceImpl()
    {
      destroyStream(color);
      destroyStream(ir);
      destroyStream(depth);
      deallocStream();
      close();
      if (reg) {
        delete reg;
        reg = NULL;
      }
    }

    // for Freenect2Device
    void setFreenect2Device(libfreenect2::Freenect2Device *dev) {
      this->dev = dev;
      dev->setColorFrameListener(&listener);
      dev->setIrAndDepthFrameListener(&listener);
      reg = new Registration(dev);
      allocStream();
    }
    void setConfigStrings(ConfigStrings& config) {
      this->config = config;
    }
    void start() {
      WriteMessage("Freenect2Driver::Device: start()");
      if (device_stop) {
        device_used = true;
        device_stop = false;
        thread = new libfreenect2::thread(&DeviceImpl::static_run, this);
        dev->start();
      }
    }
    void stop() { 
      WriteMessage("Freenect2Driver::Device: stop()");
      if (!device_stop) {
        device_stop = true;
        thread->join();
        dev->stop();
      }
    }
    void close() { 
      WriteMessage("Freenect2Driver::Device: close()");
      if (this->dev && device_used) {
        stop();
        dev->close();
      }
      this->dev = NULL;
    }

    // for DeviceBase

    OniBool isImageRegistrationModeSupported(OniImageRegistrationMode mode) { return depth->isImageRegistrationModeSupported(mode); }

    OniStatus getSensorInfoList(OniSensorInfo** pSensors, int* numSensors)
    {
      *numSensors = 3;
      OniSensorInfo * sensors = new OniSensorInfo[*numSensors];
      sensors[0] = depth->getSensorInfo();
      sensors[1] = color->getSensorInfo();
      sensors[2] = ir->getSensorInfo();
      *pSensors = sensors;
      return ONI_STATUS_OK;
    }

    oni::driver::StreamBase* createStream(OniSensorType sensorType)
    {
      switch (sensorType)
      {
        default:
          LogError("Cannot create a stream of type " + to_string(sensorType));
          return NULL;
        case ONI_SENSOR_COLOR:
          WriteMessage("Device: createStream(color)");
          return color;
        case ONI_SENSOR_DEPTH:
          WriteMessage("Device: createStream(depth)");
          return depth;
        case ONI_SENSOR_IR:
          WriteMessage("Device: createStream(ir)");
          return ir;
      }
    }

    void destroyStream(oni::driver::StreamBase* pStream)
    {
      if (pStream == color)
        WriteMessage("Device: destroyStream(color)");
      if (pStream == depth)
        WriteMessage("Device: destroyStream(depth)");
      if (pStream == ir)
        WriteMessage("Device: destroyStream(ir)");
    }

    // todo: fill out properties
    OniBool isPropertySupported(int propertyId)
    {
      if (propertyId == ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION)
        return true;
      return false;
    }

    OniStatus getProperty(int propertyId, void* data, int* pDataSize)
    {
      switch (propertyId)
      {
        default:
        case ONI_DEVICE_PROPERTY_FIRMWARE_VERSION:        // string
        case ONI_DEVICE_PROPERTY_DRIVER_VERSION:          // OniVersion
        case ONI_DEVICE_PROPERTY_HARDWARE_VERSION:        // int
        case ONI_DEVICE_PROPERTY_SERIAL_NUMBER:           // string
        case ONI_DEVICE_PROPERTY_ERROR_STATE:             // ?
        // files
        case ONI_DEVICE_PROPERTY_PLAYBACK_SPEED:          // float
        case ONI_DEVICE_PROPERTY_PLAYBACK_REPEAT_ENABLED: // OniBool
        // xn
        case XN_MODULE_PROPERTY_USB_INTERFACE:            // XnSensorUsbInterface
        case XN_MODULE_PROPERTY_MIRROR:                   // bool
        case XN_MODULE_PROPERTY_RESET_SENSOR_ON_STARTUP:  // unsigned long long
        case XN_MODULE_PROPERTY_LEAN_INIT:                // unsigned long long
        case XN_MODULE_PROPERTY_SERIAL_NUMBER:            // unsigned long long
        case XN_MODULE_PROPERTY_VERSION:                  // XnVersions
          return ONI_STATUS_NOT_SUPPORTED;

        case ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION:      // OniImageRegistrationMode
          if (*pDataSize != sizeof(OniImageRegistrationMode))
          {
            LogError("Unexpected size for ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION");
            return ONI_STATUS_ERROR;
          }
          *(static_cast<OniImageRegistrationMode*>(data)) = depth->getImageRegistrationMode();
          return ONI_STATUS_OK;
      }
    }
    
    OniStatus setProperty(int propertyId, const void* data, int dataSize)
    {
      switch (propertyId)
      {
        default:
        case ONI_DEVICE_PROPERTY_FIRMWARE_VERSION:        // By implementation
        case ONI_DEVICE_PROPERTY_DRIVER_VERSION:          // OniVersion
        case ONI_DEVICE_PROPERTY_HARDWARE_VERSION:        // int
        case ONI_DEVICE_PROPERTY_SERIAL_NUMBER:           // string
        case ONI_DEVICE_PROPERTY_ERROR_STATE:             // ?
        // files
        case ONI_DEVICE_PROPERTY_PLAYBACK_SPEED:          // float
        case ONI_DEVICE_PROPERTY_PLAYBACK_REPEAT_ENABLED: // OniBool
        // xn
        case XN_MODULE_PROPERTY_USB_INTERFACE:            // XnSensorUsbInterface
        case XN_MODULE_PROPERTY_MIRROR:                   // bool
        case XN_MODULE_PROPERTY_RESET_SENSOR_ON_STARTUP:  // unsigned long long
        case XN_MODULE_PROPERTY_LEAN_INIT:                // unsigned long long
        case XN_MODULE_PROPERTY_SERIAL_NUMBER:            // unsigned long long
        case XN_MODULE_PROPERTY_VERSION:                  // XnVersions
        // xn commands
        case XN_MODULE_PROPERTY_FIRMWARE_PARAM:           // XnInnerParam
        case XN_MODULE_PROPERTY_RESET:                    // unsigned long long
        case XN_MODULE_PROPERTY_IMAGE_CONTROL:            // XnControlProcessingData
        case XN_MODULE_PROPERTY_DEPTH_CONTROL:            // XnControlProcessingData
        case XN_MODULE_PROPERTY_AHB:                      // XnAHBData
        case XN_MODULE_PROPERTY_LED_STATE:                // XnLedState
          return ONI_STATUS_NOT_SUPPORTED;

        case ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION:      // OniImageRegistrationMode
          if (dataSize != sizeof(OniImageRegistrationMode))
          {
            LogError("Unexpected size for ONI_DEVICE_PROPERTY_IMAGE_REGISTRATION");
            return ONI_STATUS_ERROR;
          }
          OniImageRegistrationMode mode = *(static_cast<const OniImageRegistrationMode*>(data));
          color->setImageRegistrationMode(mode);
          return depth->setImageRegistrationMode(mode);
      }
    }

    OniBool isCommandSupported(int commandId)
    {
      switch (commandId)
      {
        default:
        case ONI_DEVICE_COMMAND_SEEK:
          return false;
      }
    }
    
    OniStatus invoke(int commandId, void* data, int dataSize)
    {
      switch (commandId)
      {
        default:
        case ONI_DEVICE_COMMAND_SEEK: // OniSeek
          return ONI_STATUS_NOT_SUPPORTED;
      }
    }


    /* todo: for DeviceBase
    virtual OniStatus tryManualTrigger() {return ONI_STATUS_OK;}
    */
  };


  class Driver : public oni::driver::DriverBase
  {
  private:
    typedef std::map<OniDeviceInfo, oni::driver::DeviceBase*> OniDeviceMap;
    OniDeviceMap devices;
    std::string uriScheme;
    ConfigStrings config;
    libfreenect2::Freenect2 freenect2;

    std::string devid_to_uri(int id) {
      return uriScheme + "://" + to_string(id);
    }

    int uri_to_devid(const std::string uri) {
      int id;
      std::istringstream is(uri);
      is.seekg((uriScheme + "://").length());
      is >> id;
      return id;
    }

    void register_uri(std::string uri) {
      OniDeviceInfo info;
      strncpy(info.uri, uri.c_str(), ONI_MAX_STR);
      strncpy(info.vendor, "Microsoft", ONI_MAX_STR);
      //strncpy(info.name, "Kinect 2", ONI_MAX_STR); // XXX, NiTE does not accept new name
      strncpy(info.name, "Kinect", ONI_MAX_STR);
      if (devices.find(info) == devices.end()) {
        WriteMessage("Driver: register new uri: " + uri);
        devices[info] = NULL;
        deviceConnected(&info);
        deviceStateChanged(&info, 0);
      }
    }

  public:
    Driver(OniDriverServices* pDriverServices) : DriverBase(pDriverServices),
      uriScheme("freenect2")
    {
        //WriteMessage("Using libfreenect v" + to_string(PROJECT_VER));
      WriteMessage("Using libfreenect2");

      DriverServices = &getServices();
    }
    ~Driver() { shutdown(); }

    // for DriverBase

    OniStatus initialize(oni::driver::DeviceConnectedCallback connectedCallback, oni::driver::DeviceDisconnectedCallback disconnectedCallback, oni::driver::DeviceStateChangedCallback deviceStateChangedCallback, void* pCookie)
    {
      DriverBase::initialize(connectedCallback, disconnectedCallback, deviceStateChangedCallback, pCookie);
      for (int i = 0; i < freenect2.enumerateDevices(); i++)
      {
        std::string uri = devid_to_uri(i);
        const char* modes_c[] = {
          "",
          "?depth-size=640x480",
          "?depth-size=512x424",
        };
        std::vector<std::string> modes(modes_c, modes_c + 3);

        WriteMessage("Found device " + uri);

        for (unsigned i = 0; i < modes.size(); i++) {
          register_uri(uri + modes[i]);
        }

#if 0
        freenect_device* dev;
        if (freenect_open_device(m_ctx, &dev, i) == 0)
        {
          info.usbVendorId = dev->usb_cam.VID;
          info.usbProductId = dev->usb_cam.PID;
          freenect_close_device(dev);
        }
        else
        {
          WriteMessage("Unable to open device to query VID/PID");
        }
#endif // 0
      }
      return ONI_STATUS_OK;
    }

    oni::driver::DeviceBase* deviceOpen(const char* c_uri, const char* c_mode = NULL)
    {
      std::string uri(c_uri);
      std::string mode(c_mode ? c_mode : "");
      if (uri.find("?") != std::string::npos) {
        mode += "&";
        mode += uri.substr(uri.find("?") + 1);
        uri = uri.substr(0, uri.find("?"));
      }
      std::stringstream ss(mode);
      std::string buf;
      while(std::getline(ss, buf, '&')) {
        if (buf.find("=") != std::string::npos) {
          config[buf.substr(0, buf.find("="))] = buf.substr(buf.find("=")+1);
        } else {
          if (0 < buf.length())
            config[buf] = "";
        }
      }
      WriteMessage("deiveOpen: " + uri);
      for (std::map<std::string, std::string>::iterator it = config.begin(); it != config.end(); it++) {
        WriteMessage("    " + it->first + " = " + it->second);
      }

      for (OniDeviceMap::iterator iter = devices.begin(); iter != devices.end(); iter++)
      {
        std::string iter_uri(iter->first.uri);
        if (iter_uri.substr(0, iter_uri.find("?")) == uri) // found
        {
          if (iter->second) // already open
          {
            return iter->second;
          }
          else 
          {
            WriteMessage("Opening device " + std::string(uri));
            int id = uri_to_devid(iter->first.uri);
            DeviceImpl* device = new DeviceImpl(id);
            // The LIBFREENECT2_PIPELINE variable allows to select
            // the non-default pipeline
            device->setFreenect2Device(freenect2.openDevice(id));
            device->setConfigStrings(config);
            iter->second = device;
            return device;
          }
        }
      }

      LogError("Could not find device " + std::string(uri));
      return NULL;
    }

    void deviceClose(oni::driver::DeviceBase* pDevice)
    {
      for (OniDeviceMap::iterator iter = devices.begin(); iter != devices.end(); iter++)
      {
        if (iter->second == pDevice)
        {
          WriteMessage("Closing device " + std::string(iter->first.uri));
          //int id = uri_to_devid(iter->first.uri);

          DeviceImpl* device = (DeviceImpl*)iter->second;
          device->stop();
          device->close();

          devices.erase(iter);
          return;
        }
      }

      LogError("Could not close unrecognized device");
    }

    OniStatus tryDevice(const char* uri)
    {
      oni::driver::DeviceBase* device = deviceOpen(uri);
      if (! device)
        return ONI_STATUS_ERROR;
      deviceClose(device);
      register_uri(std::string(uri)); // XXX, register new uri here.
      return ONI_STATUS_OK;
    }

    void shutdown()
    {
      for (OniDeviceMap::iterator iter = devices.begin(); iter != devices.end(); iter++)
      {
        if (iter->second) {
          deviceClose(iter->second);
        }
      }
    }


    /* todo: for DriverBase
    virtual void* enableFrameSync(oni::driver::StreamBase** pStreams, int streamCount);
    virtual void disableFrameSync(void* frameSyncGroup);
    */
  };
}


// macros defined in XnLib (not included) - workaround
#define XN_NEW(type, arg) new type(arg)
#define XN_DELETE(p) delete(p)
ONI_EXPORT_DRIVER(Freenect2Driver::Driver);
#undef XN_NEW
#undef XN_DELETE
