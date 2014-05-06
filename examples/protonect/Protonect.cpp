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

#include "Protonect.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <opencv2/opencv.hpp>

#include <libfreenect2/tables.h>
#include <libfreenect2/usb/event_loop.h>
#include <libfreenect2/usb/transfer_pool.h>
#include <libfreenect2/rgb_packet_stream_parser.h>
#include <libfreenect2/rgb_packet_processor.h>
#include <libfreenect2/depth_packet_stream_parser.h>
#include <libfreenect2/frame_listener.h>
#include <libfreenect2/protocol/command.h>
#include <libfreenect2/protocol/command_transaction.h>
#include <libfreenect2/protocol/usb_control.h>

void hexdump( uint8_t* buffer, int size, const char* info ) {
  printf("dumping %d bytes of raw data from command %s: \n",size,info);
  int lines = size >> 4;
  if (size % 16 != 0) lines += 1;
  for (int i = 0; i < lines; i++)
  {
    printf("0x%04x:  ", i*16);
    for (int j = 0; j < 16; j++)
    {
      if (j < size) printf("%02x ",buffer[i*16+j]);
      else printf("   ");
    }
    printf("    ");
    for (int j = 0; (j < 16) && (j < size); j++)
    {
      char c = buffer[i*16+j];
      printf("%c",(((c<32)||(c>128))?'.':c));
    }
    printf("\n");
    size -= 16;
  }
}

bool InitKinect2(libfreenect2::protocol::UsbControl& usb_ctrl)
{
  using namespace libfreenect2::protocol;

  if(usb_ctrl.setIsochronousDelay() != UsbControl::Success) return false;
  // TODO: always fails right now with error 6 - TRANSFER_OVERFLOW!
  //if(usb_ctrl.setPowerStateLatencies() != UsbControl::Success) return false;
  if(usb_ctrl.setIrInterfaceState(UsbControl::Disabled) != UsbControl::Success) return false;
  if(usb_ctrl.enablePowerStates() != UsbControl::Success) return false;
  if(usb_ctrl.setVideoTransferFunctionState(UsbControl::Disabled) != UsbControl::Success) return false;

  return true;
}

void RunKinect(libfreenect2::protocol::UsbControl& usb_ctrl, libusb_device_handle *handle, libfreenect2::DepthPacketProcessor& depth_processor)
{
  if (NULL == handle)
    return;

  printf("running kinect...\n");
  int r;

  using namespace libfreenect2::protocol;

  CommandTransaction tx(handle, 0x81, 0x02);
  CommandTransaction::Result result;

  uint32_t seq = cmd_seq;

  usb_ctrl.setVideoTransferFunctionState(UsbControl::Enabled);
  //r = KSetSensorStatus(handle, KSENSOR_ENABLE);

  tx.execute(ReadFirmwareVersionsCommand(seq++), result);
  hexdump(result.data, result.length, "ReadFirmwareVersions");
  //r = KReadFirmwareVersions(handle);

  tx.execute(ReadData0x14Command(seq++), result);
  hexdump(result.data, result.length, "ReadData0x14");
  //r = KReadData14(handle);

  tx.execute(ReadSerialNumberCommand(seq++), result);
  hexdump(result.data, result.length, "ReadSerialNumber");
  //r = KReadData22_1(handle);

  tx.execute(ReadDepthCameraParametersCommand(seq++), result);
  //r = KReadDepthCameraParams(handle);

  tx.execute(ReadP0TablesCommand(seq++), result);
  depth_processor.loadP0TablesFromCommandResponse(result.data, result.length);
  //r = KReadP0Tables(handle, depth_processor);

  tx.execute(ReadRgbCameraParametersCommand(seq++), result);
  //r = KReadCameraParams(handle);

  tx.execute(ReadStatus0x090000Command(seq++), result);
  hexdump(result.data, result.length, "Status");
  //r = KReadStatus90000(handle);

  tx.execute(InitStreamsCommand(seq++), result);
  //r = KInitStreams(handle);

  usb_ctrl.setIrInterfaceState(UsbControl::Enabled);
  //r = KSetStreamingInterfaceStatus(handle, KSTREAM_ENABLE);

  tx.execute(ReadStatus0x090000Command(seq++), result);
  hexdump(result.data, result.length, "Status");
  //r = KReadStatus90000(handle);

  tx.execute(SetStreamEnabledCommand(seq++), result);
  //r = KSetStreamStatus(handle, KSTREAM_ENABLE);

  cmd_seq = seq;
}

void CloseKinect2(libfreenect2::protocol::UsbControl& usb_ctrl)
{
  printf("closing kinect...\n");
  usb_ctrl.setVideoTransferFunctionState(libfreenect2::protocol::UsbControl::Disabled);
}

bool shutdown = false;

void sigint_handler(int s)
{
  shutdown = true;
}

int main(int argc, char *argv[])
{
  std::string program_path(argv[0]);
  size_t executable_name_idx = program_path.rfind("Protonect");

  std::string binpath = "/";

  if(executable_name_idx != std::string::npos)
  {
    binpath = program_path.substr(0, executable_name_idx);
  }

  uint16_t vid = 0x045E;
  uint16_t pid = 0x02C4;
  uint16_t mi = 0x00;

  bool debug_mode = false;

  libusb_device_handle *handle;
  libusb_device *dev;
  uint8_t bus;
  const char* speed_name[5] =
  { "Unknown", "1.5 Mbit/s (USB LowSpeed)", "12 Mbit/s (USB FullSpeed)", "480 Mbit/s (USB HighSpeed)", "5000 Mbit/s (USB SuperSpeed)" };

  int r;

  const struct libusb_version* version;
  version = libusb_get_version();
  printf("Using libusbx v%d.%d.%d.%d\n\n", version->major, version->minor, version->micro, version->nano);

  r = libusb_init(NULL);
  if (r < 0)
    return r;

  libusb_set_debug(NULL, debug_mode ? LIBUSB_LOG_LEVEL_DEBUG : LIBUSB_LOG_LEVEL_INFO);

  printf("Opening device %04X:%04X...\n", vid, pid);
  handle = libusb_open_device_with_vid_pid(NULL, vid, pid);

  if (handle == NULL)
  {
    perr("  Failed.\n");
    //system("PAUSE");
    return -1;
  }

  dev = libusb_get_device(handle);
  bus = libusb_get_bus_number(dev);
  /*
   struct libusb_device_descriptor dev_desc;

   printf("\nReading device descriptor:\n");
   CALL_CHECK(libusb_get_device_descriptor(dev, &dev_desc));
   printf("            length: %d\n", dev_desc.bLength);
   printf("      device class: %d\n", dev_desc.bDeviceClass);
   printf("               S/N: %d\n", dev_desc.iSerialNumber);
   printf("           VID:PID: %04X:%04X\n", dev_desc.idVendor, dev_desc.idProduct);
   printf("         bcdDevice: %04X\n", dev_desc.bcdDevice);
   printf("   iMan:iProd:iSer: %d:%d:%d\n", dev_desc.iManufacturer, dev_desc.iProduct, dev_desc.iSerialNumber);
   printf("          nb confs: %d\n", dev_desc.bNumConfigurations);
   */

  r = libusb_get_device_speed(dev);
  if ((r < 0) || (r > 4))
    r = 0;
  printf("             speed: %s\n", speed_name[r]);

  libfreenect2::protocol::UsbControl usb_ctrl(handle);
  usb_ctrl.setConfiguration();
  usb_ctrl.claimInterfaces();

  if(!InitKinect2(usb_ctrl)) return -1;

  // install signal handler now
  signal(SIGINT,sigint_handler);
  shutdown = false;

  libfreenect2::usb::EventLoop usb_loop;
  usb_loop.start();

  libfreenect2::FrameMap frames;
  libfreenect2::FrameListener frame_listener(libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth);

  //libfreenect2::DumpRgbPacketProcessor rgb_processor;
  libfreenect2::TurboJpegRgbPacketProcessor rgb_processor;
  rgb_processor.setFrameListener(&frame_listener);
  libfreenect2::RgbPacketStreamParser rgb_packet_stream_parser(&rgb_processor);

  libfreenect2::usb::BulkTransferPool rgb_bulk_transfers(handle, 0x83);
  rgb_bulk_transfers.allocate(50, 0x4000);
  rgb_bulk_transfers.setCallback(&rgb_packet_stream_parser);
  rgb_bulk_transfers.enableSubmission();

  glfwInit();
  //glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  //glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


  GLFWwindow* window = 0;//glfwCreateWindow(800, 600, "OpenGL", 0, 0); // Windowed

  libfreenect2::OpenGLDepthPacketProcessor depth_processor(window);
  depth_processor.setFrameListener(&frame_listener);
  depth_processor.load11To16LutFromFile((binpath + "../11to16.bin").c_str());
  depth_processor.loadXTableFromFile((binpath + "../xTable.bin").c_str());
  depth_processor.loadZTableFromFile((binpath + "../zTable.bin").c_str());

  libfreenect2::DepthPacketStreamParser depth_packet_stream_parser(&depth_processor);

  size_t max_packet_size = libusb_get_max_iso_packet_size(dev, 0x84);
  std::cout << "iso max_packet_size: " << max_packet_size << std::endl;

  libfreenect2::usb::IsoTransferPool depth_iso_transfers(handle, 0x84);
  depth_iso_transfers.allocate(80, 8, max_packet_size);
  depth_iso_transfers.setCallback(&depth_packet_stream_parser);
  depth_iso_transfers.enableSubmission();

  r = libusb_get_device_speed(dev);
  if ((r < 0) || (r > 4))
    r = 0;
  printf("             speed: %s\n", speed_name[r]);

  RunKinect(usb_ctrl, handle, depth_processor);

  rgb_bulk_transfers.submit(10);
  depth_iso_transfers.submit(60);

  r = libusb_get_device_speed(dev);
  if ((r < 0) || (r > 4))
    r = 0;
  printf("             speed: %s\n", speed_name[r]);


  while(!shutdown)
  {
    frame_listener.waitForNewFrame(frames);

    libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
    libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
    libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

    cv::imshow("rgb", cv::Mat(rgb->height, rgb->width, CV_8UC3, rgb->data));
    cv::imshow("ir", cv::Mat(ir->height, ir->width, CV_32FC1, ir->data) / 20000.0f);
    cv::imshow("depth", cv::Mat(depth->height, depth->width, CV_32FC1, depth->data) / 4500.0f);
    cv::waitKey(1);

    frame_listener.release(frames);
  }

  //glfwDestroyWindow(window);

  r = libusb_get_device_speed(dev);
  if ((r < 0) || (r > 4))
    r = 0;
  printf("             speed: %s\n", speed_name[r]);

  rgb_bulk_transfers.disableSubmission();
  depth_iso_transfers.disableSubmission();
  CloseKinect2(usb_ctrl);

  rgb_bulk_transfers.cancel();
  depth_iso_transfers.cancel();

  // wait for all transfers to cancel
  // TODO: better implementation
  libfreenect2::this_thread::sleep_for(libfreenect2::chrono::seconds(2));

  rgb_bulk_transfers.deallocate();
  depth_iso_transfers.deallocate();

  r = libusb_get_device_speed(dev);
  if ((r < 0) || (r > 4))
    r = 0;
  printf("             speed: %s\n", speed_name[r]);

  usb_ctrl.releaseInterfaces();

  printf("Closing device...\n");
  libusb_close(handle);

  usb_loop.stop();

  libusb_exit(NULL);
  // TODO: causes segfault
  //glfwTerminate();

  //system("PAUSE");
  return 0;
}
