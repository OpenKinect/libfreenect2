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

#include <libfreenect2/depth_packet_processor.h>
#include <libfreenect2/resource.h>
#include <libfreenect2/protocol/response.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>

#include <stdexcept>
#include <cstring>
#include <cmath>

#if defined(WIN32)
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#define OUT_NAME(FUNCTION) "[CudaDepthPacketProcessor::" FUNCTION "] "
namespace libfreenect2
{

class CudaDepthPacketProcessorImpl
{
public:
  short lut11to16[2048];
  float x_table[512 * 424];
  float z_table[512 * 424];
  Float4 p0_table[512 * 424];
  libfreenect2::DepthPacketProcessor::Config config;
  DepthPacketProcessor::Parameters params;

  CudaDepthPacketProcessorKernel kernel;

  double timing_acc;
  double timing_acc_n;

  double timing_current_start;

  Frame *ir_frame, *depth_frame;

  size_t image_size;

  bool deviceInitialized;
  bool programInitialized;

  CudaDepthPacketProcessorImpl(const int deviceId = -1) : deviceInitialized(false), programInitialized(false)
  {
    newIrFrame();
    newDepthFrame();

    timing_acc = 0.0;
    timing_acc_n = 0.0;
    timing_current_start = 0.0;
    image_size = 512 * 424;

    deviceInitialized = initDevice(deviceId);
  }

  bool initDevice(const int deviceId)
  {
    size_t block_size = 128;
    try
    {
      kernel.initDevice(deviceId, image_size, block_size);
    }
    catch (const std::runtime_error &err)
    {
      std::cerr << OUT_NAME("initDevice") << err.what() << std::endl;
      return false;
    }
    return true;
  }

  bool initProgram()
  {
    if(!deviceInitialized)
    {
      return false;
    }

    try
    {
      kernel.generateOptions(params, config);
      kernel.loadTables(lut11to16, p0_table, x_table, z_table);
    }
    catch (const std::runtime_error &err)
    {
      std::cerr << OUT_NAME("initProgram") << err.what() << std::endl;
      throw err;
    }
    programInitialized = true;
    return true;
  }

  void run(const DepthPacket &packet)
  {
    kernel.run(packet, ir_frame, depth_frame, config);
  }

  void startTiming()
  {
    timing_current_start = cv::getTickCount();
  }

  void stopTiming()
  {
    timing_acc += (cv::getTickCount() - timing_current_start) / cv::getTickFrequency();
    timing_acc_n += 1.0;

    if(timing_acc_n >= 100.0)
    {
      double avg = (timing_acc / timing_acc_n);
      std::cout << "[CudaDepthPacketProcessor] avg. time: " << (avg * 1000) << "ms -> ~" << (1.0 / avg) << "Hz" << std::endl;
      timing_acc = 0.0;
      timing_acc_n = 0.0;
    }
  }

  void newIrFrame()
  {
    ir_frame = new Frame(512, 424, 4);
  }

  void newDepthFrame()
  {
    depth_frame = new Frame(512, 424, 4);
  }

  void fill_trig_table(const libfreenect2::protocol::P0TablesResponse *p0table)
  {
    for(int r = 0; r < 424; ++r)
    {
      Float4 *it = &p0_table[r * 512];
      const uint16_t *it0 = &p0table->p0table0[r * 512];
      const uint16_t *it1 = &p0table->p0table1[r * 512];
      const uint16_t *it2 = &p0table->p0table2[r * 512];
      for(int c = 0; c < 512; ++c, ++it, ++it0, ++it1, ++it2)
      {
        it->x = -((float) * it0) * 0.000031 * M_PI;
        it->y = -((float) * it1) * 0.000031 * M_PI;
        it->z = -((float) * it2) * 0.000031 * M_PI;
        it->w = 0.0f;
      }
    }
  }
};

CudaDepthPacketProcessor::CudaDepthPacketProcessor(const int deviceId) :
  impl_(new CudaDepthPacketProcessorImpl(deviceId))
{
}

CudaDepthPacketProcessor::~CudaDepthPacketProcessor()
{
  delete impl_;
}

void CudaDepthPacketProcessor::setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config)
{
  DepthPacketProcessor::setConfiguration(config);
  impl_->config = config;
  impl_->programInitialized = false;
}

void CudaDepthPacketProcessor::loadP0TablesFromCommandResponse(unsigned char *buffer, size_t buffer_length)
{
  libfreenect2::protocol::P0TablesResponse *p0table = (libfreenect2::protocol::P0TablesResponse *)buffer;

  if(buffer_length < sizeof(libfreenect2::protocol::P0TablesResponse))
  {
    std::cerr << OUT_NAME("loadP0TablesFromCommandResponse") "P0Table response too short!" << std::endl;
    return;
  }

  impl_->fill_trig_table(p0table);
}

void CudaDepthPacketProcessor::loadXTableFromFile(const char *filename)
{
  if(!loadBufferFromResources(filename, (unsigned char *)impl_->x_table, impl_->image_size * sizeof(float)))
  {
    std::cerr << OUT_NAME("loadXTableFromFile") "could not load x table from: " << filename << std::endl;
  }
}

void CudaDepthPacketProcessor::loadZTableFromFile(const char *filename)
{
  if(!loadBufferFromResources(filename, (unsigned char *)impl_->z_table, impl_->image_size * sizeof(float)))
  {
    std::cerr << OUT_NAME("loadZTableFromFile") "could not load z table from: " << filename << std::endl;
  }
}

void CudaDepthPacketProcessor::load11To16LutFromFile(const char *filename)
{
  if(!loadBufferFromResources(filename, (unsigned char *)impl_->lut11to16, 2048 * sizeof(short)))
  {
    std::cerr << OUT_NAME("load11To16LutFromFile") "could not load lut table from: " << filename << std::endl;
  }
}

void CudaDepthPacketProcessor::process(const DepthPacket &packet)
{
  bool has_listener = this->listener_ != 0;

  if(!impl_->programInitialized && !impl_->initProgram())
  {
    std::cerr << OUT_NAME("process") "could not initialize CudaDepthPacketProcessor" << std::endl;
    return;
  }

  impl_->startTiming();

  impl_->ir_frame->timestamp = packet.timestamp;
  impl_->depth_frame->timestamp = packet.timestamp;
  impl_->ir_frame->sequence = packet.sequence;
  impl_->depth_frame->sequence = packet.sequence;

  impl_->run(packet);

  impl_->stopTiming();

  if(has_listener)
  {
    if(this->listener_->onNewFrame(Frame::Ir, impl_->ir_frame))
    {
      impl_->newIrFrame();
    }

    if(this->listener_->onNewFrame(Frame::Depth, impl_->depth_frame))
    {
      impl_->newDepthFrame();
    }
  }
}

} /* namespace libfreenect2 */

