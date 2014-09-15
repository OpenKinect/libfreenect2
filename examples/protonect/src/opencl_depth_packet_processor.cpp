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

#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
  #include <OpenCL/cl.hpp>
#else
  #include <CL/cl.hpp>
#endif

#ifndef REG_OPENCL_FILE
#define REG_OPENCL_FILE ""
#endif

namespace libfreenect2
{

bool loadBufferFromResources(const std::string &filename, unsigned char *buffer, const size_t n)
{
  size_t length = 0;
  const unsigned char *data = NULL;

  if(!loadResource(filename, &data, &length))
  {
    std::cerr << "failed to load resource: " << filename << std::endl;
    return false;
  }

  if(length != n)
  {
    std::cerr << "wrong size of resource: " << filename << std::endl;
    return false;
  }

  memcpy(buffer, data, length);
  return true;
}

std::string loadCLSource(const std::string &filename)
{
  const unsigned char *data;
  size_t length = 0;

  if(!loadResource(filename, &data, &length))
  {
    std::cerr << "failed to load cl source!" << std::endl;
    return "";
  }

  return std::string(reinterpret_cast<const char *>(data), length);
}

class OpenCLDepthPacketProcessorImpl
{
public:
  cl_short lut11to16[2048];
  cl_float x_table[512 * 424];
  cl_float z_table[512 * 424];
  cl_float3 p0_table[512 * 424];
  libfreenect2::DepthPacketProcessor::Config config;
  DepthPacketProcessor::Parameters params;

  double timing_acc;
  double timing_acc_n;

  double timing_current_start;

  bool enable_bilateral_filter, enable_edge_filter;

  Frame *ir_frame, *depth_frame;

  cl::Context context;
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;

  cl::Program program;
  cl::CommandQueue queue;

  cl::Kernel kernel_processPixelStage1;
  cl::Kernel kernel_filterPixelStage1;
  cl::Kernel kernel_processPixelStage2;
  cl::Kernel kernel_filterPixelStage2;

  size_t image_size;

  // Read only buffers
  size_t buf_lut11to16_size;
  size_t buf_p0_table_size;
  size_t buf_x_table_size;
  size_t buf_z_table_size;
  size_t buf_packet_size;

  cl::Buffer buf_lut11to16;
  cl::Buffer buf_p0_table;
  cl::Buffer buf_x_table;
  cl::Buffer buf_z_table;
  cl::Buffer buf_packet;

  // Read-Write buffers
  size_t buf_a_size;
  size_t buf_b_size;
  size_t buf_n_size;
  size_t buf_ir_size;
  size_t buf_a_filtered_size;
  size_t buf_b_filtered_size;
  size_t buf_edge_test_size;
  size_t buf_depth_size;
  size_t buf_ir_sum_size;
  size_t buf_filtered_size;

  cl::Buffer buf_a;
  cl::Buffer buf_b;
  cl::Buffer buf_n;
  cl::Buffer buf_ir;
  cl::Buffer buf_a_filtered;
  cl::Buffer buf_b_filtered;
  cl::Buffer buf_edge_test;
  cl::Buffer buf_depth;
  cl::Buffer buf_ir_sum;
  cl::Buffer buf_filtered;

  bool isInitialized;

  OpenCLDepthPacketProcessorImpl() : isInitialized(false)
  {
    newIrFrame();
    newDepthFrame();

    timing_acc = 0.0;
    timing_acc_n = 0.0;
    timing_current_start = 0.0;
    image_size = 512 * 424;

    enable_bilateral_filter = true;
    enable_edge_filter = true;
  }

  void generateOptions(std::string &options) const
  {
    std::ostringstream oss;
    oss.precision(16);
    oss << std::scientific;
    oss << " -D BFI_BITMASK=" << "0x180";

    oss << " -D AB_MULTIPLIER=" << params.ab_multiplier << "f";
    oss << " -D AB_MULTIPLIER_PER_FRQ0=" << params.ab_multiplier_per_frq[0] << "f";
    oss << " -D AB_MULTIPLIER_PER_FRQ1=" << params.ab_multiplier_per_frq[1] << "f";
    oss << " -D AB_MULTIPLIER_PER_FRQ2=" << params.ab_multiplier_per_frq[2] << "f";
    oss << " -D AB_OUTPUT_MULTIPLIER=" << params.ab_output_multiplier << "f";

    oss << " -D PHASE_IN_RAD0=" << params.phase_in_rad[0] << "f";
    oss << " -D PHASE_IN_RAD1=" << params.phase_in_rad[1] << "f";
    oss << " -D PHASE_IN_RAD2=" << params.phase_in_rad[2] << "f";

    oss << " -D JOINT_BILATERAL_AB_THRESHOLD=" << params.joint_bilateral_ab_threshold << "f";
    oss << " -D JOINT_BILATERAL_MAX_EDGE=" << params.joint_bilateral_max_edge << "f";
    oss << " -D JOINT_BILATERAL_EXP=" << params.joint_bilateral_exp << "f";
    oss << " -D JOINT_BILATERAL_THRESHOLD=" << (params.joint_bilateral_ab_threshold * params.joint_bilateral_ab_threshold) / (params.ab_multiplier * params.ab_multiplier) << "f";
    oss << " -D GAUSSIAN_KERNEL_0=" << params.gaussian_kernel[0] << "f";
    oss << " -D GAUSSIAN_KERNEL_1=" << params.gaussian_kernel[1] << "f";
    oss << " -D GAUSSIAN_KERNEL_2=" << params.gaussian_kernel[2] << "f";
    oss << " -D GAUSSIAN_KERNEL_3=" << params.gaussian_kernel[3] << "f";
    oss << " -D GAUSSIAN_KERNEL_4=" << params.gaussian_kernel[4] << "f";
    oss << " -D GAUSSIAN_KERNEL_5=" << params.gaussian_kernel[5] << "f";
    oss << " -D GAUSSIAN_KERNEL_6=" << params.gaussian_kernel[6] << "f";
    oss << " -D GAUSSIAN_KERNEL_7=" << params.gaussian_kernel[7] << "f";
    oss << " -D GAUSSIAN_KERNEL_8=" << params.gaussian_kernel[8] << "f";

    oss << " -D PHASE_OFFSET=" << params.phase_offset << "f";
    oss << " -D UNAMBIGIOUS_DIST=" << params.unambigious_dist << "f";
    oss << " -D INDIVIDUAL_AB_THRESHOLD=" << params.individual_ab_threshold << "f";
    oss << " -D AB_THRESHOLD=" << params.ab_threshold << "f";
    oss << " -D AB_CONFIDENCE_SLOPE=" << params.ab_confidence_slope << "f";
    oss << " -D AB_CONFIDENCE_OFFSET=" << params.ab_confidence_offset << "f";
    oss << " -D MIN_DEALIAS_CONFIDENCE=" << params.min_dealias_confidence << "f";
    oss << " -D MAX_DEALIAS_CONFIDENCE=" << params.max_dealias_confidence << "f";

    oss << " -D EDGE_AB_AVG_MIN_VALUE=" << params.edge_ab_avg_min_value << "f";
    oss << " -D EDGE_AB_STD_DEV_THRESHOLD=" << params.edge_ab_std_dev_threshold << "f";
    oss << " -D EDGE_CLOSE_DELTA_THRESHOLD=" << params.edge_close_delta_threshold << "f";
    oss << " -D EDGE_FAR_DELTA_THRESHOLD=" << params.edge_far_delta_threshold << "f";
    oss << " -D EDGE_MAX_DELTA_THRESHOLD=" << params.edge_max_delta_threshold << "f";
    oss << " -D EDGE_AVG_DELTA_THRESHOLD=" << params.edge_avg_delta_threshold << "f";
    oss << " -D MAX_EDGE_COUNT=" << params.max_edge_count << "f";

    oss << " -D MIN_DEPTH=" << params.min_depth << "f";
    oss << " -D MAX_DEPTH=" << params.max_depth << "f";
    options = oss.str();
  }

  bool init()
  {
    if(isInitialized)
    {
      return true;
    }

    std::string sourceCode;
    if(!readProgram(sourceCode))
    {
      return false;
    }

    cl_int err = CL_SUCCESS;
    try
    {
      cl::Platform::get(&platforms);
      if(platforms.size() == 0)
      {
        std::cerr << "Platform size 0" << std::endl;
        return false;
      }

      cl_context_properties properties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0};
      context = cl::Context(CL_DEVICE_TYPE_GPU, properties);

      devices = context.getInfo<CL_CONTEXT_DEVICES>();

      std::string options;
      generateOptions(options);

      cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
      program = cl::Program(context, source);
      program.build(devices, options.c_str());

      queue = cl::CommandQueue(context, devices[0], 0, &err);

      //Read only
      buf_lut11to16_size = 2048 * sizeof(cl_short);
      buf_p0_table_size = image_size * sizeof(cl_float3);
      buf_x_table_size = image_size * sizeof(cl_float);
      buf_z_table_size = image_size * sizeof(cl_float);
      buf_packet_size = ((image_size * 11) / 16) * 10 * sizeof(cl_ushort);

      buf_lut11to16 = cl::Buffer(context, CL_READ_ONLY_CACHE, buf_lut11to16_size, NULL, &err);
      buf_p0_table = cl::Buffer(context, CL_READ_ONLY_CACHE, buf_p0_table_size, NULL, &err);
      buf_x_table = cl::Buffer(context, CL_READ_ONLY_CACHE, buf_x_table_size, NULL, &err);
      buf_z_table = cl::Buffer(context, CL_READ_ONLY_CACHE, buf_z_table_size, NULL, &err);
      buf_packet = cl::Buffer(context, CL_READ_ONLY_CACHE, buf_packet_size, NULL, &err);

      //Read-Write
      buf_a_size = image_size * sizeof(cl_float3);
      buf_b_size = image_size * sizeof(cl_float3);
      buf_n_size = image_size * sizeof(cl_float3);
      buf_ir_size = image_size * sizeof(cl_float);
      buf_a_filtered_size = image_size * sizeof(cl_float3);
      buf_b_filtered_size = image_size * sizeof(cl_float3);
      buf_edge_test_size = image_size * sizeof(cl_uchar);
      buf_depth_size = image_size * sizeof(cl_float);
      buf_ir_sum_size = image_size * sizeof(cl_float);
      buf_filtered_size = image_size * sizeof(cl_float);

      buf_a = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_a_size, NULL, &err);
      buf_b = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_b_size, NULL, &err);
      buf_n = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_n_size, NULL, &err);
      buf_ir = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_ir_size, NULL, &err);
      buf_a_filtered = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_a_filtered_size, NULL, &err);
      buf_b_filtered = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_b_filtered_size, NULL, &err);
      buf_edge_test = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_edge_test_size, NULL, &err);
      buf_depth = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
      buf_ir_sum = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_ir_sum_size, NULL, &err);
      buf_filtered = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_filtered_size, NULL, &err);

      kernel_processPixelStage1 = cl::Kernel(program, "processPixelStage1", &err);
      kernel_processPixelStage1.setArg(0, buf_lut11to16);
      kernel_processPixelStage1.setArg(1, buf_z_table);
      kernel_processPixelStage1.setArg(2, buf_p0_table);
      kernel_processPixelStage1.setArg(3, buf_packet);
      kernel_processPixelStage1.setArg(4, buf_a);
      kernel_processPixelStage1.setArg(5, buf_b);
      kernel_processPixelStage1.setArg(6, buf_n);
      kernel_processPixelStage1.setArg(7, buf_ir);

      kernel_filterPixelStage1 = cl::Kernel(program, "filterPixelStage1", &err);
      kernel_filterPixelStage1.setArg(0, buf_a);
      kernel_filterPixelStage1.setArg(1, buf_b);
      kernel_filterPixelStage1.setArg(2, buf_n);
      kernel_filterPixelStage1.setArg(3, buf_a_filtered);
      kernel_filterPixelStage1.setArg(4, buf_b_filtered);
      kernel_filterPixelStage1.setArg(5, buf_edge_test);

      kernel_processPixelStage2 = cl::Kernel(program, "processPixelStage2", &err);
      kernel_processPixelStage2.setArg(0, config.EnableBilateralFilter ? buf_a_filtered : buf_a);
      kernel_processPixelStage2.setArg(1, config.EnableBilateralFilter ? buf_b_filtered : buf_b);
      kernel_processPixelStage2.setArg(2, buf_x_table);
      kernel_processPixelStage2.setArg(3, buf_z_table);
      kernel_processPixelStage2.setArg(4, buf_depth);
      kernel_processPixelStage2.setArg(5, buf_ir_sum);

      kernel_filterPixelStage2 = cl::Kernel(program, "filterPixelStage2", &err);
      kernel_filterPixelStage2.setArg(0, buf_depth);
      kernel_filterPixelStage2.setArg(1, buf_ir_sum);
      kernel_filterPixelStage2.setArg(2, buf_edge_test);
      kernel_filterPixelStage2.setArg(3, buf_filtered);

      cl::Event event0, event1, event2, event3;
      queue.enqueueWriteBuffer(buf_lut11to16, CL_FALSE, 0, buf_lut11to16_size, lut11to16, NULL, &event0);
      queue.enqueueWriteBuffer(buf_p0_table, CL_FALSE, 0, buf_p0_table_size, p0_table, NULL, &event1);
      queue.enqueueWriteBuffer(buf_x_table, CL_FALSE, 0, buf_x_table_size, x_table, NULL, &event2);
      queue.enqueueWriteBuffer(buf_z_table, CL_FALSE, 0, buf_z_table_size, z_table, NULL, &event3);

      event0.wait();
      event1.wait();
      event2.wait();
      event3.wait();
    }
    catch(const cl::Error &err)
    {
      std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;

      if(err.err() == CL_BUILD_PROGRAM_FAILURE)
      {
        std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
        std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
        std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
      }

      throw err;
      return false;
    }
    isInitialized = true;
    return true;
  }

  void run(const DepthPacket &packet)
  {
    try
    {
      std::vector<cl::Event> eventWrite(1), eventPPS1(1), eventFPS1(1), eventPPS2(1), eventFPS2(1);
      cl::Event event0, event1;

      queue.enqueueWriteBuffer(buf_packet, CL_FALSE, 0, buf_packet_size, packet.buffer, NULL, &eventWrite[0]);

      queue.enqueueNDRangeKernel(kernel_processPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &eventWrite, &eventPPS1[0]);
      queue.enqueueReadBuffer(buf_ir, CL_FALSE, 0, buf_ir_size, ir_frame->data, &eventPPS1, &event0);

      if(config.EnableBilateralFilter)
      {
        queue.enqueueNDRangeKernel(kernel_filterPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &eventPPS1, &eventFPS1[0]);
      }
      else
      {
        eventFPS1[0] = eventPPS1[0];
      }

      queue.enqueueNDRangeKernel(kernel_processPixelStage2, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &eventFPS1, &eventPPS2[0]);

      if(config.EnableEdgeAwareFilter)
      {
        queue.enqueueNDRangeKernel(kernel_filterPixelStage2, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &eventPPS2, &eventFPS2[0]);
      }
      else
      {
        eventFPS2[0] = eventPPS2[0];
      }

      queue.enqueueReadBuffer(enable_edge_filter ? buf_filtered : buf_depth, CL_FALSE, 0, buf_depth_size, depth_frame->data, &eventFPS2, &event1);
      event0.wait();
      event1.wait();
    }
    catch(const cl::Error &err)
    {
      std::cerr << "ERROR: " << err.what() << " (" << err.err() << ")" << std::endl;
      throw err;
      return;
    }
  }

  bool readProgram(std::string &source) const
  {
    source = loadCLSource(REG_OPENCL_FILE);
    return !source.empty();
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
      std::cout << "[OpenCLDepthPacketProcessor] avg. time: " << (avg * 1000) << "ms -> ~" << (1.0 / avg) << "Hz" << std::endl;
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
      cl_float3 *it = &p0_table[r * 512];
      const uint16_t *it0 = &p0table->p0table0[r * 512];
      const uint16_t *it1 = &p0table->p0table1[r * 512];
      const uint16_t *it2 = &p0table->p0table2[r * 512];
      for(int c = 0; c < 512; ++c, ++it, ++it0, ++it1, ++it2)
      {
        it->s[0] = -((float) * it0) * 0.000031 * M_PI;
        it->s[1] = -((float) * it1) * 0.000031 * M_PI;
        it->s[2] = -((float) * it2) * 0.000031 * M_PI;
        it->s[3] = 0.0f;
      }
    }
  }
};

OpenCLDepthPacketProcessor::OpenCLDepthPacketProcessor() :
  impl_(new OpenCLDepthPacketProcessorImpl())
{
}

OpenCLDepthPacketProcessor::~OpenCLDepthPacketProcessor()
{
  delete impl_;
}

void OpenCLDepthPacketProcessor::setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config)
{
  DepthPacketProcessor::setConfiguration(config);
  impl_->config = config;

  impl_->enable_bilateral_filter = config.EnableBilateralFilter;
  impl_->enable_edge_filter = config.EnableEdgeAwareFilter;
}

void OpenCLDepthPacketProcessor::loadP0TablesFromCommandResponse(unsigned char *buffer, size_t buffer_length)
{
  libfreenect2::protocol::P0TablesResponse *p0table = (libfreenect2::protocol::P0TablesResponse *)buffer;

  if(buffer_length < sizeof(libfreenect2::protocol::P0TablesResponse))
  {
    std::cerr << "[CpuDepthPacketProcessor::loadP0TablesFromCommandResponse] P0Table response too short!" << std::endl;
    return;
  }

  impl_->fill_trig_table(p0table);
  impl_->init();
}

void OpenCLDepthPacketProcessor::loadXTableFromFile(const char *filename)
{
  if(!loadBufferFromResources(filename, (unsigned char *)impl_->x_table, impl_->image_size * sizeof(float)))
  {
    std::cerr << "could not load x table from: " << filename << std::endl;
  }
}

void OpenCLDepthPacketProcessor::loadZTableFromFile(const char *filename)
{
  if(!loadBufferFromResources(filename, (unsigned char *)impl_->z_table, impl_->image_size * sizeof(float)))
  {
    std::cerr << "could not load z table from: " << filename << std::endl;
  }
}

void OpenCLDepthPacketProcessor::load11To16LutFromFile(const char *filename)
{
  if(!loadBufferFromResources(filename, (unsigned char *)impl_->lut11to16, 2048 * sizeof(cl_ushort)))
  {
    std::cerr << "could not load lut table from: " << filename << std::endl;
  }
}

void OpenCLDepthPacketProcessor::process(const DepthPacket &packet)
{
  bool has_listener = this->listener_ != 0;

  if(!impl_->init())
  {
    std::cerr << "could not initialize OpenCLDepthPacketProcessor" << std::endl;
    return;
  }

  impl_->startTiming();

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

