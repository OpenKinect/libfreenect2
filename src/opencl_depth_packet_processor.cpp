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

/** @file opencl_depth_packet_processor.cl Implementation of the OpenCL depth packet processor. */

#include <libfreenect2/depth_packet_processor.h>
#include <libfreenect2/resource.h>
#include <libfreenect2/protocol/response.h>
#include <libfreenect2/logging.h>

#include <sstream>

#define _USE_MATH_DEFINES
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#undef CL_VERSION_1_2
#include <CL/cl.hpp>
#endif

#ifndef REG_OPENCL_FILE
#define REG_OPENCL_FILE ""
#endif

namespace libfreenect2
{

std::string loadCLSource(const std::string &filename)
{
  const unsigned char *data;
  size_t length = 0;

  if(!loadResource(filename, &data, &length))
  {
    LOG_ERROR << "failed to load cl source!";
    return "";
  }

  return std::string(reinterpret_cast<const char *>(data), length);
}

class OpenCLDepthPacketProcessorImpl: public WithPerfLogging
{
public:
  cl_short lut11to16[2048];
  cl_float x_table[512 * 424];
  cl_float z_table[512 * 424];
  cl_float3 p0_table[512 * 424];
  libfreenect2::DepthPacketProcessor::Config config;
  DepthPacketProcessor::Parameters params;

  Frame *ir_frame, *depth_frame;

  cl::Context context;
  cl::Device device;

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

  bool deviceInitialized;
  bool programBuilt;
  bool programInitialized;
  std::string sourceCode;

  OpenCLDepthPacketProcessorImpl(const int deviceId = -1) 
    : deviceInitialized(false)
    , programBuilt(false)
    , programInitialized(false)
  {
    newIrFrame();
    newDepthFrame();

    image_size = 512 * 424;

    deviceInitialized = initDevice(deviceId);

    const int CL_ICDL_VERSION = 2;
    typedef cl_int (*icdloader_func)(int, size_t, void*, size_t*);
    icdloader_func clGetICDLoaderInfoOCLICD = (icdloader_func)clGetExtensionFunctionAddress("clGetICDLoaderInfoOCLICD");
    if (clGetICDLoaderInfoOCLICD != NULL)
    {
      char buf[16];
      if (clGetICDLoaderInfoOCLICD(CL_ICDL_VERSION, sizeof(buf), buf, NULL) == CL_SUCCESS)
      {
        if (strcmp(buf, "2.2.4") < 0)
          LOG_WARNING << "Your ocl-icd has deadlock bugs. Update to 2.2.4+ is recommended.";
      }
    }
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

    oss << " -D MIN_DEPTH=" << config.MinDepth * 1000.0f << "f";
    oss << " -D MAX_DEPTH=" << config.MaxDepth * 1000.0f << "f";
    options = oss.str();
  }

  void getDevices(const std::vector<cl::Platform> &platforms, std::vector<cl::Device> &devices)
  {
    devices.clear();
    for(size_t i = 0; i < platforms.size(); ++i)
    {
      const cl::Platform &platform = platforms[i];

      std::vector<cl::Device> devs;
      if(platform.getDevices(CL_DEVICE_TYPE_ALL, &devs) != CL_SUCCESS)
      {
        continue;
      }

      devices.insert(devices.end(), devs.begin(), devs.end());
    }
  }

  std::string deviceString(cl::Device &dev)
  {
    std::string devName, devVendor, devType;
    cl_device_type devTypeID;
    dev.getInfo(CL_DEVICE_NAME, &devName);
    dev.getInfo(CL_DEVICE_VENDOR, &devVendor);
    dev.getInfo(CL_DEVICE_TYPE, &devTypeID);

    switch(devTypeID)
    {
    case CL_DEVICE_TYPE_CPU:
      devType = "CPU";
      break;
    case CL_DEVICE_TYPE_GPU:
      devType = "GPU";
      break;
    case CL_DEVICE_TYPE_ACCELERATOR:
      devType = "ACCELERATOR";
      break;
    default:
      devType = "CUSTOM/UNKNOWN";
    }

    return devName + " (" + devType + ")[" + devVendor + ']';
  }

  void listDevice(std::vector<cl::Device> &devices)
  {
    LOG_INFO << " devices:";
    for(size_t i = 0; i < devices.size(); ++i)
    {
      LOG_INFO << "  " << i << ": " << deviceString(devices[i]);
    }
  }

  bool selectDevice(std::vector<cl::Device> &devices, const int deviceId)
  {
    if(deviceId != -1 && devices.size() > (size_t)deviceId)
    {
      device = devices[deviceId];
      return true;
    }

    bool selected = false;
    size_t selectedType = 0;

    for(size_t i = 0; i < devices.size(); ++i)
    {
      cl::Device &dev = devices[i];
      cl_device_type devTypeID = 0;
      dev.getInfo(CL_DEVICE_TYPE, &devTypeID);

      if(!selected || (selectedType != CL_DEVICE_TYPE_GPU && devTypeID == CL_DEVICE_TYPE_GPU))
      {
        selectedType = devTypeID;
        selected = true;
        device = dev;
      }
    }
    return selected;
  }

#define CHECK_CL_ERROR(err, str) do {if (err != CL_SUCCESS) {LOG_ERROR << str << " failed: " << err; return false; } } while(0)

  bool initDevice(const int deviceId)
  {
    if(!readProgram(sourceCode))
    {
      return false;
    }

    cl_int err = CL_SUCCESS;
    {
      std::vector<cl::Platform> platforms;
      err = cl::Platform::get(&platforms);
      CHECK_CL_ERROR(err, "cl::Platform::get");

      if(platforms.empty())
      {
        LOG_ERROR << "no opencl platforms found.";
        return false;
      }

      std::vector<cl::Device> devices;
      getDevices(platforms, devices);
      listDevice(devices);
      if(!selectDevice(devices, deviceId))
      {
        LOG_ERROR << "could not find any suitable device";
        return false;
      }
      LOG_INFO << "selected device: " << deviceString(device);

      context = cl::Context(device, NULL, NULL, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Context");
    }

    return buildProgram(sourceCode);
  }

  bool initProgram()
  {
    if(!deviceInitialized)
    {
      return false;
    }

    if (!programBuilt)
      if (!buildProgram(sourceCode))
        return false;

    cl_int err = CL_SUCCESS;
    {
      queue = cl::CommandQueue(context, device, 0, &err);
      CHECK_CL_ERROR(err, "cl::CommandQueue");

      //Read only
      buf_lut11to16_size = 2048 * sizeof(cl_short);
      buf_p0_table_size = image_size * sizeof(cl_float3);
      buf_x_table_size = image_size * sizeof(cl_float);
      buf_z_table_size = image_size * sizeof(cl_float);
      buf_packet_size = ((image_size * 11) / 16) * 10 * sizeof(cl_ushort);

      buf_lut11to16 = cl::Buffer(context, CL_READ_ONLY_CACHE, buf_lut11to16_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_p0_table = cl::Buffer(context, CL_READ_ONLY_CACHE, buf_p0_table_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_x_table = cl::Buffer(context, CL_READ_ONLY_CACHE, buf_x_table_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_z_table = cl::Buffer(context, CL_READ_ONLY_CACHE, buf_z_table_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_packet = cl::Buffer(context, CL_READ_ONLY_CACHE, buf_packet_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");

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
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_b = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_b_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_n = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_n_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_ir = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_ir_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_a_filtered = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_a_filtered_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_b_filtered = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_b_filtered_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_edge_test = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_edge_test_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_depth = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_depth_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_ir_sum = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_ir_sum_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");
      buf_filtered = cl::Buffer(context, CL_READ_WRITE_CACHE, buf_filtered_size, NULL, &err);
      CHECK_CL_ERROR(err, "cl::Buffer");

      kernel_processPixelStage1 = cl::Kernel(program, "processPixelStage1", &err);
      CHECK_CL_ERROR(err, "cl::Kernel");
      err = kernel_processPixelStage1.setArg(0, buf_lut11to16);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage1.setArg(1, buf_z_table);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage1.setArg(2, buf_p0_table);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage1.setArg(3, buf_packet);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage1.setArg(4, buf_a);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage1.setArg(5, buf_b);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage1.setArg(6, buf_n);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage1.setArg(7, buf_ir);
      CHECK_CL_ERROR(err, "setArg");

      kernel_filterPixelStage1 = cl::Kernel(program, "filterPixelStage1", &err);
      CHECK_CL_ERROR(err, "cl::Kernel");
      err = kernel_filterPixelStage1.setArg(0, buf_a);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_filterPixelStage1.setArg(1, buf_b);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_filterPixelStage1.setArg(2, buf_n);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_filterPixelStage1.setArg(3, buf_a_filtered);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_filterPixelStage1.setArg(4, buf_b_filtered);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_filterPixelStage1.setArg(5, buf_edge_test);
      CHECK_CL_ERROR(err, "setArg");

      kernel_processPixelStage2 = cl::Kernel(program, "processPixelStage2", &err);
      CHECK_CL_ERROR(err, "cl::Kernel");
      err = kernel_processPixelStage2.setArg(0, config.EnableBilateralFilter ? buf_a_filtered : buf_a);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage2.setArg(1, config.EnableBilateralFilter ? buf_b_filtered : buf_b);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage2.setArg(2, buf_x_table);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage2.setArg(3, buf_z_table);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage2.setArg(4, buf_depth);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_processPixelStage2.setArg(5, buf_ir_sum);
      CHECK_CL_ERROR(err, "setArg");

      kernel_filterPixelStage2 = cl::Kernel(program, "filterPixelStage2", &err);
      CHECK_CL_ERROR(err, "cl::Kernel");
      err = kernel_filterPixelStage2.setArg(0, buf_depth);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_filterPixelStage2.setArg(1, buf_ir_sum);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_filterPixelStage2.setArg(2, buf_edge_test);
      CHECK_CL_ERROR(err, "setArg");
      err = kernel_filterPixelStage2.setArg(3, buf_filtered);
      CHECK_CL_ERROR(err, "setArg");

      cl::Event event0, event1, event2, event3;
      err = queue.enqueueWriteBuffer(buf_lut11to16, CL_FALSE, 0, buf_lut11to16_size, lut11to16, NULL, &event0);
      CHECK_CL_ERROR(err, "enqueueWriteBuffer");
      err = queue.enqueueWriteBuffer(buf_p0_table, CL_FALSE, 0, buf_p0_table_size, p0_table, NULL, &event1);
      CHECK_CL_ERROR(err, "enqueueWriteBuffer");
      err = queue.enqueueWriteBuffer(buf_x_table, CL_FALSE, 0, buf_x_table_size, x_table, NULL, &event2);
      CHECK_CL_ERROR(err, "enqueueWriteBuffer");
      err = queue.enqueueWriteBuffer(buf_z_table, CL_FALSE, 0, buf_z_table_size, z_table, NULL, &event3);
      CHECK_CL_ERROR(err, "enqueueWriteBuffer");

      err = event0.wait();
      CHECK_CL_ERROR(err, "wait");
      err = event1.wait();
      CHECK_CL_ERROR(err, "wait");
      err = event2.wait();
      CHECK_CL_ERROR(err, "wait");
      err = event3.wait();
      CHECK_CL_ERROR(err, "wait");
    }

    programInitialized = true;
    return true;
  }

  bool run(const DepthPacket &packet)
  {
    cl_int err;
    {
      std::vector<cl::Event> eventWrite(1), eventPPS1(1), eventFPS1(1), eventPPS2(1), eventFPS2(1);
      cl::Event event0, event1;

      err = queue.enqueueWriteBuffer(buf_packet, CL_FALSE, 0, buf_packet_size, packet.buffer, NULL, &eventWrite[0]);
      CHECK_CL_ERROR(err, "enqueueWriteBuffer");

      err = queue.enqueueNDRangeKernel(kernel_processPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &eventWrite, &eventPPS1[0]);
      CHECK_CL_ERROR(err, "enqueueNDRangeKernel");
      err = queue.enqueueReadBuffer(buf_ir, CL_FALSE, 0, buf_ir_size, ir_frame->data, &eventPPS1, &event0);
      CHECK_CL_ERROR(err, "enqueueReadBuffer");

      if(config.EnableBilateralFilter)
      {
        err = queue.enqueueNDRangeKernel(kernel_filterPixelStage1, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &eventPPS1, &eventFPS1[0]);
        CHECK_CL_ERROR(err, "enqueueNDRangeKernel");
      }
      else
      {
        eventFPS1[0] = eventPPS1[0];
      }

      err = queue.enqueueNDRangeKernel(kernel_processPixelStage2, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &eventFPS1, &eventPPS2[0]);
      CHECK_CL_ERROR(err, "enqueueNDRangeKernel");

      if(config.EnableEdgeAwareFilter)
      {
        err = queue.enqueueNDRangeKernel(kernel_filterPixelStage2, cl::NullRange, cl::NDRange(image_size), cl::NullRange, &eventPPS2, &eventFPS2[0]);
        CHECK_CL_ERROR(err, "enqueueWriteBuffer");
      }
      else
      {
        eventFPS2[0] = eventPPS2[0];
      }

      err = queue.enqueueReadBuffer(config.EnableEdgeAwareFilter ? buf_filtered : buf_depth, CL_FALSE, 0, buf_depth_size, depth_frame->data, &eventFPS2, &event1);
      CHECK_CL_ERROR(err, "enqueueReadBuffer");
      err = event0.wait();
      CHECK_CL_ERROR(err, "wait");
      err = event1.wait();
      CHECK_CL_ERROR(err, "wait");
    }
    return true;
  }

  bool readProgram(std::string &source) const
  {
    source = loadCLSource("opencl_depth_packet_processor.cl");
    return !source.empty();
  }

  bool buildProgram(const std::string& sources)
  {
    cl_int err;
    {
      LOG_INFO << "building OpenCL program...";

      std::string options;
      generateOptions(options);

      cl::Program::Sources source(1, std::make_pair(sources.c_str(), sources.length()));
      program = cl::Program(context, source, &err);
      CHECK_CL_ERROR(err, "cl::Program");

      err = program.build(options.c_str());
      if (err != CL_SUCCESS)
      {
        LOG_ERROR << "failed to build program: " << err;
        LOG_ERROR << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
        LOG_ERROR << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device);
        LOG_ERROR << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        programBuilt = false;
        return false;
      }
    }

    LOG_INFO << "OpenCL program built successfully";
    programBuilt = true;
    return true;
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

OpenCLDepthPacketProcessor::OpenCLDepthPacketProcessor(const int deviceId) :
  impl_(new OpenCLDepthPacketProcessorImpl(deviceId))
{
}

OpenCLDepthPacketProcessor::~OpenCLDepthPacketProcessor()
{
  delete impl_;
}

void OpenCLDepthPacketProcessor::setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config)
{
  DepthPacketProcessor::setConfiguration(config);

  if ( impl_->config.MaxDepth != config.MaxDepth 
    || impl_->config.MinDepth != config.MinDepth)
  {
    // OpenCL program needs to be rebuilt, then reinitialized
    impl_->programBuilt = false;
    impl_->programInitialized = false;
  }
  else if (impl_->config.EnableBilateralFilter != config.EnableBilateralFilter
    || impl_->config.EnableEdgeAwareFilter != config.EnableEdgeAwareFilter)
  {
    // OpenCL program only needs to be reinitialized
    impl_->programInitialized = false;
  }

  impl_->config = config;
  if (!impl_->programBuilt)
    impl_->buildProgram(impl_->sourceCode);
}

void OpenCLDepthPacketProcessor::loadP0TablesFromCommandResponse(unsigned char *buffer, size_t buffer_length)
{
  libfreenect2::protocol::P0TablesResponse *p0table = (libfreenect2::protocol::P0TablesResponse *)buffer;

  if(buffer_length < sizeof(libfreenect2::protocol::P0TablesResponse))
  {
    LOG_ERROR << "P0Table response too short!";
    return;
  }

  impl_->fill_trig_table(p0table);
}

void OpenCLDepthPacketProcessor::loadXZTables(const float *xtable, const float *ztable)
{
  std::copy(xtable, xtable + TABLE_SIZE, impl_->x_table);
  std::copy(ztable, ztable + TABLE_SIZE, impl_->z_table);
}

void OpenCLDepthPacketProcessor::loadLookupTable(const short *lut)
{
  std::copy(lut, lut + LUT_SIZE, impl_->lut11to16);
}

void OpenCLDepthPacketProcessor::process(const DepthPacket &packet)
{
  bool has_listener = this->listener_ != 0;

  if(!impl_->programInitialized && !impl_->initProgram())
  {
    LOG_ERROR << "could not initialize OpenCLDepthPacketProcessor";
    return;
  }

  impl_->startTiming();

  impl_->ir_frame->timestamp = packet.timestamp;
  impl_->depth_frame->timestamp = packet.timestamp;
  impl_->ir_frame->sequence = packet.sequence;
  impl_->depth_frame->sequence = packet.sequence;

  bool r = impl_->run(packet);

  impl_->stopTiming(LOG_INFO);

  if(has_listener && r)
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

