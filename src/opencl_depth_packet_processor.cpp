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

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS

#ifdef LIBFREENECT2_OPENCL_ICD_LOADER_IS_OLD
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.h>
#ifdef CL_VERSION_1_2
#undef CL_VERSION_1_2
#endif //CL_VERSION_1_2
#endif //LIBFREENECT2_OPENCL_ICD_LOADER_IS_OLD

#include <CL/cl.hpp>

#ifndef REG_OPENCL_FILE
#define REG_OPENCL_FILE ""
#endif

#include <cstdlib>

#define CHECK_CL_PARAM(expr) do { cl_int err = CL_SUCCESS; (expr); if (err != CL_SUCCESS) { LOG_ERROR << #expr ": " << err; return false; } } while(0)
#define CHECK_CL_RETURN(expr) do { cl_int err = (expr); if (err != CL_SUCCESS) { LOG_ERROR << #expr ": " << err; return false; } } while(0)
#define CHECK_CL_ON_FAIL(expr, on_fail) do { cl_int err = (expr); if (err != CL_SUCCESS) { LOG_ERROR << #expr ": " << err; on_fail; return false; } } while(0)

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

class OpenCLDepthPacketProcessorImpl;

class OpenCLBuffer: public Buffer
{
public:
  cl::Buffer buffer;
};

class OpenCLAllocator: public Allocator
{
private:
  cl::Context &context;
  cl::CommandQueue &queue;
  const bool isInputBuffer;

  bool allocate_opencl(OpenCLBuffer *b, size_t size)
  {
    if(isInputBuffer)
    {
      CHECK_CL_PARAM(b->buffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &err));
      CHECK_CL_PARAM(b->data = (unsigned char*)queue.enqueueMapBuffer(b->buffer, CL_TRUE, CL_MAP_WRITE, 0, size, NULL, NULL, &err));
    }
    else
    {
      CHECK_CL_PARAM(b->buffer = cl::Buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, size, NULL, &err));
      CHECK_CL_PARAM(b->data = (unsigned char*)queue.enqueueMapBuffer(b->buffer, CL_TRUE, CL_MAP_READ, 0, size, NULL, NULL, &err));
    }

    b->length = 0;
    b->capacity = size;
    return true;
  }

  bool release_opencl(OpenCLBuffer *b)
  {
    cl::Event event;
    CHECK_CL_RETURN(queue.enqueueUnmapMemObject(b->buffer, b->data, NULL, &event));
    CHECK_CL_RETURN(event.wait());
    return true;
  }

public:
  OpenCLAllocator(cl::Context &context, cl::CommandQueue &queue, bool isInputBuffer) : context(context), queue(queue), isInputBuffer(isInputBuffer)
  {
  }

  virtual Buffer *allocate(size_t size)
  {
    OpenCLBuffer *b = new OpenCLBuffer();
    if(!allocate_opencl(b, size))
      b->data = NULL;
    return b;
  }

  virtual void free(Buffer *b)
  {
    if(b == NULL)
      return;
    release_opencl(static_cast<OpenCLBuffer *>(b));
    delete b;
  }
};

class OpenCLFrame: public Frame
{
private:
  OpenCLBuffer *buffer;

public:
  OpenCLFrame(OpenCLBuffer *buffer)
    : Frame(512, 424, 4, (unsigned char*)-1)
    , buffer(buffer)
  {
    data = buffer->data;
  }

  virtual ~OpenCLFrame()
  {
    buffer->allocator->free(buffer);
    data = NULL;
  }
};

class OpenCLDepthPacketProcessorImpl: public WithPerfLogging
{
public:
  static const size_t IMAGE_SIZE = 512*424;
  static const size_t LUT_SIZE = 2048;

  libfreenect2::DepthPacketProcessor::Config config;
  DepthPacketProcessor::Parameters params;

  Frame *ir_frame, *depth_frame;
  Allocator *input_buffer_allocator;
  Allocator *ir_buffer_allocator;
  Allocator *depth_buffer_allocator;

  cl::Context context;
  cl::Device device;

  cl::Program program;
  cl::CommandQueue queue;

  cl::Kernel kernel_processPixelStage1;
  cl::Kernel kernel_filterPixelStage1;
  cl::Kernel kernel_processPixelStage2;
  cl::Kernel kernel_filterPixelStage2;

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
  bool runtimeOk;
  std::string sourceCode;

#ifdef LIBFREENECT2_WITH_PROFILING_CL
  std::vector<double> timings;
  int count;
#endif

  OpenCLDepthPacketProcessorImpl(const int deviceId = -1)
    : deviceInitialized(false)
    , programBuilt(false)
    , programInitialized(false)
    , runtimeOk(true)
  {
#if _BSD_SOURCE || _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
    setenv("OCL_IGNORE_SELF_TEST", "1", 0);
    setenv("OCL_STRICT_CONFORMANCE", "0", 0);
#endif

    deviceInitialized = initDevice(deviceId);

    input_buffer_allocator = new PoolAllocator(new OpenCLAllocator(context, queue, true));
    ir_buffer_allocator = new PoolAllocator(new OpenCLAllocator(context, queue, false));
    depth_buffer_allocator = new PoolAllocator(new OpenCLAllocator(context, queue, false));

    newIrFrame();
    newDepthFrame();

    const int CL_ICDL_VERSION = 2;
    typedef cl_int (*icdloader_func)(int, size_t, void*, size_t*);
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    icdloader_func clGetICDLoaderInfoOCLICD = (icdloader_func)clGetExtensionFunctionAddress("clGetICDLoaderInfoOCLICD");
#ifdef _MSC_VER
#pragma warning(pop)
#else
#pragma GCC diagnostic pop
#endif
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

  ~OpenCLDepthPacketProcessorImpl()
  {
    delete ir_frame;
    delete depth_frame;
    delete input_buffer_allocator;
    delete ir_buffer_allocator;
    delete depth_buffer_allocator;
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

    oss << " -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math";
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

  bool initDevice(const int deviceId)
  {
    if(!readProgram(sourceCode))
    {
      return false;
    }

    std::vector<cl::Platform> platforms;
    CHECK_CL_RETURN(cl::Platform::get(&platforms));

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

    CHECK_CL_PARAM(context = cl::Context(device, NULL, NULL, NULL, &err));

    if(!initBuffers())
      return false;

    return buildProgram(sourceCode);
  }

  bool initBuffers()
  {
#ifdef LIBFREENECT2_WITH_PROFILING_CL
    count = 0;
    CHECK_CL_PARAM(queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
#else
    CHECK_CL_PARAM(queue = cl::CommandQueue(context, device, 0, &err));
#endif

    //Read only
    buf_lut11to16_size = LUT_SIZE * sizeof(cl_short);
    buf_p0_table_size = IMAGE_SIZE * sizeof(cl_float3);
    buf_x_table_size = IMAGE_SIZE * sizeof(cl_float);
    buf_z_table_size = IMAGE_SIZE * sizeof(cl_float);
    buf_packet_size = ((IMAGE_SIZE * 11) / 16) * 10 * sizeof(cl_ushort);

    CHECK_CL_PARAM(buf_lut11to16 = cl::Buffer(context, CL_MEM_READ_ONLY, buf_lut11to16_size, NULL, &err));
    CHECK_CL_PARAM(buf_p0_table = cl::Buffer(context, CL_MEM_READ_ONLY, buf_p0_table_size, NULL, &err));
    CHECK_CL_PARAM(buf_x_table = cl::Buffer(context, CL_MEM_READ_ONLY, buf_x_table_size, NULL, &err));
    CHECK_CL_PARAM(buf_z_table = cl::Buffer(context, CL_MEM_READ_ONLY, buf_z_table_size, NULL, &err));
    CHECK_CL_PARAM(buf_packet = cl::Buffer(context, CL_MEM_READ_ONLY, buf_packet_size, NULL, &err));

    //Read-Write
    buf_a_size = IMAGE_SIZE * sizeof(cl_float3);
    buf_b_size = IMAGE_SIZE * sizeof(cl_float3);
    buf_n_size = IMAGE_SIZE * sizeof(cl_float3);
    buf_ir_size = IMAGE_SIZE * sizeof(cl_float);
    buf_a_filtered_size = IMAGE_SIZE * sizeof(cl_float3);
    buf_b_filtered_size = IMAGE_SIZE * sizeof(cl_float3);
    buf_edge_test_size = IMAGE_SIZE * sizeof(cl_uchar);
    buf_depth_size = IMAGE_SIZE * sizeof(cl_float);
    buf_ir_sum_size = IMAGE_SIZE * sizeof(cl_float);
    buf_filtered_size = IMAGE_SIZE * sizeof(cl_float);

    CHECK_CL_PARAM(buf_a = cl::Buffer(context, CL_MEM_READ_WRITE, buf_a_size, NULL, &err));
    CHECK_CL_PARAM(buf_b = cl::Buffer(context, CL_MEM_READ_WRITE, buf_b_size, NULL, &err));
    CHECK_CL_PARAM(buf_n = cl::Buffer(context, CL_MEM_READ_WRITE, buf_n_size, NULL, &err));
    CHECK_CL_PARAM(buf_ir = cl::Buffer(context, CL_MEM_WRITE_ONLY, buf_ir_size, NULL, &err));
    CHECK_CL_PARAM(buf_a_filtered = cl::Buffer(context, CL_MEM_READ_WRITE, buf_a_filtered_size, NULL, &err));
    CHECK_CL_PARAM(buf_b_filtered = cl::Buffer(context, CL_MEM_READ_WRITE, buf_b_filtered_size, NULL, &err));
    CHECK_CL_PARAM(buf_edge_test = cl::Buffer(context, CL_MEM_READ_WRITE, buf_edge_test_size, NULL, &err));
    CHECK_CL_PARAM(buf_depth = cl::Buffer(context, CL_MEM_READ_WRITE, buf_depth_size, NULL, &err));
    CHECK_CL_PARAM(buf_ir_sum = cl::Buffer(context, CL_MEM_READ_WRITE, buf_ir_sum_size, NULL, &err));
    CHECK_CL_PARAM(buf_filtered = cl::Buffer(context, CL_MEM_WRITE_ONLY, buf_filtered_size, NULL, &err));

    return true;
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

    CHECK_CL_PARAM(kernel_processPixelStage1 = cl::Kernel(program, "processPixelStage1", &err));
    CHECK_CL_RETURN(kernel_processPixelStage1.setArg(0, buf_lut11to16));
    CHECK_CL_RETURN(kernel_processPixelStage1.setArg(1, buf_z_table));
    CHECK_CL_RETURN(kernel_processPixelStage1.setArg(2, buf_p0_table));
    CHECK_CL_RETURN(kernel_processPixelStage1.setArg(3, buf_packet));
    CHECK_CL_RETURN(kernel_processPixelStage1.setArg(4, buf_a));
    CHECK_CL_RETURN(kernel_processPixelStage1.setArg(5, buf_b));
    CHECK_CL_RETURN(kernel_processPixelStage1.setArg(6, buf_n));
    CHECK_CL_RETURN(kernel_processPixelStage1.setArg(7, buf_ir));

    CHECK_CL_PARAM(kernel_filterPixelStage1 = cl::Kernel(program, "filterPixelStage1", &err));
    CHECK_CL_RETURN(kernel_filterPixelStage1.setArg(0, buf_a));
    CHECK_CL_RETURN(kernel_filterPixelStage1.setArg(1, buf_b));
    CHECK_CL_RETURN(kernel_filterPixelStage1.setArg(2, buf_n));
    CHECK_CL_RETURN(kernel_filterPixelStage1.setArg(3, buf_a_filtered));
    CHECK_CL_RETURN(kernel_filterPixelStage1.setArg(4, buf_b_filtered));
    CHECK_CL_RETURN(kernel_filterPixelStage1.setArg(5, buf_edge_test));

    CHECK_CL_PARAM(kernel_processPixelStage2 = cl::Kernel(program, "processPixelStage2", &err));
    CHECK_CL_RETURN(kernel_processPixelStage2.setArg(0, config.EnableBilateralFilter ? buf_a_filtered : buf_a));
    CHECK_CL_RETURN(kernel_processPixelStage2.setArg(1, config.EnableBilateralFilter ? buf_b_filtered : buf_b));
    CHECK_CL_RETURN(kernel_processPixelStage2.setArg(2, buf_x_table));
    CHECK_CL_RETURN(kernel_processPixelStage2.setArg(3, buf_z_table));
    CHECK_CL_RETURN(kernel_processPixelStage2.setArg(4, buf_depth));
    CHECK_CL_RETURN(kernel_processPixelStage2.setArg(5, buf_ir_sum));

    CHECK_CL_PARAM(kernel_filterPixelStage2 = cl::Kernel(program, "filterPixelStage2", &err));
    CHECK_CL_RETURN(kernel_filterPixelStage2.setArg(0, buf_depth));
    CHECK_CL_RETURN(kernel_filterPixelStage2.setArg(1, buf_ir_sum));
    CHECK_CL_RETURN(kernel_filterPixelStage2.setArg(2, buf_edge_test));
    CHECK_CL_RETURN(kernel_filterPixelStage2.setArg(3, buf_filtered));

    programInitialized = true;
    return true;
  }

  bool run(const DepthPacket &packet)
  {
    std::vector<cl::Event> eventWrite(1), eventPPS1(1), eventFPS1(1), eventPPS2(1), eventFPS2(1);
    cl::Event eventReadIr, eventReadDepth;

    CHECK_CL_RETURN(queue.enqueueWriteBuffer(buf_packet, CL_FALSE, 0, buf_packet_size, packet.buffer, NULL, &eventWrite[0]));
    CHECK_CL_RETURN(queue.enqueueNDRangeKernel(kernel_processPixelStage1, cl::NullRange, cl::NDRange(IMAGE_SIZE), cl::NullRange, &eventWrite, &eventPPS1[0]));
    CHECK_CL_RETURN(queue.enqueueReadBuffer(buf_ir, CL_FALSE, 0, buf_ir_size, ir_frame->data, &eventPPS1, &eventReadIr));

    if(config.EnableBilateralFilter)
    {
      CHECK_CL_RETURN(queue.enqueueNDRangeKernel(kernel_filterPixelStage1, cl::NullRange, cl::NDRange(IMAGE_SIZE), cl::NullRange, &eventPPS1, &eventFPS1[0]));
    }
    else
    {
      eventFPS1[0] = eventPPS1[0];
    }

    CHECK_CL_RETURN(queue.enqueueNDRangeKernel(kernel_processPixelStage2, cl::NullRange, cl::NDRange(IMAGE_SIZE), cl::NullRange, &eventFPS1, &eventPPS2[0]));

    if(config.EnableEdgeAwareFilter)
    {
      CHECK_CL_RETURN(queue.enqueueNDRangeKernel(kernel_filterPixelStage2, cl::NullRange, cl::NDRange(IMAGE_SIZE), cl::NullRange, &eventPPS2, &eventFPS2[0]));
    }
    else
    {
      eventFPS2[0] = eventPPS2[0];
    }

    CHECK_CL_RETURN(queue.enqueueReadBuffer(config.EnableEdgeAwareFilter ? buf_filtered : buf_depth, CL_FALSE, 0, buf_depth_size, depth_frame->data, &eventFPS2, &eventReadDepth));
    CHECK_CL_RETURN(eventReadIr.wait());
    CHECK_CL_RETURN(eventReadDepth.wait());

#ifdef LIBFREENECT2_WITH_PROFILING_CL
    if(count == 0)
    {
      timings.clear();
      timings.resize(7, 0.0);
    }

    timings[0] += eventWrite[0].getProfilingInfo<CL_PROFILING_COMMAND_END>() - eventWrite[0].getProfilingInfo<CL_PROFILING_COMMAND_START>();
    timings[1] += eventPPS1[0].getProfilingInfo<CL_PROFILING_COMMAND_END>() - eventPPS1[0].getProfilingInfo<CL_PROFILING_COMMAND_START>();
    timings[2] += eventFPS1[0].getProfilingInfo<CL_PROFILING_COMMAND_END>() - eventFPS1[0].getProfilingInfo<CL_PROFILING_COMMAND_START>();
    timings[3] += eventPPS2[0].getProfilingInfo<CL_PROFILING_COMMAND_END>() - eventPPS2[0].getProfilingInfo<CL_PROFILING_COMMAND_START>();
    timings[4] += eventFPS2[0].getProfilingInfo<CL_PROFILING_COMMAND_END>() - eventFPS2[0].getProfilingInfo<CL_PROFILING_COMMAND_START>();
    timings[5] += eventReadIr.getProfilingInfo<CL_PROFILING_COMMAND_END>() - eventReadIr.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    timings[6] += eventReadDepth.getProfilingInfo<CL_PROFILING_COMMAND_END>() - eventReadDepth.getProfilingInfo<CL_PROFILING_COMMAND_START>();

    if(++count == 100)
    {
      double sum = timings[0] + timings[1] + timings[2] + timings[3] + timings[4] + timings[5] + timings[6];
      LOG_INFO << "writing package: " << timings[0] / 100000000.0 << " ms.";
      LOG_INFO << "stage 1: " << timings[1] / 100000000.0 << " ms.";
      LOG_INFO << "filter 1: " << timings[2] / 100000000.0 << " ms.";
      LOG_INFO << "stage 2: " << timings[3] / 100000000.0 << " ms.";
      LOG_INFO << "filter 2: " << timings[4] / 100000000.0 << " ms.";
      LOG_INFO << "reading ir: " << timings[5] / 100000000.0 << " ms.";
      LOG_INFO << "reading depth: " << timings[6] / 100000000.0 << " ms.";
      LOG_INFO << "overall: " << sum / 100000000.0 << " ms.";
      count = 0;
    }
#endif

    return true;
  }

  bool readProgram(std::string &source) const
  {
    source = loadCLSource("opencl_depth_packet_processor.cl");
    return !source.empty();
  }

  bool buildProgram(const std::string &sources)
  {
    LOG_INFO << "building OpenCL program...";

    std::string options;
    generateOptions(options);

    cl::Program::Sources source(1, std::make_pair(sources.c_str(), sources.length()));
    CHECK_CL_PARAM(program = cl::Program(context, source, &err));

    CHECK_CL_ON_FAIL(program.build(options.c_str()),
      LOG_ERROR << "failed to build program: " << err;
      LOG_ERROR << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
      LOG_ERROR << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device);
      LOG_ERROR << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));

    LOG_INFO << "OpenCL program built successfully";
    programBuilt = true;
    return true;
  }

  void newIrFrame()
  {
    ir_frame = new OpenCLFrame(static_cast<OpenCLBuffer *>(ir_buffer_allocator->allocate(IMAGE_SIZE * sizeof(cl_float))));
    ir_frame->format = Frame::Float;
  }

  void newDepthFrame()
  {
    depth_frame = new OpenCLFrame(static_cast<OpenCLBuffer *>(depth_buffer_allocator->allocate(IMAGE_SIZE * sizeof(cl_float))));
    depth_frame->format = Frame::Float;
  }

  bool fill_trig_table(const libfreenect2::protocol::P0TablesResponse *p0table)
  {
    if(!deviceInitialized)
    {
      LOG_ERROR << "OpenCLDepthPacketProcessor is not initialized!";
      return false;
    }

    cl_float3 *p0_table = new cl_float3[IMAGE_SIZE];

    for(int r = 0; r < 424; ++r)
    {
      cl_float3 *it = &p0_table[r * 512];
      const uint16_t *it0 = &p0table->p0table0[r * 512];
      const uint16_t *it1 = &p0table->p0table1[r * 512];
      const uint16_t *it2 = &p0table->p0table2[r * 512];
      for(int c = 0; c < 512; ++c, ++it, ++it0, ++it1, ++it2)
      {
        it->s[0] = -((float)*it0) * 0.000031 * M_PI;
        it->s[1] = -((float)*it1) * 0.000031 * M_PI;
        it->s[2] = -((float)*it2) * 0.000031 * M_PI;
        it->s[3] = 0.0f;
      }
    }

    cl::Event event;
    CHECK_CL_ON_FAIL(queue.enqueueWriteBuffer(buf_p0_table, CL_FALSE, 0, buf_p0_table_size, p0_table, NULL, &event), delete[] p0_table);
    CHECK_CL_ON_FAIL(event.wait(), delete[] p0_table);
    delete[] p0_table;
    return true;
  }

  bool fill_xz_tables(const float *xtable, const float *ztable)
  {
    if(!deviceInitialized)
    {
      LOG_ERROR << "OpenCLDepthPacketProcessor is not initialized!";
      return false;
    }

    cl::Event event0, event1;
    CHECK_CL_RETURN(queue.enqueueWriteBuffer(buf_x_table, CL_FALSE, 0, buf_x_table_size, xtable, NULL, &event0));
    CHECK_CL_RETURN(queue.enqueueWriteBuffer(buf_z_table, CL_FALSE, 0, buf_z_table_size, ztable, NULL, &event1));
    CHECK_CL_RETURN(event0.wait());
    CHECK_CL_RETURN(event1.wait());
    return true;
  }

  bool fill_lut(const short *lut)
  {
    if(!deviceInitialized)
    {
      LOG_ERROR << "OpenCLDepthPacketProcessor is not initialized!";
      return false;
    }

    cl::Event event;
    CHECK_CL_RETURN(queue.enqueueWriteBuffer(buf_lut11to16, CL_FALSE, 0, buf_lut11to16_size, lut, NULL, &event));
    CHECK_CL_RETURN(event.wait());
    return true;
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
  impl_->fill_xz_tables(xtable, ztable);
}

void OpenCLDepthPacketProcessor::loadLookupTable(const short *lut)
{
  impl_->fill_lut(lut);
}

bool OpenCLDepthPacketProcessor::good()
{
  return impl_->deviceInitialized && impl_->runtimeOk;
}

void OpenCLDepthPacketProcessor::process(const DepthPacket &packet)
{
  if (!listener_)
    return;

  if(!impl_->programInitialized && !impl_->initProgram())
  {
    impl_->runtimeOk = false;
    LOG_ERROR << "could not initialize OpenCLDepthPacketProcessor";
    return;
  }

  impl_->startTiming();

  impl_->ir_frame->timestamp = packet.timestamp;
  impl_->depth_frame->timestamp = packet.timestamp;
  impl_->ir_frame->sequence = packet.sequence;
  impl_->depth_frame->sequence = packet.sequence;

  impl_->runtimeOk = impl_->run(packet);

  impl_->stopTiming(LOG_INFO);

  if (!impl_->runtimeOk)
  {
    impl_->ir_frame->status = 1;
    impl_->depth_frame->status = 1;
  }

  if(listener_->onNewFrame(Frame::Ir, impl_->ir_frame))
    impl_->newIrFrame();
  if(listener_->onNewFrame(Frame::Depth, impl_->depth_frame))
    impl_->newDepthFrame();
}

Allocator *OpenCLDepthPacketProcessor::getAllocator()
{
  return impl_->input_buffer_allocator;
}
} /* namespace libfreenect2 */

