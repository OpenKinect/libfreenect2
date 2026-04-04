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

/** @file metal_depth_packet_processor.mm Metal GPU depth packet processor host code. */

#include <libfreenect2/depth_packet_processor.h>
#include <libfreenect2/protocol/response.h>
#include <libfreenect2/logging.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <string>
#include <dlfcn.h>
#include <cstring>

#define _USE_MATH_DEFINES
#include <math.h>

namespace libfreenect2
{

static const size_t IMAGE_SIZE = 512 * 424;
static const size_t IMAGE_WIDTH = 512;
static const size_t IMAGE_HEIGHT = 424;
static const size_t LUT_SIZE = 2048;

/** Parameters struct laid out to match the Metal shader's MetalDepthParams. */
struct MetalDepthParamsBuffer
{
  float ab_multiplier;
  float ab_multiplier_per_frq0;
  float ab_multiplier_per_frq1;
  float ab_multiplier_per_frq2;
  float ab_output_multiplier;
  float padding0[3];

  float phase_in_rad0;
  float phase_in_rad1;
  float phase_in_rad2;
  float padding1;

  float joint_bilateral_ab_threshold;
  float joint_bilateral_max_edge;
  float joint_bilateral_exp;
  float joint_bilateral_threshold;

  float gaussian_kernel0;
  float gaussian_kernel1;
  float gaussian_kernel2;
  float gaussian_kernel3;
  float gaussian_kernel4;
  float gaussian_kernel5;
  float gaussian_kernel6;
  float gaussian_kernel7;
  float gaussian_kernel8;
  float padding2[3];

  float phase_offset;
  float unambigious_dist;
  float individual_ab_threshold;
  float ab_threshold;
  float ab_confidence_slope;
  float ab_confidence_offset;
  float min_dealias_confidence;
  float max_dealias_confidence;

  float edge_ab_avg_min_value;
  float edge_ab_std_dev_threshold;
  float edge_close_delta_threshold;
  float edge_far_delta_threshold;
  float edge_max_delta_threshold;
  float edge_avg_delta_threshold;
  float max_edge_count;
  float padding3;

  float min_depth;
  float max_depth;
  float padding4[2];
};

/** Populate a MetalDepthParamsBuffer from a DepthPacketProcessor::Parameters
 *  and config values (MinDepth/MaxDepth in metres, converted to mm here). */
static void fillParamsBuffer(MetalDepthParamsBuffer &dst,
                             const DepthPacketProcessor::Parameters &p,
                             const DepthPacketProcessor::Config &cfg)
{
  dst.ab_multiplier = p.ab_multiplier;
  dst.ab_multiplier_per_frq0 = p.ab_multiplier_per_frq[0];
  dst.ab_multiplier_per_frq1 = p.ab_multiplier_per_frq[1];
  dst.ab_multiplier_per_frq2 = p.ab_multiplier_per_frq[2];
  dst.ab_output_multiplier = p.ab_output_multiplier;
  dst.padding0[0] = dst.padding0[1] = dst.padding0[2] = 0.0f;

  dst.phase_in_rad0 = p.phase_in_rad[0];
  dst.phase_in_rad1 = p.phase_in_rad[1];
  dst.phase_in_rad2 = p.phase_in_rad[2];
  dst.padding1 = 0.0f;

  dst.joint_bilateral_ab_threshold = p.joint_bilateral_ab_threshold;
  dst.joint_bilateral_max_edge = p.joint_bilateral_max_edge;
  dst.joint_bilateral_exp = p.joint_bilateral_exp;
  /* Precomputed threshold used by filterPixelStage1. */
  dst.joint_bilateral_threshold = (p.joint_bilateral_ab_threshold * p.joint_bilateral_ab_threshold)
                                  / (p.ab_multiplier * p.ab_multiplier);

  dst.gaussian_kernel0 = p.gaussian_kernel[0];
  dst.gaussian_kernel1 = p.gaussian_kernel[1];
  dst.gaussian_kernel2 = p.gaussian_kernel[2];
  dst.gaussian_kernel3 = p.gaussian_kernel[3];
  dst.gaussian_kernel4 = p.gaussian_kernel[4];
  dst.gaussian_kernel5 = p.gaussian_kernel[5];
  dst.gaussian_kernel6 = p.gaussian_kernel[6];
  dst.gaussian_kernel7 = p.gaussian_kernel[7];
  dst.gaussian_kernel8 = p.gaussian_kernel[8];
  dst.padding2[0] = dst.padding2[1] = dst.padding2[2] = 0.0f;

  dst.phase_offset = p.phase_offset;
  dst.unambigious_dist = p.unambigious_dist;
  dst.individual_ab_threshold = p.individual_ab_threshold;
  dst.ab_threshold = p.ab_threshold;
  dst.ab_confidence_slope = p.ab_confidence_slope;
  dst.ab_confidence_offset = p.ab_confidence_offset;
  dst.min_dealias_confidence = p.min_dealias_confidence;
  dst.max_dealias_confidence = p.max_dealias_confidence;

  dst.edge_ab_avg_min_value = p.edge_ab_avg_min_value;
  dst.edge_ab_std_dev_threshold = p.edge_ab_std_dev_threshold;
  dst.edge_close_delta_threshold = p.edge_close_delta_threshold;
  dst.edge_far_delta_threshold = p.edge_far_delta_threshold;
  dst.edge_max_delta_threshold = p.edge_max_delta_threshold;
  dst.edge_avg_delta_threshold = p.edge_avg_delta_threshold;
  dst.max_edge_count = p.max_edge_count;
  dst.padding3 = 0.0f;

  /* Config values are in metres; shaders expect millimetres. */
  dst.min_depth = cfg.MinDepth * 1000.0f;
  dst.max_depth = cfg.MaxDepth * 1000.0f;
  dst.padding4[0] = dst.padding4[1] = 0.0f;
}

/** PIMPL implementation struct holding all Metal objects. */
class MetalDepthPacketProcessorImpl
{
public:
  id<MTLDevice>              device;
  id<MTLCommandQueue>        command_queue;

  id<MTLComputePipelineState> pipeline_stage1;
  id<MTLComputePipelineState> pipeline_filter_stage1;
  id<MTLComputePipelineState> pipeline_stage2;
  id<MTLComputePipelineState> pipeline_filter_stage2;

  /* Static lookup/calibration buffers. */
  id<MTLBuffer> buf_lut11to16;
  id<MTLBuffer> buf_p0_table;   /* float3 (16-byte aligned), IMAGE_SIZE entries */
  id<MTLBuffer> buf_x_table;
  id<MTLBuffer> buf_z_table;

  /* Per-frame input buffer (raw USB packet). */
  id<MTLBuffer> buf_packet;

  /* Intermediate GPU buffers. */
  id<MTLBuffer> buf_a;
  id<MTLBuffer> buf_b;
  id<MTLBuffer> buf_n;
  id<MTLBuffer> buf_ir;
  id<MTLBuffer> buf_a_filtered;
  id<MTLBuffer> buf_b_filtered;
  id<MTLBuffer> buf_edge_test;
  id<MTLBuffer> buf_depth;
  id<MTLBuffer> buf_ir_sum;
  id<MTLBuffer> buf_filtered;

  /* Parameters constant buffer. */
  id<MTLBuffer> buf_params;

  /* Output frames. */
  Frame *ir_frame;
  Frame *depth_frame;

  DepthPacketProcessor::Parameters params;
  DepthPacketProcessor::Config config;

  bool device_initialized;
  bool runtime_ok;

  MetalDepthPacketProcessorImpl()
    : device(nil)
    , command_queue(nil)
    , pipeline_stage1(nil)
    , pipeline_filter_stage1(nil)
    , pipeline_stage2(nil)
    , pipeline_filter_stage2(nil)
    , buf_lut11to16(nil)
    , buf_p0_table(nil)
    , buf_x_table(nil)
    , buf_z_table(nil)
    , buf_packet(nil)
    , buf_a(nil)
    , buf_b(nil)
    , buf_n(nil)
    , buf_ir(nil)
    , buf_a_filtered(nil)
    , buf_b_filtered(nil)
    , buf_edge_test(nil)
    , buf_depth(nil)
    , buf_ir_sum(nil)
    , buf_filtered(nil)
    , buf_params(nil)
    , ir_frame(NULL)
    , depth_frame(NULL)
    , device_initialized(false)
    , runtime_ok(true)
  {
    device_initialized = init();
    if(device_initialized)
    {
      newIrFrame();
      newDepthFrame();
    }
  }

  ~MetalDepthPacketProcessorImpl()
  {
    delete ir_frame;
    delete depth_frame;

    /* ARC / manual release: set to nil to release all Metal objects. */
    buf_params = nil;
    buf_filtered = nil;
    buf_ir_sum = nil;
    buf_depth = nil;
    buf_edge_test = nil;
    buf_b_filtered = nil;
    buf_a_filtered = nil;
    buf_ir = nil;
    buf_n = nil;
    buf_b = nil;
    buf_a = nil;
    buf_packet = nil;
    buf_z_table = nil;
    buf_x_table = nil;
    buf_p0_table = nil;
    buf_lut11to16 = nil;
    pipeline_filter_stage2 = nil;
    pipeline_stage2 = nil;
    pipeline_filter_stage1 = nil;
    pipeline_stage1 = nil;
    command_queue = nil;
    device = nil;
  }

  /** Create a new (empty) IR output frame. */
  void newIrFrame()
  {
    delete ir_frame;
    ir_frame = new Frame(IMAGE_WIDTH, IMAGE_HEIGHT, 4);
    ir_frame->format = Frame::Float;
  }

  /** Create a new (empty) depth output frame. */
  void newDepthFrame()
  {
    delete depth_frame;
    depth_frame = new Frame(IMAGE_WIDTH, IMAGE_HEIGHT, 4);
    depth_frame->format = Frame::Float;
  }

  /** Allocate a shared-storage MTLBuffer (zero-copy on Apple Silicon). */
  id<MTLBuffer> makeBuffer(size_t size)
  {
    id<MTLBuffer> buf = [device newBufferWithLength:size
                                           options:MTLResourceStorageModeShared];
    if(!buf)
    {
      LOG_ERROR << "MetalDepthPacketProcessor: failed to allocate MTLBuffer of size " << size;
    }
    return buf;
  }

  /** Locate the compiled Metal library alongside the running dylib.
   *
   *  Strategy: walk up the dylib path to find the build/install directory,
   *  then look for default.metallib next to it or in a known relative path.
   *  Falls back to the executable's bundle or current working directory. */
  id<MTLLibrary> loadMetalLibrary()
  {
    NSError *error = nil;
    id<MTLLibrary> lib = nil;

    /* 1. Try to load from the same directory as this dylib. */
    Dl_info info;
    if(dladdr((void *)&loadMetalLibrary_stub, &info) && info.dli_fname)
    {
      NSString *dylib_path = [NSString stringWithUTF8String:info.dli_fname];
      NSString *dylib_dir = [dylib_path stringByDeletingLastPathComponent];
      NSString *metallib_path = [dylib_dir stringByAppendingPathComponent:@"default.metallib"];

      lib = [device newLibraryWithURL:[NSURL fileURLWithPath:metallib_path] error:&error];
      if(lib)
      {
        LOG_INFO << "MetalDepthPacketProcessor: loaded Metal library from " << [metallib_path UTF8String];
        return lib;
      }
    }

    /* 2. Try the main bundle (useful when linked into an app). */
    lib = [device newDefaultLibraryWithBundle:[NSBundle mainBundle] error:&error];
    if(lib)
    {
      LOG_INFO << "MetalDepthPacketProcessor: loaded Metal library from main bundle";
      return lib;
    }

    /* 3. Try the default library of the device (works if the .metal was compiled
     *    into the app target itself). */
    lib = [device newDefaultLibrary];
    if(lib)
    {
      LOG_INFO << "MetalDepthPacketProcessor: using device default Metal library";
      return lib;
    }

    LOG_ERROR << "MetalDepthPacketProcessor: could not find default.metallib. "
              << "Make sure it is installed alongside libfreenect2.dylib.";
    return nil;
  }

  /** Dummy static function whose address is used for dladdr dylib path lookup. */
  static void loadMetalLibrary_stub() {}

  /** Initialise the Metal device, command queue, pipelines, and buffers. */
  bool init()
  {
    @autoreleasepool
    {
      device = MTLCreateSystemDefaultDevice();
      if(!device)
      {
        LOG_ERROR << "MetalDepthPacketProcessor: no Metal device available.";
        return false;
      }

      LOG_INFO << "MetalDepthPacketProcessor: using device " << [[device name] UTF8String];

      command_queue = [device newCommandQueue];
      if(!command_queue)
      {
        LOG_ERROR << "MetalDepthPacketProcessor: failed to create command queue.";
        return false;
      }

      id<MTLLibrary> library = loadMetalLibrary();
      if(!library)
        return false;

      if(!buildPipelines(library))
        return false;

      if(!allocateBuffers())
        return false;

      return true;
    }
  }

  /** Build the four compute pipeline states from the Metal library. */
  bool buildPipelines(id<MTLLibrary> library)
  {
    NSError *error = nil;

    id<MTLFunction> fn_stage1 = [library newFunctionWithName:@"processPixelStage1"];
    if(!fn_stage1)
    {
      LOG_ERROR << "MetalDepthPacketProcessor: kernel 'processPixelStage1' not found.";
      return false;
    }
    pipeline_stage1 = [device newComputePipelineStateWithFunction:fn_stage1 error:&error];
    if(!pipeline_stage1)
    {
      LOG_ERROR << "MetalDepthPacketProcessor: failed to build processPixelStage1 pipeline: "
                << [[error localizedDescription] UTF8String];
      return false;
    }

    id<MTLFunction> fn_filter1 = [library newFunctionWithName:@"filterPixelStage1"];
    if(!fn_filter1)
    {
      LOG_ERROR << "MetalDepthPacketProcessor: kernel 'filterPixelStage1' not found.";
      return false;
    }
    pipeline_filter_stage1 = [device newComputePipelineStateWithFunction:fn_filter1 error:&error];
    if(!pipeline_filter_stage1)
    {
      LOG_ERROR << "MetalDepthPacketProcessor: failed to build filterPixelStage1 pipeline: "
                << [[error localizedDescription] UTF8String];
      return false;
    }

    id<MTLFunction> fn_stage2 = [library newFunctionWithName:@"processPixelStage2"];
    if(!fn_stage2)
    {
      LOG_ERROR << "MetalDepthPacketProcessor: kernel 'processPixelStage2' not found.";
      return false;
    }
    pipeline_stage2 = [device newComputePipelineStateWithFunction:fn_stage2 error:&error];
    if(!pipeline_stage2)
    {
      LOG_ERROR << "MetalDepthPacketProcessor: failed to build processPixelStage2 pipeline: "
                << [[error localizedDescription] UTF8String];
      return false;
    }

    id<MTLFunction> fn_filter2 = [library newFunctionWithName:@"filterPixelStage2"];
    if(!fn_filter2)
    {
      LOG_ERROR << "MetalDepthPacketProcessor: kernel 'filterPixelStage2' not found.";
      return false;
    }
    pipeline_filter_stage2 = [device newComputePipelineStateWithFunction:fn_filter2 error:&error];
    if(!pipeline_filter_stage2)
    {
      LOG_ERROR << "MetalDepthPacketProcessor: failed to build filterPixelStage2 pipeline: "
                << [[error localizedDescription] UTF8String];
      return false;
    }

    return true;
  }

  /** Allocate all Metal buffers. */
  bool allocateBuffers()
  {
    /* float3 in Metal is stored 16-byte aligned (4 floats per element). */
    const size_t float3_size = 4 * sizeof(float);

    buf_lut11to16  = makeBuffer(LUT_SIZE * sizeof(short));
    buf_p0_table   = makeBuffer(IMAGE_SIZE * float3_size);
    buf_x_table    = makeBuffer(IMAGE_SIZE * sizeof(float));
    buf_z_table    = makeBuffer(IMAGE_SIZE * sizeof(float));

    /* Raw packet: 10 sub-frames, each 424 rows * 352 ushorts. */
    buf_packet = makeBuffer(((IMAGE_SIZE * 11) / 16) * 10 * sizeof(unsigned short));

    buf_a          = makeBuffer(IMAGE_SIZE * float3_size);
    buf_b          = makeBuffer(IMAGE_SIZE * float3_size);
    buf_n          = makeBuffer(IMAGE_SIZE * float3_size);
    buf_ir         = makeBuffer(IMAGE_SIZE * sizeof(float));
    buf_a_filtered = makeBuffer(IMAGE_SIZE * float3_size);
    buf_b_filtered = makeBuffer(IMAGE_SIZE * float3_size);
    buf_edge_test  = makeBuffer(IMAGE_SIZE * sizeof(uint8_t));
    buf_depth      = makeBuffer(IMAGE_SIZE * sizeof(float));
    buf_ir_sum     = makeBuffer(IMAGE_SIZE * sizeof(float));
    buf_filtered   = makeBuffer(IMAGE_SIZE * sizeof(float));
    buf_params     = makeBuffer(sizeof(MetalDepthParamsBuffer));

    /* Verify that all allocations succeeded. */
    if(!buf_lut11to16 || !buf_p0_table || !buf_x_table || !buf_z_table ||
       !buf_packet || !buf_a || !buf_b || !buf_n || !buf_ir ||
       !buf_a_filtered || !buf_b_filtered || !buf_edge_test ||
       !buf_depth || !buf_ir_sum || !buf_filtered || !buf_params)
    {
      LOG_ERROR << "MetalDepthPacketProcessor: buffer allocation failed.";
      return false;
    }

    return true;
  }

  /** Upload processing parameters to the GPU constant buffer. */
  void uploadParams()
  {
    MetalDepthParamsBuffer *dst = (MetalDepthParamsBuffer *)[buf_params contents];
    fillParamsBuffer(*dst, params, config);
  }

  /** Dispatch one compute pass using threadgroupsWithRemainder (non-uniform). */
  void dispatchKernel(id<MTLComputeCommandEncoder> enc,
                      id<MTLComputePipelineState> pso)
  {
    /* One thread per pixel over the 512x424 image. */
    MTLSize threads_per_grid = MTLSizeMake(IMAGE_SIZE, 1, 1);
    /* 64-thread threadgroups work well on all Apple GPU generations. */
    NSUInteger tg_size = MIN(64u, [pso maxTotalThreadsPerThreadgroup]);
    MTLSize threads_per_tg = MTLSizeMake(tg_size, 1, 1);
    /* Non-uniform dispatch: handles IMAGE_SIZE not divisible by tg_size. */
    [enc dispatchThreads:threads_per_grid threadsPerThreadgroup:threads_per_tg];
  }

  /** Run the full depth processing pipeline for one packet.
   *
   *  Stages dispatched:
   *    1. processPixelStage1
   *    2. filterPixelStage1  (only when bilateral filter is enabled)
   *    3. processPixelStage2
   *    4. filterPixelStage2  (only when edge-aware filter is enabled)
   *
   *  The call blocks until all GPU work is complete so that the output
   *  frames are ready when process() returns. */
  bool run(const DepthPacket &packet)
  {
    @autoreleasepool
    {
      /* Upload raw packet data (zero-copy on unified memory). */
      memcpy([buf_packet contents], packet.buffer, MIN(packet.buffer_length, [buf_packet length]));

      /* Upload current parameters. */
      uploadParams();

      id<MTLCommandBuffer> cmd = [command_queue commandBuffer];
      if(!cmd)
      {
        LOG_ERROR << "MetalDepthPacketProcessor: failed to create command buffer.";
        return false;
      }

      /* ------------------------------------------------------------------ */
      /* Stage 1: processPixelStage1                                         */
      /* ------------------------------------------------------------------ */
      {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline_stage1];
        [enc setBuffer:buf_lut11to16 offset:0 atIndex:0];
        [enc setBuffer:buf_z_table   offset:0 atIndex:1];
        [enc setBuffer:buf_p0_table  offset:0 atIndex:2];
        [enc setBuffer:buf_packet    offset:0 atIndex:3];
        [enc setBuffer:buf_a         offset:0 atIndex:4];
        [enc setBuffer:buf_b         offset:0 atIndex:5];
        [enc setBuffer:buf_n         offset:0 atIndex:6];
        [enc setBuffer:buf_ir        offset:0 atIndex:7];
        [enc setBuffer:buf_params    offset:0 atIndex:8];
        dispatchKernel(enc, pipeline_stage1);
        [enc endEncoding];
      }

      /* ------------------------------------------------------------------ */
      /* Stage 1 filter (optional)                                           */
      /* ------------------------------------------------------------------ */
      if(config.EnableBilateralFilter)
      {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline_filter_stage1];
        [enc setBuffer:buf_a           offset:0 atIndex:0];
        [enc setBuffer:buf_b           offset:0 atIndex:1];
        [enc setBuffer:buf_n           offset:0 atIndex:2];
        [enc setBuffer:buf_a_filtered  offset:0 atIndex:3];
        [enc setBuffer:buf_b_filtered  offset:0 atIndex:4];
        [enc setBuffer:buf_edge_test   offset:0 atIndex:5];
        [enc setBuffer:buf_params      offset:0 atIndex:6];
        dispatchKernel(enc, pipeline_filter_stage1);
        [enc endEncoding];
      }

      /* ------------------------------------------------------------------ */
      /* Stage 2: processPixelStage2                                         */
      /* ------------------------------------------------------------------ */
      {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline_stage2];
        /* Use filtered a/b if bilateral filter was run, otherwise raw. */
        id<MTLBuffer> a_src = config.EnableBilateralFilter ? buf_a_filtered : buf_a;
        id<MTLBuffer> b_src = config.EnableBilateralFilter ? buf_b_filtered : buf_b;
        [enc setBuffer:a_src       offset:0 atIndex:0];
        [enc setBuffer:b_src       offset:0 atIndex:1];
        [enc setBuffer:buf_x_table offset:0 atIndex:2];
        [enc setBuffer:buf_z_table offset:0 atIndex:3];
        [enc setBuffer:buf_depth   offset:0 atIndex:4];
        [enc setBuffer:buf_ir_sum  offset:0 atIndex:5];
        [enc setBuffer:buf_params  offset:0 atIndex:6];
        dispatchKernel(enc, pipeline_stage2);
        [enc endEncoding];
      }

      /* ------------------------------------------------------------------ */
      /* Stage 2 filter (optional)                                           */
      /* ------------------------------------------------------------------ */
      if(config.EnableEdgeAwareFilter)
      {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipeline_filter_stage2];
        [enc setBuffer:buf_depth      offset:0 atIndex:0];
        [enc setBuffer:buf_ir_sum     offset:0 atIndex:1];
        [enc setBuffer:buf_edge_test  offset:0 atIndex:2];
        [enc setBuffer:buf_filtered   offset:0 atIndex:3];
        [enc setBuffer:buf_params     offset:0 atIndex:4];
        dispatchKernel(enc, pipeline_filter_stage2);
        [enc endEncoding];
      }

      /* Commit and wait — process() must return with frames ready. */
      [cmd commit];
      [cmd waitUntilCompleted];

      if([cmd status] == MTLCommandBufferStatusError)
      {
        LOG_ERROR << "MetalDepthPacketProcessor: command buffer execution error: "
                  << [[[cmd error] localizedDescription] UTF8String];
        return false;
      }

      /* Copy results into output frames (zero-copy on shared storage). */
      const float *depth_src = config.EnableEdgeAwareFilter
                               ? (const float *)[buf_filtered contents]
                               : (const float *)[buf_depth contents];
      const float *ir_src = (const float *)[buf_ir contents];

      memcpy(ir_frame->data,    ir_src,    IMAGE_SIZE * sizeof(float));
      memcpy(depth_frame->data, depth_src, IMAGE_SIZE * sizeof(float));

      return true;
    }
  }
};

/* -------------------------------------------------------------------------- */
/* MetalDepthPacketProcessor public interface                                  */
/* -------------------------------------------------------------------------- */

MetalDepthPacketProcessor::MetalDepthPacketProcessor(const int /*deviceIndex*/)
  : impl_(new MetalDepthPacketProcessorImpl())
{
}

MetalDepthPacketProcessor::~MetalDepthPacketProcessor()
{
  delete impl_;
}

void MetalDepthPacketProcessor::setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config)
{
  DepthPacketProcessor::setConfiguration(config);
  impl_->config = config;
  /* Re-upload parameters on next process() call via uploadParams(). */
}

void MetalDepthPacketProcessor::loadP0TablesFromCommandResponse(unsigned char *buffer,
                                                                 size_t buffer_length)
{
  if(!impl_->device_initialized)
  {
    LOG_ERROR << "MetalDepthPacketProcessor: not initialized.";
    return;
  }

  libfreenect2::protocol::P0TablesResponse *p0table =
      (libfreenect2::protocol::P0TablesResponse *)buffer;

  if(buffer_length < sizeof(libfreenect2::protocol::P0TablesResponse))
  {
    LOG_ERROR << "P0Table response too short!";
    return;
  }

  /* Convert uint16 p0 values to float radians and pack into float3 (float4)
   * layout matching the Metal shader buffer expectation. */
  float *p0_dst = (float *)[impl_->buf_p0_table contents];

  for(int r = 0; r < 424; ++r)
  {
    float *it = p0_dst + r * 512 * 4; /* 4 floats per float3 slot */
    const uint16_t *it0 = &p0table->p0table0[r * 512];
    const uint16_t *it1 = &p0table->p0table1[r * 512];
    const uint16_t *it2 = &p0table->p0table2[r * 512];

    for(int c = 0; c < 512; ++c, it += 4, ++it0, ++it1, ++it2)
    {
      it[0] = -((float)*it0) * 0.000031f * (float)M_PI;
      it[1] = -((float)*it1) * 0.000031f * (float)M_PI;
      it[2] = -((float)*it2) * 0.000031f * (float)M_PI;
      it[3] = 0.0f;
    }
  }
}

void MetalDepthPacketProcessor::loadXZTables(const float *xtable, const float *ztable)
{
  if(!impl_->device_initialized)
  {
    LOG_ERROR << "MetalDepthPacketProcessor: not initialized.";
    return;
  }

  memcpy([impl_->buf_x_table contents], xtable, TABLE_SIZE * sizeof(float));
  memcpy([impl_->buf_z_table contents], ztable, TABLE_SIZE * sizeof(float));
}

void MetalDepthPacketProcessor::loadLookupTable(const short *lut)
{
  if(!impl_->device_initialized)
  {
    LOG_ERROR << "MetalDepthPacketProcessor: not initialized.";
    return;
  }

  memcpy([impl_->buf_lut11to16 contents], lut, LUT_SIZE * sizeof(short));
}

bool MetalDepthPacketProcessor::good()
{
  return impl_->device_initialized && impl_->runtime_ok;
}

void MetalDepthPacketProcessor::process(const DepthPacket &packet)
{
  if(!listener_)
    return;

  if(!impl_->device_initialized)
  {
    LOG_ERROR << "MetalDepthPacketProcessor: not initialized, dropping packet.";
    return;
  }

  impl_->ir_frame->timestamp = packet.timestamp;
  impl_->depth_frame->timestamp = packet.timestamp;
  impl_->ir_frame->sequence = packet.sequence;
  impl_->depth_frame->sequence = packet.sequence;

  impl_->runtime_ok = impl_->run(packet);

  if(!impl_->runtime_ok)
  {
    impl_->ir_frame->status = 1;
    impl_->depth_frame->status = 1;
  }

  if(listener_->onNewFrame(Frame::Ir, impl_->ir_frame))
  {
    impl_->ir_frame = NULL;   // listener took ownership; don't let newIrFrame() delete it
    impl_->newIrFrame();
  }
  if(listener_->onNewFrame(Frame::Depth, impl_->depth_frame))
  {
    impl_->depth_frame = NULL; // listener took ownership; don't let newDepthFrame() delete it
    impl_->newDepthFrame();
  }
}

} /* namespace libfreenect2 */
