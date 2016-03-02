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
#include <libfreenect2/protocol/response.h>
#include "libfreenect2/logging.h"

#include <helper_math.h>
#include <math_constants.h>

__constant__ static unsigned int BFI_BITMASK;
__constant__ static float AB_MULTIPLIER;
__constant__ static float AB_MULTIPLIER_PER_FRQ0;
__constant__ static float AB_MULTIPLIER_PER_FRQ1;
__constant__ static float AB_MULTIPLIER_PER_FRQ2;
__constant__ static float AB_OUTPUT_MULTIPLIER;
;
__constant__ static float PHASE_IN_RAD0;
__constant__ static float PHASE_IN_RAD1;
__constant__ static float PHASE_IN_RAD2;
;
__constant__ static float JOINT_BILATERAL_AB_THRESHOLD;
__constant__ static float JOINT_BILATERAL_MAX_EDGE;
__constant__ static float JOINT_BILATERAL_EXP;
__constant__ static float JOINT_BILATERAL_THRESHOLD;
;
__constant__ static float GAUSSIAN_KERNEL_0;
__constant__ static float GAUSSIAN_KERNEL_1;
__constant__ static float GAUSSIAN_KERNEL_2;
__constant__ static float GAUSSIAN_KERNEL_3;
__constant__ static float GAUSSIAN_KERNEL_4;
__constant__ static float GAUSSIAN_KERNEL_5;
__constant__ static float GAUSSIAN_KERNEL_6;
__constant__ static float GAUSSIAN_KERNEL_7;
__constant__ static float GAUSSIAN_KERNEL_8;
;
__constant__ static float PHASE_OFFSET;
__constant__ static float UNAMBIGIOUS_DIST;
__constant__ static float INDIVIDUAL_AB_THRESHOLD;
__constant__ static float AB_THRESHOLD;
__constant__ static float AB_CONFIDENCE_SLOPE;
__constant__ static float AB_CONFIDENCE_OFFSET;
__constant__ static float MIN_DEALIAS_CONFIDENCE;
__constant__ static float MAX_DEALIAS_CONFIDENCE;
;
__constant__ static float EDGE_AB_AVG_MIN_VALUE;
__constant__ static float EDGE_AB_STD_DEV_THRESHOLD;
__constant__ static float EDGE_CLOSE_DELTA_THRESHOLD;
__constant__ static float EDGE_FAR_DELTA_THRESHOLD;
__constant__ static float EDGE_MAX_DELTA_THRESHOLD;
__constant__ static float EDGE_AVG_DELTA_THRESHOLD;
__constant__ static float MAX_EDGE_COUNT;
;
__constant__ static float MIN_DEPTH;
__constant__ static float MAX_DEPTH;

#define sqrt(x) sqrtf(x)
#define sincos(x, a, b) sincosf(x, a, b)
#define atan2(a, b) atan2f(a, b)
#define log(x) logf(x)
#define exp(x) expf(x)
#define max(x, y) fmaxf(x, y)
#define min(x, y) fminf(x, y)
#define M_PI_F CUDART_PI_F
#ifndef M_PI
#define M_PI CUDART_PI
#endif

typedef unsigned char uchar;

inline  __device__ uint get_global_id(uint i)
{
  if (i == 0)
    return blockIdx.x*blockDim.x + threadIdx.x;
  // NOT IMPLEMENTED for i > 0
  return 0;
}

static inline __device__ int3 isnan(float3 v)
{
  return make_int3(isnan(v.x) ? -1 : 0, isnan(v.y) ? -1 : 0, isnan(v.z) ? -1 : 0);
}
static inline __device__ float3 sqrtf(float3 v)
{
  return make_float3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z));
}
static inline __device__ void sincosf(float3 v, float3 *a, float3 *b)
{
  sincosf(v.x, &a->x, &b->x);
  sincosf(v.y, &a->y, &b->y);
  sincosf(v.z, &a->z, &b->z);
}
static inline __device__ float3 atan2f(float3 a, float3 b)
{
  return make_float3(atan2f(a.x, b.x), atan2f(a.y, b.y), atan2f(a.z, b.z));
}
static inline __device__ float3 expf(float3 v)
{
  return make_float3(expf(v.x), expf(v.y), expf(v.z));
}
static inline __device__ float3 select(float3 a, float3 b, int3 c)
{
  return make_float3(c.x < 0 ? b.x : a.x, c.y < 0 ? b.y : a.y, c.z < 0 ? b.z : a.z);
}
static inline __device__ int3 isless(float3 a, float3 b)
{
  return make_int3(a.x < b.x ? -1 : 0, a.y < b.y ? -1 : 0, a.z < b.z ? -1 : 0);
}
static inline __device__ int3 isequal(float3 a, float3 b)
{
  return make_int3(a.x == b.x ? -1 : 0, a.y == b.y ? -1 : 0, a.z == b.z ? -1 : 0);
}
static inline __device__ int any(int3 v)
{
  return (v.x | v.y | v.z) < 0;
}
static inline __device__ int all(int3 v)
{
  return (v.x & v.y & v.z) < 0;
}

/*******************************************************************************
 * Process pixel stage 1
 ******************************************************************************/

static __device__
float decodePixelMeasurement(const ushort* __restrict__ data, const short* __restrict__ lut11to16, const uint sub, const uint x, const uint y)
{
  uint row_idx = (424 * sub + (y < 212 ? y + 212 : 423 - y)) * 352;
  uint idx = (((x >> 2) + ((x << 7) & BFI_BITMASK)) * 11) & (uint)0xffffffff;

  uint col_idx = idx >> 4;
  uint upper_bytes = idx & 15;
  uint lower_bytes = 16 - upper_bytes;

  uint data_idx0 = row_idx + col_idx;
  uint data_idx1 = row_idx + col_idx + 1;

  return (float)lut11to16[(x < 1 || 510 < x || col_idx > 352) ? 0 : ((data[data_idx0] >> upper_bytes) | (data[data_idx1] << lower_bytes)) & 2047];
}

static __device__
float2 processMeasurementTriple(const float ab_multiplier_per_frq, const float p0, const float3 v, int *invalid)
{
  float3 p0vec = make_float3(p0 + PHASE_IN_RAD0, p0 + PHASE_IN_RAD1, p0 + PHASE_IN_RAD2);
  float3 p0sin, p0cos;
  sincos(p0vec, &p0sin, &p0cos);

  *invalid = *invalid && any(isequal(v, make_float3(32767.0f)));

  return make_float2(dot(v, p0cos), -dot(v, p0sin)) * ab_multiplier_per_frq;
}

static __global__
void processPixelStage1(const short* __restrict__ lut11to16, const float* __restrict__ z_table, const float4* __restrict__ p0_table, const ushort* __restrict__ data,
                               float4 *a_out, float4 *b_out, float4 *n_out, float *ir_out)
{
  const uint i = get_global_id(0);

  const uint x = i % 512;
  const uint y = i / 512;

  const uint y_in = (423 - y);

  const float zmultiplier = z_table[i];
  int valid = (int)(0.0f < zmultiplier);
  int saturatedX = valid;
  int saturatedY = valid;
  int saturatedZ = valid;
  int3 invalid_pixel = make_int3((int)(!valid));
  const float3 p0 = make_float3(p0_table[i]);

  const float3 v0 = make_float3(decodePixelMeasurement(data, lut11to16, 0, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 1, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 2, x, y_in));
  const float2 ab0 = processMeasurementTriple(AB_MULTIPLIER_PER_FRQ0, p0.x, v0, &saturatedX);

  const float3 v1 = make_float3(decodePixelMeasurement(data, lut11to16, 3, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 4, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 5, x, y_in));
  const float2 ab1 = processMeasurementTriple(AB_MULTIPLIER_PER_FRQ1, p0.y, v1, &saturatedY);

  const float3 v2 = make_float3(decodePixelMeasurement(data, lut11to16, 6, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 7, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 8, x, y_in));
  const float2 ab2 = processMeasurementTriple(AB_MULTIPLIER_PER_FRQ2, p0.z, v2, &saturatedZ);

  float3 a = select(make_float3(ab0.x, ab1.x, ab2.x), make_float3(0.0f), invalid_pixel);
  float3 b = select(make_float3(ab0.y, ab1.y, ab2.y), make_float3(0.0f), invalid_pixel);
  float3 n = sqrt(a * a + b * b);

  int3 saturated = make_int3(saturatedX, saturatedY, saturatedZ);
  a = select(a, make_float3(0.0f), saturated);
  b = select(b, make_float3(0.0f), saturated);

  a_out[i] = make_float4(a);
  b_out[i] = make_float4(b);
  n_out[i] = make_float4(n);
  ir_out[i] = min(dot(select(n, make_float3(65535.0f), saturated), make_float3(0.333333333f  * AB_MULTIPLIER * AB_OUTPUT_MULTIPLIER)), 65535.0f);
}

/*******************************************************************************
 * Filter pixel stage 1
 ******************************************************************************/
static __global__
void filterPixelStage1(const float4* __restrict__ a, const float4* __restrict__ b, const float4* __restrict__ n,
                              float4 *a_out, float4 *b_out, uchar *max_edge_test)
{
  const uint i = get_global_id(0);

  const uint x = i % 512;
  const uint y = i / 512;

  const float3 self_a = make_float3(a[i]);
  const float3 self_b = make_float3(b[i]);

  const float gaussian[9] = {GAUSSIAN_KERNEL_0, GAUSSIAN_KERNEL_1, GAUSSIAN_KERNEL_2, GAUSSIAN_KERNEL_3, GAUSSIAN_KERNEL_4, GAUSSIAN_KERNEL_5, GAUSSIAN_KERNEL_6, GAUSSIAN_KERNEL_7, GAUSSIAN_KERNEL_8};

  if(x < 1 || y < 1 || x > 510 || y > 422)
  {
    a_out[i] = make_float4(self_a);
    b_out[i] = make_float4(self_b);
    max_edge_test[i] = 1;
  }
  else
  {
    float3 threshold = make_float3(sqrt(JOINT_BILATERAL_THRESHOLD));
    float3 joint_bilateral_exp = make_float3(JOINT_BILATERAL_EXP);

    const float3 self_norm = make_float3(n[i]);
    const float3 self_normalized_a = self_a / self_norm;
    const float3 self_normalized_b = self_b / self_norm;

    float3 weight_acc = make_float3(0.0f);
    float3 weighted_a_acc = make_float3(0.0f);
    float3 weighted_b_acc = make_float3(0.0f);
    float3 dist_acc = make_float3(0.0f);

    const int3 c0 = isless(self_norm, threshold);

    threshold = select(threshold, make_float3(0.0f), c0);
    joint_bilateral_exp = select(joint_bilateral_exp, make_float3(0.0f), c0);

    for(int yi = -1, j = 0; yi < 2; ++yi)
    {
      uint i_other = (y + yi) * 512 + x - 1;

      for(int xi = -1; xi < 2; ++xi, ++j, ++i_other)
      {
        const float3 other_a = make_float3(a[i_other]);
        const float3 other_b = make_float3(b[i_other]);
        const float3 other_norm = make_float3(n[i_other]);
        const float3 other_normalized_a = other_a / other_norm;
        const float3 other_normalized_b = other_b / other_norm;

        const int3 c1 = isless(other_norm, threshold);

        const float3 dist = 0.5f * (1.0f - (self_normalized_a * other_normalized_a + self_normalized_b * other_normalized_b));
        const float3 weight = select(gaussian[j] * exp(-1.442695f * joint_bilateral_exp * dist), make_float3(0.0f), c1);

        weighted_a_acc += weight * other_a;
        weighted_b_acc += weight * other_b;
        weight_acc += weight;
        dist_acc += select(dist, make_float3(0.0f), c1);
      }
    }

    const int3 c2 = isless(make_float3(0.0f), weight_acc);
    a_out[i] = make_float4(select(make_float3(0.0f), weighted_a_acc / weight_acc, c2));
    b_out[i] = make_float4(select(make_float3(0.0f), weighted_b_acc / weight_acc, c2));

    max_edge_test[i] = all(isless(dist_acc, make_float3(JOINT_BILATERAL_MAX_EDGE)));
  }
}

/*******************************************************************************
 * Process pixel stage 2
 ******************************************************************************/
static __global__
void processPixelStage2(const float4* __restrict__ a_in, const float4* __restrict__ b_in, const float* __restrict__ x_table, const float* __restrict__ z_table,
                               float *depth, float *ir_sums)
{
  const uint i = get_global_id(0);
  float3 a = make_float3(a_in[i]);
  float3 b = make_float3(b_in[i]);

  float3 phase = atan2(b, a);
  phase = select(phase, phase + 2.0f * M_PI_F, isless(phase, make_float3(0.0f)));
  phase = select(phase, make_float3(0.0f), isnan(phase));
  float3 ir = sqrt(a * a + b * b) * AB_MULTIPLIER;

  float ir_sum = ir.x + ir.y + ir.z;
  float ir_min = min(ir.x, min(ir.y, ir.z));
  float ir_max = max(ir.x, max(ir.y, ir.z));

  float phase_final = 0;

  if(ir_min >= INDIVIDUAL_AB_THRESHOLD && ir_sum >= AB_THRESHOLD)
  {
    float3 t = phase / (2.0f * M_PI_F) * make_float3(3.0f, 15.0f, 2.0f);

    float t0 = t.x;
    float t1 = t.y;
    float t2 = t.z;

    float t5 = (floor((t1 - t0) * 0.333333f + 0.5f) * 3.0f + t0);
    float t3 = (-t2 + t5);
    float t4 = t3 * 2.0f;

    bool c1 = t4 >= -t4; // true if t4 positive

    float f1 = c1 ? 2.0f : -2.0f;
    float f2 = c1 ? 0.5f : -0.5f;
    t3 *= f2;
    t3 = (t3 - floor(t3)) * f1;

    bool c2 = 0.5f < fabs(t3) && fabs(t3) < 1.5f;

    float t6 = c2 ? t5 + 15.0f : t5;
    float t7 = c2 ? t1 + 15.0f : t1;

    float t8 = (floor((-t2 + t6) * 0.5f + 0.5f) * 2.0f + t2) * 0.5f;

    t6 *= 0.333333f; // = / 3
    t7 *= 0.066667f; // = / 15

    float t9 = (t8 + t6 + t7); // transformed phase measurements (they are transformed and divided by the values the original values were multiplied with)
    float t10 = t9 * 0.333333f; // some avg

    t6 *= 2.0f * M_PI_F;
    t7 *= 2.0f * M_PI_F;
    t8 *= 2.0f * M_PI_F;

    // some cross product
    float t8_new = t7 * 0.826977f - t8 * 0.110264f;
    float t6_new = t8 * 0.551318f - t6 * 0.826977f;
    float t7_new = t6 * 0.110264f - t7 * 0.551318f;

    t8 = t8_new;
    t6 = t6_new;
    t7 = t7_new;

    float norm = t8 * t8 + t6 * t6 + t7 * t7;
    float mask = t9 >= 0.0f ? 1.0f : 0.0f;
    t10 *= mask;

    bool slope_positive = 0 < AB_CONFIDENCE_SLOPE;

    float ir_x = slope_positive ? ir_min : ir_max;

    ir_x = log(ir_x);
    ir_x = (ir_x * AB_CONFIDENCE_SLOPE * 0.301030f + AB_CONFIDENCE_OFFSET) * 3.321928f;
    ir_x = exp(ir_x);
    ir_x = clamp(ir_x, MIN_DEALIAS_CONFIDENCE, MAX_DEALIAS_CONFIDENCE);
    ir_x *= ir_x;

    float mask2 = ir_x >= norm ? 1.0f : 0.0f;

    float t11 = t10 * mask2;

    float mask3 = MAX_DEALIAS_CONFIDENCE * MAX_DEALIAS_CONFIDENCE >= norm ? 1.0f : 0.0f;
    t10 *= mask3;
    phase_final = true/*(modeMask & 2) != 0*/ ? t11 : t10;
  }

  float zmultiplier = z_table[i];
  float xmultiplier = x_table[i];

  phase_final = 0.0f < phase_final ? phase_final + PHASE_OFFSET : phase_final;

  float depth_linear = zmultiplier * phase_final;
  float max_depth = phase_final * UNAMBIGIOUS_DIST * 2.0;

  bool cond1 = /*(modeMask & 32) != 0*/ true && 0.0f < depth_linear && 0.0f < max_depth;

  xmultiplier = (xmultiplier * 90.0) / (max_depth * max_depth * 8192.0);

  float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1);
  depth_fit = depth_fit < 0.0f ? 0.0f : depth_fit;

  float d = cond1 ? depth_fit : depth_linear; // r1.y -> later r2.z
  depth[i] = d;
  ir_sums[i] = ir_sum;
}

/*******************************************************************************
 * Filter pixel stage 2
 ******************************************************************************/
static __global__
void filterPixelStage2(const float* __restrict__ depth, const float* __restrict__ ir_sums, const uchar* __restrict__ max_edge_test, float *filtered)
{
  const uint i = get_global_id(0);

  const uint x = i % 512;
  const uint y = i / 512;

  const float raw_depth = depth[i];
  const float ir_sum = ir_sums[i];
  const uchar edge_test = max_edge_test[i];

  if(raw_depth >= MIN_DEPTH && raw_depth <= MAX_DEPTH)
  {
    if(x < 1 || y < 1 || x > 510 || y > 422)
    {
      filtered[i] = raw_depth;
    }
    else
    {
      float ir_sum_acc = ir_sum;
      float squared_ir_sum_acc = ir_sum * ir_sum;
      float min_depth = raw_depth;
      float max_depth = raw_depth;

      for(int yi = -1; yi < 2; ++yi)
      {
        uint i_other = (y + yi) * 512 + x - 1;

        for(int xi = -1; xi < 2; ++xi, ++i_other)
        {
          if(i_other == i)
          {
            continue;
          }

          const float raw_depth_other = depth[i_other];
          const float ir_sum_other = ir_sums[i_other];

          ir_sum_acc += ir_sum_other;
          squared_ir_sum_acc += ir_sum_other * ir_sum_other;

          if(0.0f < raw_depth_other)
          {
            min_depth = min(min_depth, raw_depth_other);
            max_depth = max(max_depth, raw_depth_other);
          }
        }
      }

      float tmp0 = sqrt(squared_ir_sum_acc * 9.0f - ir_sum_acc * ir_sum_acc) / 9.0f;
      float edge_avg = max(ir_sum_acc / 9.0f, EDGE_AB_AVG_MIN_VALUE);
      tmp0 /= edge_avg;

      float abs_min_diff = fabs(raw_depth - min_depth);
      float abs_max_diff = fabs(raw_depth - max_depth);

      float avg_diff = (abs_min_diff + abs_max_diff) * 0.5f;
      float max_abs_diff = max(abs_min_diff, abs_max_diff);

      bool cond0 =
          0.0f < raw_depth &&
          tmp0 >= EDGE_AB_STD_DEV_THRESHOLD &&
          EDGE_CLOSE_DELTA_THRESHOLD < abs_min_diff &&
          EDGE_FAR_DELTA_THRESHOLD < abs_max_diff &&
          EDGE_MAX_DELTA_THRESHOLD < max_abs_diff &&
          EDGE_AVG_DELTA_THRESHOLD < avg_diff;

      if(!cond0)
      {
        if(edge_test != 0)
        {
          //float tmp1 = 1500.0f > raw_depth ? 30.0f : 0.02f * raw_depth;
          float edge_count = 0.0f;

          filtered[i] = edge_count > MAX_EDGE_COUNT ? 0.0f : raw_depth;
        }
        else
        {
          filtered[i] = 0.0f;
        }
      }
      else
      {
        filtered[i] = 0.0f;
      }
    }
  }
  else
  {
    filtered[i] = 0.0f;
  }
}

#define CHECK_CUDA(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { LOG_ERROR << #expr ": " << cudaGetErrorString(err); return false; } } while(0)
#define CALL_CUDA(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { LOG_ERROR << #expr ": " << cudaGetErrorString(err); } } while(0)

namespace libfreenect2
{

class CudaFrame: public Frame
{
public:
  CudaFrame(Buffer *buffer):
    Frame(512, 424, 4, (unsigned char*)-1)
  {
    data = buffer->data;
    rawdata = reinterpret_cast<unsigned char *>(buffer);
  }

  virtual ~CudaFrame()
  {
    Buffer *buffer = reinterpret_cast<Buffer*>(rawdata);
    buffer->allocator->free(buffer);
    rawdata = NULL;
  }
};

class CudaAllocator: public Allocator
{
private:
  const bool input;

  bool allocate_cuda(Buffer *b, size_t size)
  {
    unsigned int flags = cudaHostAllocPortable;
    if (input)
      flags |= cudaHostAllocWriteCombined;
    CHECK_CUDA(cudaHostAlloc(&b->data, size, flags));
    b->length = 0;
    b->capacity = size;
    return true;
  }

public:
  CudaAllocator(bool input): input(input) {}

  virtual Buffer *allocate(size_t size)
  {
    Buffer *b = new Buffer();
    if (!allocate_cuda(b, size))
      b->data = NULL;
    return b;
  }

  virtual void free(Buffer *b)
  {
    if (b == NULL)
      return;
    if (b->data)
      CALL_CUDA(cudaFreeHost(b->data));
    delete b;
  }
};

class CudaDepthPacketProcessorImpl: public WithPerfLogging
{
public:
  static const size_t IMAGE_SIZE = 512*424;
  static const size_t LUT_SIZE = 2048;

  size_t d_lut_size;
  size_t d_xtable_size;
  size_t d_ztable_size;
  size_t d_p0table_size;

  short *d_lut;
  float *d_xtable;
  float *d_ztable;
  float4 *d_p0table;
  float4 h_p0table[IMAGE_SIZE];

  size_t d_packet_size;
  unsigned short *d_packet;

  float4 *d_a;
  float4 *d_b;
  float4 *d_n;
  float *d_ir;
  float4 *d_a_filtered;
  float4 *d_b_filtered;
  unsigned char *d_edge_test;
  float *d_depth;
  float *d_ir_sum;
  float *d_filtered;

  size_t block_size;
  size_t grid_size;

  DepthPacketProcessor::Config config;
  DepthPacketProcessor::Parameters params;

  Frame *ir_frame, *depth_frame;

  Allocator *input_allocator;
  Allocator *ir_allocator;
  Allocator *depth_allocator;

  bool good;

  CudaDepthPacketProcessorImpl(const int deviceId):
    block_size(128),
    grid_size(IMAGE_SIZE/block_size),
    config(),
    params(),
    ir_frame(NULL),
    depth_frame(NULL),
    input_allocator(NULL),
    ir_allocator(NULL),
    depth_allocator(NULL)
  {
    good = initDevice(deviceId);
    if (!good)
      return;

    input_allocator = new PoolAllocator(new CudaAllocator(true));
    ir_allocator = new PoolAllocator(new CudaAllocator(false));
    depth_allocator = new PoolAllocator(new CudaAllocator(false));

    newIrFrame();
    newDepthFrame();
  }

  ~CudaDepthPacketProcessorImpl()
  {
    delete ir_frame;
    delete depth_frame;
    delete input_allocator;
    delete ir_allocator;
    delete depth_allocator;
    if (good)
      freeDeviceMemory();
  }

  bool setParameters(const DepthPacketProcessor::Parameters &params)
  {
    unsigned int bfi_bitmask = 0x180;
    cudaMemcpyToSymbolAsync(BFI_BITMASK, &bfi_bitmask, sizeof(int));

    #define COPY(upper, lower) cudaMemcpyToSymbolAsync(upper, &params.lower, sizeof(params.lower));
    COPY(AB_MULTIPLIER, ab_multiplier)
    COPY(AB_MULTIPLIER_PER_FRQ0, ab_multiplier_per_frq[0])
    COPY(AB_MULTIPLIER_PER_FRQ1, ab_multiplier_per_frq[1])
    COPY(AB_MULTIPLIER_PER_FRQ2, ab_multiplier_per_frq[2])
    COPY(AB_OUTPUT_MULTIPLIER, ab_output_multiplier)

    COPY(PHASE_IN_RAD0, phase_in_rad[0])
    COPY(PHASE_IN_RAD1, phase_in_rad[1])
    COPY(PHASE_IN_RAD2, phase_in_rad[2])

    COPY(JOINT_BILATERAL_AB_THRESHOLD, joint_bilateral_ab_threshold)
    COPY(JOINT_BILATERAL_MAX_EDGE, joint_bilateral_max_edge)
    COPY(JOINT_BILATERAL_EXP, joint_bilateral_exp)
    float joint_bilateral_threshold;
    joint_bilateral_threshold = (params.joint_bilateral_ab_threshold * params.joint_bilateral_ab_threshold) / (params.ab_multiplier * params.ab_multiplier);
    cudaMemcpyToSymbolAsync(JOINT_BILATERAL_THRESHOLD, &joint_bilateral_threshold, sizeof(float));

    COPY(GAUSSIAN_KERNEL_0, gaussian_kernel[0])
    COPY(GAUSSIAN_KERNEL_1, gaussian_kernel[1])
    COPY(GAUSSIAN_KERNEL_2, gaussian_kernel[2])
    COPY(GAUSSIAN_KERNEL_3, gaussian_kernel[3])
    COPY(GAUSSIAN_KERNEL_4, gaussian_kernel[4])
    COPY(GAUSSIAN_KERNEL_5, gaussian_kernel[5])
    COPY(GAUSSIAN_KERNEL_6, gaussian_kernel[6])
    COPY(GAUSSIAN_KERNEL_7, gaussian_kernel[7])
    COPY(GAUSSIAN_KERNEL_8, gaussian_kernel[8])

    COPY(PHASE_OFFSET, phase_offset)
    COPY(UNAMBIGIOUS_DIST, unambigious_dist)
    COPY(INDIVIDUAL_AB_THRESHOLD, individual_ab_threshold)
    COPY(AB_THRESHOLD, ab_threshold)
    COPY(AB_CONFIDENCE_SLOPE, ab_confidence_slope)
    COPY(AB_CONFIDENCE_OFFSET, ab_confidence_offset)
    COPY(MIN_DEALIAS_CONFIDENCE, min_dealias_confidence)
    COPY(MAX_DEALIAS_CONFIDENCE, max_dealias_confidence)

    COPY(EDGE_AB_AVG_MIN_VALUE, edge_ab_avg_min_value)
    COPY(EDGE_AB_STD_DEV_THRESHOLD, edge_ab_std_dev_threshold)
    COPY(EDGE_CLOSE_DELTA_THRESHOLD, edge_close_delta_threshold)
    COPY(EDGE_FAR_DELTA_THRESHOLD, edge_far_delta_threshold)
    COPY(EDGE_MAX_DELTA_THRESHOLD, edge_max_delta_threshold)
    COPY(EDGE_AVG_DELTA_THRESHOLD, edge_avg_delta_threshold)
    COPY(MAX_EDGE_COUNT, max_edge_count)

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaGetLastError());
    return true;
  }

  bool allocateDeviceMemory()
  {
    //Read only
    d_p0table_size = IMAGE_SIZE * sizeof(float4);
    d_xtable_size = IMAGE_SIZE * sizeof(float);
    d_ztable_size = IMAGE_SIZE * sizeof(float);
    d_lut_size = LUT_SIZE * sizeof(short);

    CHECK_CUDA(cudaMalloc(&d_p0table, d_p0table_size));
    CHECK_CUDA(cudaMalloc(&d_xtable, d_xtable_size));
    CHECK_CUDA(cudaMalloc(&d_ztable, d_ztable_size));
    CHECK_CUDA(cudaMalloc(&d_lut, d_lut_size));

    d_packet_size = (IMAGE_SIZE * 11 / 8) * 10;

    CHECK_CUDA(cudaMalloc(&d_packet, d_packet_size));

    //Read-Write
    size_t d_a_size = IMAGE_SIZE * sizeof(float4);
    size_t d_b_size = IMAGE_SIZE * sizeof(float4);
    size_t d_n_size = IMAGE_SIZE * sizeof(float4);
    size_t d_ir_size = IMAGE_SIZE * sizeof(float);
    size_t d_a_filtered_size = IMAGE_SIZE * sizeof(float4);
    size_t d_b_filtered_size = IMAGE_SIZE * sizeof(float4);
    size_t d_edge_test_size = IMAGE_SIZE * sizeof(char);
    size_t d_depth_size = IMAGE_SIZE * sizeof(float);
    size_t d_ir_sum_size = IMAGE_SIZE * sizeof(float);
    size_t d_filtered_size = IMAGE_SIZE * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_a, d_a_size));
    CHECK_CUDA(cudaMalloc(&d_b, d_b_size));
    CHECK_CUDA(cudaMalloc(&d_n, d_n_size));
    CHECK_CUDA(cudaMalloc(&d_ir, d_ir_size));
    CHECK_CUDA(cudaMalloc(&d_a_filtered, d_a_filtered_size));
    CHECK_CUDA(cudaMalloc(&d_b_filtered, d_b_filtered_size));
    CHECK_CUDA(cudaMalloc(&d_edge_test, d_edge_test_size));
    CHECK_CUDA(cudaMalloc(&d_depth, d_depth_size));
    CHECK_CUDA(cudaMalloc(&d_ir_sum, d_ir_sum_size));
    CHECK_CUDA(cudaMalloc(&d_filtered, d_filtered_size));

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaGetLastError());
    return true;
  }

  void freeDeviceMemory()
  {
    CALL_CUDA(cudaFree(d_p0table));
    CALL_CUDA(cudaFree(d_xtable));
    CALL_CUDA(cudaFree(d_ztable));
    CALL_CUDA(cudaFree(d_lut));

    CALL_CUDA(cudaFree(d_packet));

    CALL_CUDA(cudaFree(d_a));
    CALL_CUDA(cudaFree(d_b));
    CALL_CUDA(cudaFree(d_n));
    CALL_CUDA(cudaFree(d_ir));
    CALL_CUDA(cudaFree(d_a_filtered));
    CALL_CUDA(cudaFree(d_b_filtered));
    CALL_CUDA(cudaFree(d_edge_test));
    CALL_CUDA(cudaFree(d_depth));
    CALL_CUDA(cudaFree(d_ir_sum));
    CALL_CUDA(cudaFree(d_filtered));
  }

  bool initDevice(const int deviceId)
  {
    int deviceCount = 0;

    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    int devId = -1;
    for (int i = 0; i < deviceCount; i++) {
      if (deviceId != -1 && i != deviceId)
        continue;

      cudaDeviceProp prop;
      CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
      LOG_INFO << "device " << i << ": " << prop.name << " @ " << (prop.clockRate / 1000) << "MHz Memory " << (prop.totalGlobalMem >> 20) << "MB";

      if (prop.computeMode == cudaComputeModeProhibited) {
        LOG_INFO << " Compute Mode Prohibited";
        continue;
      }

      if (prop.major < 1) {
        LOG_INFO << " does not support CUDA";
        continue;
      }

      devId = i;
      break;
    }

    if (devId == -1) {
      LOG_ERROR << "No CUDA device found";
      return false;
    }

    CHECK_CUDA(cudaSetDevice(devId));
    LOG_INFO << "selected device " << devId;

    if (!setParameters(params))
      return false;

    if (!setConfiguration(config))
      return false;

    if (!allocateDeviceMemory())
      return false;

    return true;
  }

  bool setConfiguration(const DepthPacketProcessor::Config &cfg)
  {
    config = cfg;
    float tmpf;

    tmpf = cfg.MinDepth * 1000.0f;
    cudaMemcpyToSymbolAsync(MIN_DEPTH, &tmpf, sizeof(tmpf));

    tmpf = cfg.MaxDepth * 1000.0f;
    cudaMemcpyToSymbolAsync(MAX_DEPTH, &tmpf, sizeof(tmpf));

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaGetLastError());
    return true;
  }

  bool run(const DepthPacket &packet)
  {
    size_t ir_frame_size = ir_frame->width * ir_frame->height * ir_frame->bytes_per_pixel;
    size_t depth_frame_size = depth_frame->width * depth_frame->height * depth_frame->bytes_per_pixel;

    cudaMemcpyAsync(d_packet, packet.buffer, packet.buffer_length, cudaMemcpyHostToDevice);

    processPixelStage1<<<grid_size, block_size>>>(d_lut, d_ztable, d_p0table, d_packet, d_a, d_b, d_n, d_ir);

    cudaMemcpyAsync(ir_frame->data, d_ir, ir_frame_size, cudaMemcpyDeviceToHost);

    if (config.EnableBilateralFilter) {
      filterPixelStage1<<<grid_size, block_size>>>(d_a, d_b, d_n, d_a_filtered, d_b_filtered, d_edge_test);
    }

    processPixelStage2<<<grid_size, block_size>>>(
      config.EnableBilateralFilter ? d_a_filtered : d_a,
      config.EnableBilateralFilter ? d_b_filtered : d_b,
      d_xtable, d_ztable, d_depth, d_ir_sum);

    if (config.EnableEdgeAwareFilter) {
      filterPixelStage2<<<grid_size, block_size>>>(d_depth, d_ir_sum, d_edge_test, d_filtered);
    }

    cudaMemcpyAsync(depth_frame->data, config.EnableEdgeAwareFilter ? d_filtered : d_depth, depth_frame_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaGetLastError());
    return true;
  }

  void newIrFrame()
  {
    ir_frame = new CudaFrame(ir_allocator->allocate(IMAGE_SIZE*sizeof(float)));
    ir_frame->format = Frame::Float;
  }

  void newDepthFrame()
  {
    depth_frame = new CudaFrame(depth_allocator->allocate(IMAGE_SIZE*sizeof(float)));
    depth_frame->format = Frame::Float;
  }

  void fill_trig_table(const protocol::P0TablesResponse *p0table)
  {
    for (int r = 0; r < 424; ++r) {
      float4 *it = &h_p0table[r * 512];
      const uint16_t *it0 = &p0table->p0table0[r * 512];
      const uint16_t *it1 = &p0table->p0table1[r * 512];
      const uint16_t *it2 = &p0table->p0table2[r * 512];
      for (int c = 0; c < 512; ++c, ++it, ++it0, ++it1, ++it2) {
        it->x = -((float) * it0) * 0.000031 * M_PI;
        it->y = -((float) * it1) * 0.000031 * M_PI;
        it->z = -((float) * it2) * 0.000031 * M_PI;
        it->w = 0.0f;
      }
    }
  }
};

CudaDepthPacketProcessor::CudaDepthPacketProcessor(const int deviceId):
  impl_(new CudaDepthPacketProcessorImpl(deviceId))
{
}

CudaDepthPacketProcessor::~CudaDepthPacketProcessor()
{
  delete impl_;
}

void CudaDepthPacketProcessor::setConfiguration(const DepthPacketProcessor::Config &config)
{
  DepthPacketProcessor::setConfiguration(config);

  impl_->good = impl_->setConfiguration(config);
}

void CudaDepthPacketProcessor::loadP0TablesFromCommandResponse(unsigned char *buffer, size_t buffer_length)
{
  impl_->fill_trig_table((protocol::P0TablesResponse *)buffer);
  cudaMemcpy(impl_->d_p0table, impl_->h_p0table, impl_->d_p0table_size, cudaMemcpyHostToDevice);
}

void CudaDepthPacketProcessor::loadXZTables(const float *xtable, const float *ztable)
{
  cudaMemcpy(impl_->d_xtable, xtable, impl_->d_xtable_size, cudaMemcpyHostToDevice);
  cudaMemcpy(impl_->d_ztable, ztable, impl_->d_ztable_size, cudaMemcpyHostToDevice);
}

void CudaDepthPacketProcessor::loadLookupTable(const short *lut)
{
  cudaMemcpy(impl_->d_lut, lut, impl_->d_lut_size, cudaMemcpyHostToDevice);
}

bool CudaDepthPacketProcessor::good()
{
  return impl_->good;
}

void CudaDepthPacketProcessor::process(const DepthPacket &packet)
{
  if (listener_ == NULL)
    return;

  impl_->startTiming();

  impl_->ir_frame->timestamp = packet.timestamp;
  impl_->depth_frame->timestamp = packet.timestamp;
  impl_->ir_frame->sequence = packet.sequence;
  impl_->depth_frame->sequence = packet.sequence;

  impl_->good = impl_->run(packet);

  impl_->stopTiming(LOG_INFO);

  if (!impl_->good) {
    impl_->ir_frame->status = 1;
    impl_->depth_frame->status = 1;
  }

  if (listener_->onNewFrame(Frame::Ir, impl_->ir_frame))
    impl_->newIrFrame();
  if (listener_->onNewFrame(Frame::Depth, impl_->depth_frame))
    impl_->newDepthFrame();
}

Allocator *CudaDepthPacketProcessor::getAllocator()
{
  return impl_->input_allocator;
}
} // namespace libfreenect2
