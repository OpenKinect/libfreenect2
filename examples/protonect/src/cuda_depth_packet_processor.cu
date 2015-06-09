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
#include <libfreenect2/frame_listener.hpp>
#include <helper_math.h>
#include <math_constants.h>

#include <iostream>
#include <stdexcept>
#define cudaSafeCall(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); } while(0)

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

#define OUT_NAME(FUNCTION) "[CudaDepthPacketProcessorKernel::" FUNCTION "] "
namespace libfreenect2
{
class CudaDepthPacketProcessorKernelImpl
{
public:
  short *buf_lut11to16;
  float4 *buf_p0_table;
  float *buf_x_table;
  float *buf_z_table;
  unsigned short *buf_packet;

  float4 *buf_a;
  float4 *buf_b;
  float4 *buf_n;
  float *buf_ir;
  float4 *buf_a_filtered;
  float4 *buf_b_filtered;
  unsigned char *buf_edge_test;
  float *buf_depth;
  float *buf_ir_sum;
  float *buf_filtered;

  size_t image_size;
  size_t grid_size;
  size_t block_size;

  void initDevice(const int deviceId, size_t image_size_, size_t block)
  {
    int deviceCount = 0;

    cudaSafeCall(cudaGetDeviceCount(&deviceCount));

    int devId = -1;
    for (int i = 0; i < deviceCount; i++)
    {
      if (deviceId != -1 && i != deviceId)
        continue;

      cudaDeviceProp prop;
      cudaSafeCall(cudaGetDeviceProperties(&prop, i));
      std::cout << OUT_NAME("initDevice") "device " << i << ": " << prop.name << " @ " << (prop.clockRate / 1000) << "MHz Memory " << (prop.totalGlobalMem >> 20) << "MB";

      if (prop.computeMode == cudaComputeModeProhibited)
      {
        std::cout << " Compute Mode Prohibited" << std::endl;
        continue;
      }

      if (prop.major < 1)
      {
        std::cout << " does not support CUDA" << std::endl;
        continue;
      }

      std::cout << std::endl;
      devId = i;
      break;
    }

    if (devId == -1)
    {
      throw std::runtime_error("No suitable CUDA devices found.");
    }

    cudaSafeCall(cudaSetDevice(devId));
    std::cout << OUT_NAME("initDevice") "selected device " << devId << std::endl;

    image_size = image_size_;
    grid_size = image_size_/block;
    block_size = block;
  }

  void loadTables(const short *lut11to16, const Float4 *p0_table, const float *x_table, const float *z_table)
  {
    //Read only
    size_t buf_lut11to16_size = 2048 * sizeof(short);
    size_t buf_p0_table_size = image_size * sizeof(float4);
    size_t buf_x_table_size = image_size * sizeof(float);
    size_t buf_z_table_size = image_size * sizeof(float);
    size_t buf_packet_size = ((image_size * 11) / 16) * 10 * sizeof(short);

    cudaSafeCall(cudaMalloc(&buf_lut11to16, buf_lut11to16_size));
    cudaSafeCall(cudaMalloc(&buf_p0_table, buf_p0_table_size));
    cudaSafeCall(cudaMalloc(&buf_x_table, buf_x_table_size));
    cudaSafeCall(cudaMalloc(&buf_z_table, buf_z_table_size));
    cudaSafeCall(cudaMalloc(&buf_packet, buf_packet_size));

    cudaMemcpyAsync(buf_lut11to16, lut11to16, buf_lut11to16_size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(buf_p0_table, p0_table, buf_p0_table_size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(buf_x_table, x_table, buf_x_table_size, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(buf_z_table, z_table, buf_z_table_size, cudaMemcpyHostToDevice);

    //Read-Write
    size_t buf_a_size = image_size * sizeof(float4);
    size_t buf_b_size = image_size * sizeof(float4);
    size_t buf_n_size = image_size * sizeof(float4);
    size_t buf_ir_size = image_size * sizeof(float);
    size_t buf_a_filtered_size = image_size * sizeof(float4);
    size_t buf_b_filtered_size = image_size * sizeof(float4);
    size_t buf_edge_test_size = image_size * sizeof(char);
    size_t buf_depth_size = image_size * sizeof(float);
    size_t buf_ir_sum_size = image_size * sizeof(float);
    size_t buf_filtered_size = image_size * sizeof(float);

    cudaSafeCall(cudaMalloc(&buf_a, buf_a_size));
    cudaSafeCall(cudaMalloc(&buf_b, buf_b_size));
    cudaSafeCall(cudaMalloc(&buf_n, buf_n_size));
    cudaSafeCall(cudaMalloc(&buf_ir, buf_ir_size));
    cudaSafeCall(cudaMalloc(&buf_a_filtered, buf_a_filtered_size));
    cudaSafeCall(cudaMalloc(&buf_b_filtered, buf_b_filtered_size));
    cudaSafeCall(cudaMalloc(&buf_edge_test, buf_edge_test_size));
    cudaSafeCall(cudaMalloc(&buf_depth, buf_depth_size));
    cudaSafeCall(cudaMalloc(&buf_ir_sum, buf_ir_sum_size));
    cudaSafeCall(cudaMalloc(&buf_filtered, buf_filtered_size));

    cudaDeviceSynchronize();

    cudaSafeCall(cudaGetLastError());
  }

  void generateOptions(const DepthPacketProcessor::Parameters &params, const DepthPacketProcessor::Config &config)
  {
    unsigned int tmpi;
    float tmpf;

    #define COPY(upper, lower) cudaMemcpyToSymbolAsync(upper, &params.lower, sizeof(params.lower));
    tmpi = 0x180;
    cudaMemcpyToSymbolAsync(BFI_BITMASK, &tmpi, sizeof(int));

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
    tmpf = (params.joint_bilateral_ab_threshold * params.joint_bilateral_ab_threshold) / (params.ab_multiplier * params.ab_multiplier);
    cudaMemcpyToSymbolAsync(JOINT_BILATERAL_THRESHOLD, &tmpf, sizeof(tmpf));

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

    tmpf = config.MinDepth * 1000.0f;
    cudaMemcpyToSymbolAsync(MIN_DEPTH, &tmpf, sizeof(tmpf));

    tmpf = config.MaxDepth * 1000.0f;
    cudaMemcpyToSymbolAsync(MAX_DEPTH, &tmpf, sizeof(tmpf));

    cudaDeviceSynchronize();

    cudaSafeCall(cudaGetLastError());
  }

  void run(const DepthPacket &packet, Frame *ir_frame, Frame *depth_frame, const DepthPacketProcessor::Config &config)
  {
    size_t ir_frame_size = ir_frame->width * ir_frame->height * ir_frame->bytes_per_pixel;
    size_t depth_frame_size = depth_frame->width * depth_frame->height * depth_frame->bytes_per_pixel;

    cudaMemcpyAsync(buf_packet, packet.buffer, packet.buffer_length, cudaMemcpyHostToDevice);

    processPixelStage1<<<grid_size, block_size>>>(buf_lut11to16, buf_z_table, buf_p0_table, buf_packet, buf_a, buf_b, buf_n, buf_ir);

    cudaMemcpyAsync(ir_frame->data, buf_ir, ir_frame_size, cudaMemcpyDeviceToHost);

    if (config.EnableBilateralFilter)
    {
      filterPixelStage1<<<grid_size, block_size>>>(buf_a, buf_b, buf_n, buf_a_filtered, buf_b_filtered, buf_edge_test);
    }

    processPixelStage2<<<grid_size, block_size>>>(
      config.EnableBilateralFilter ? buf_a_filtered : buf_a,
      config.EnableBilateralFilter ? buf_b_filtered : buf_b,
      buf_x_table, buf_z_table, buf_depth, buf_ir_sum);

    if (config.EnableEdgeAwareFilter)
    {
      filterPixelStage2<<<grid_size, block_size>>>(buf_depth, buf_ir_sum, buf_edge_test, buf_filtered);
    }

    cudaMemcpyAsync(depth_frame->data, config.EnableEdgeAwareFilter ? buf_filtered : buf_depth, depth_frame_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaSafeCall(cudaGetLastError());
  }
};

CudaDepthPacketProcessorKernel::CudaDepthPacketProcessorKernel():
  impl_(new CudaDepthPacketProcessorKernelImpl())
{
}

CudaDepthPacketProcessorKernel::~CudaDepthPacketProcessorKernel()
{
  delete impl_;
}

void CudaDepthPacketProcessorKernel::initDevice(const int deviceId, size_t image_size_, size_t block)
{
  impl_->initDevice(deviceId, image_size_, block);
}

void CudaDepthPacketProcessorKernel::loadTables(const short *lut11to16, const Float4 *p0_table, const float *x_table, const float *z_table)
{
  impl_->loadTables(lut11to16, p0_table, x_table, z_table);
}

void CudaDepthPacketProcessorKernel::generateOptions(const DepthPacketProcessor::Parameters &params, const DepthPacketProcessor::Config &config)
{
  impl_->generateOptions(params, config);
}

void CudaDepthPacketProcessorKernel::run(const DepthPacket &packet, Frame *ir_frame, Frame *depth_frame, const DepthPacketProcessor::Config &config)
{
  impl_->run(packet, ir_frame, depth_frame, config);
}
}
