/*
 * This code implements a depth packet processor using the phase unwrapping
 * algorithm described in the paper "Efficient Phase Unwrapping using Kernel
 * Density Estimation", ECCV 2016, Felix Järemo Lawin, Per-Erik Forssen and
 * Hannes Ovren, see http://www.cvl.isy.liu.se/research/datasets/kinect2-dataset/.
 */


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

__constant__ static float KDE_SIGMA_SQR;
__constant__ static unsigned int KDE_NEIGBORHOOD_SIZE;
__constant__ static float UNWRAPPING_LIKELIHOOD_SCALE;
__constant__ static float PHASE_CONFIDENCE_SCALE;
__constant__ static float KDE_THRESHOLD;

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
#define NUM_HYPOTHESES 30

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
 * KDE phase unwrapping
 ******************************************************************************/

//arrays for hypotheses
__device__ float k_list[NUM_HYPOTHESES] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
__device__ float n_list[NUM_HYPOTHESES] = {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 5.0f, 6.0f, 5.0f, 6.0f, 6.0f, 7.0f, 7.0f, 8.0f, 8.0f, 7.0f, 8.0f, 9.0f, 9.0f};
__device__ float m_list[NUM_HYPOTHESES] = {0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 6.0f, 7.0f, 7.0f, 7.0f, 7.0f, 8.0f, 8.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 13.0f, 13.0f, 14.0f};

static __device__
void calcErr(const float k, const float n, const float m, const float t0, const float t1, const float t2, float* err1, float* err2, float* err3)
{
  //phase unwrapping equation residuals.
  //The coefficients 2.0, 15.0 and 3.0 are the smallest common multiples of the sequence of modulation frequencies 16, 80 and 120 MHz
  *err1 = 3.0f*n - 15.0f*k - (t1 - t0);
  *err2 = 3.0f*n - 2.0f*m - (t2 - t0);
  *err3 = 15.0f*k - 2.0f*m - (t2 - t1);
}

/********************************************************************************
 * Rank all 30 phase hypothses and returns the two most likley
 ********************************************************************************/
static __device__
void phaseUnWrapper(float t0, float t1,float t2, float* phase_first, float* phase_second, float* err_w1, float* err_w2)
{
  float err;
  float err1, err2, err3;

  //unwrapping weight for cost function
  float w1 = 1.0f;
  float w2 = 10.0f;
  float w3 = 1.0218f;

  //initialize residuals with som large numbers
  float err_min = 100000.0f;
  float err_min_second = 200000.0f;
  unsigned int ind_min, ind_second;

  float k, n, m;

  for(int i = 0; i < NUM_HYPOTHESES; i++)
  {
    m = m_list[i];
    n = n_list[i];
    k = k_list[i];
    calcErr(k, n, m, t0, t1, t2, &err1, &err2, &err3);
    err = w1 * err1 * err1 + w2 * err2 * err2 + w3 * err3 * err3;
    if(err < err_min)
    {
      err_min_second = err_min;
      ind_second = ind_min;
      err_min = err;
      ind_min = i;

    }
    else if(err < err_min_second)
    {
      err_min_second = err;
      ind_second = i;
    }

  }

  //decode ind_min
  float mvals = m_list[ind_min];
  float nvals = n_list[ind_min];
  float kvals = k_list[ind_min];

  //Weighted phases for phase fusion weighted average
  float phi2_out = (t2 / 2.0f + mvals);
  float phi1_out = (t1 / 15.0f + kvals);
  float phi0_out = (t0 / 3.0f + nvals);

  *err_w1 = err_min;

  //phase fusion
  *phase_first = (phi2_out + phi1_out + phi0_out) / 3.0f;

  mvals = m_list[ind_second];
  nvals = n_list[ind_second];
  kvals = k_list[ind_second];

  //Weighted phases for phase fusion weighted average
  phi2_out = (t2 / 2.0f + mvals);
  phi1_out = (t1 / 15.0f + kvals);
  phi0_out = (t0 / 3.0f + nvals);

  *err_w2 = err_min_second;

  //phase fusion
  *phase_second = (phi2_out + phi1_out + phi0_out) / 3.0f;

}

/*******************************************************************************
 * Predict phase variance from amplitude direct quadratic model
 ******************************************************************************/
static __device__
void calculatePhaseUnwrappingVarDirect(float3 ir, float3* var)
{
  //Model: sigma = 1/(gamma0*a+gamma1*a^2+gamma2). Possibly better than calculatePhaseUnwrappingVar
  //The gammas are optimized using lsqnonlin in matlab.
  //For more details see the paper "Efficient Phase Unwrapping using Kernel Density Estimation"
  //section 3.3 and 4.4.
  float sigma_max = 0.5f * M_PI_F;

  //Set sigma = pi/2 as a maximum standard deviation of the phase. Cut off function after root
  float q0 = ir.x > 5.244404f ? 0.7919451669f * ir.x - 0.002363097609f * ir.x * ir.x - 3.088285897f : 1.0f / sigma_max;
  float q1 = ir.y > 4.084835f ? 1.214266794f * ir.y - 0.00581082634f * ir.y * ir.y - 3.863119924f : 1.0f / sigma_max;
  float q2 = ir.z > 6.379475f ? 0.6101457464f * ir.z - 0.00113679233f * ir.z * ir.z - 2.84614442f : 1.0f / sigma_max;
  float3 q = make_float3(q0, q1, q2);
  float3 roots = make_float3(5.244404f, 4.084835f, 6.379475f);
  float3 sigma = make_float3(1.0f)/q;
  sigma = select(sigma, make_float3(sigma_max), isless(make_float3(sigma_max), sigma));
  *var = sigma;
}


/*******************************************************************************
 * Predict phase variance from amplitude (quadratic atan model)
 ******************************************************************************/
static __device__
void calculatePhaseUnwrappingVar(float3 ir, float3 *var)
{
  //Model: sigma = atan(sqrt(1/(gamma0*a+gamma1*a^2+gamma2)-1)). The gammas are optimized using lsqnonlin in matlab.
  //For more details see the paper "Efficient Phase Unwrapping using Kernel Density Estimation",
  //section 3.3 and 4.4.
  float q0 = 0.8211288451f * ir.x - 0.002601348899f * ir.x * ir.x - 3.549793908f;
  float q1 = 1.259642407f * ir.y - 0.005478390508f * ir.y * ir.y - 4.335841127f;
  float q2 = 0.6447928035f * ir.z - 0.0009627273649f * ir.z * ir.z - 3.368205575f;
  float3 q = make_float3(q0, q1, q2);
  q *= q;
  float3 roots = make_float3(5.64173671f, 4.31705182f, 6.84453530f);
  float3 asdf = atan2(make_float3(0.5f), make_float3(1.0f));
  float3 sigma = select(select(make_float3(0.5f * M_PI_F), roots * 0.5f * M_PI_F / ir, isless(roots,ir)), atan2(sqrt(make_float3(1.0f) / (q - make_float3(1.0f))), make_float3(1.0f)), isless(make_float3(1.0f), q));
  sigma = select(sigma, make_float3(0.001f), isless(sigma, make_float3(0.001f)));
  *var = sigma*sigma;
}

static __global__
void processPixelStage2_phase(const float4* __restrict__ a_in, const float4* __restrict__ b_in, float4 *phase_conf_vec)
{
  const uint i = get_global_id(0);

  //read complex number real (a) and imaginary part (b)
  float3 a = make_float3(a_in[i]);
  float3 b = make_float3(b_in[i]);

  //calculate complex argument
  float3 phase = atan2(b, a);
  phase = select(phase, make_float3(0.0f), isnan(phase));
  phase = select(phase, phase + 2.0f * M_PI_F, isless(phase, make_float3(0.0f)));

  //calculate amplitude or the absolute value
  float3 ir = sqrt(a * a + b * b) * AB_MULTIPLIER;

  float ir_sum = ir.x + ir.y + ir.z;
  float ir_min = min(ir.x, min(ir.y, ir.z));
  float ir_max = max(ir.x, max(ir.y, ir.z));

  float phase_first = 0.0f;
  float phase_second = 0.0f;

  float J_1, J_2, unwrapping_likelihood1, unwrapping_likelihood2;

  //scale with least common multiples of modulation frequencies
  float3 t = phase / (2.0f * M_PI_F) * make_float3(3.0f, 15.0f, 2.0f);

  float t0 = t.x;
  float t1 = t.y;
  float t2 = t.z;

  //rank and extract two most likely phase hypothises
  phaseUnWrapper(t0, t1, t2, &phase_first, &phase_second, &J_1, &J_2);

  float phase_likelihood;

  //check if near saturation
  if(ir_sum < 0.4f * 65535.0f)
  {
    //calculate phase likelihood from amplitude
    float3 var;
    calculatePhaseUnwrappingVar(ir, &var);
    phase_likelihood = exp(-(var.x + var.y + var.z) / (2.0f * PHASE_CONFIDENCE_SCALE));
    phase_likelihood = isnan(phase_likelihood) ? 0.0f : phase_likelihood;
  }
  else
  {
    phase_likelihood = 0.0f;
  }

  //merge phase likelihood with phase likelihood
  unwrapping_likelihood1 = phase_likelihood*exp(-J_1 / (2 * UNWRAPPING_LIKELIHOOD_SCALE));
  unwrapping_likelihood2 = phase_likelihood*exp(-J_2 / (2 * UNWRAPPING_LIKELIHOOD_SCALE));

  //suppress confidence if phase is beyond allowed range
  unwrapping_likelihood1 = phase_first > MAX_DEPTH * 9.0f / 18750.0f ? 0.0f: unwrapping_likelihood1;
  unwrapping_likelihood2 = phase_second > MAX_DEPTH * 9.0f / 18750.0f ? 0.0f: unwrapping_likelihood2;

  phase_conf_vec[i] = make_float4(phase_first,phase_second, unwrapping_likelihood1, unwrapping_likelihood2);

}

static __global__
void filter_kde(const float4 *phase_conf_vec, const float* gauss_filt_array, const float* __restrict__ x_table, const float* __restrict__ z_table, float* depth)
{
  const uint i = get_global_id(0);
  float kde_val_1, kde_val_2;

  const int loadX = i % 512;
  const int loadY = i / 512;

  int k, l;
  float sum_1, sum_2;

  //initialize neighborhood boundaries
  int from_x = (loadX > KDE_NEIGBORHOOD_SIZE ? -KDE_NEIGBORHOOD_SIZE : -loadX + 1);
  int from_y = (loadY > KDE_NEIGBORHOOD_SIZE ? -KDE_NEIGBORHOOD_SIZE : -loadY + 1);
  int to_x = (loadX < 511 - KDE_NEIGBORHOOD_SIZE - 1 ? KDE_NEIGBORHOOD_SIZE: 511 - loadX - 1);
  int to_y = (loadY < 423 - KDE_NEIGBORHOOD_SIZE ? KDE_NEIGBORHOOD_SIZE: 423 - loadY);

  kde_val_1 = 0.0f;
  kde_val_2 = 0.0f;
  float4 phase_local = phase_conf_vec[i];
  if(loadX >= 1 && loadX < 511 && loadY >= 0 && loadY<424)
  {
    sum_1 = 0.0f;
    sum_2 = 0.0f;
    float gauss;
    float sum_gauss = 0.0f;

    float phase_1_local;
    float phase_2_local;
    float conf1_local;
    float conf2_local;
    float4 phase_conf_local;
    uint ind;
    float diff11, diff21, diff12, diff22;
    //calculate KDE for all hypothesis within the neigborhood
    for(k = from_y; k <= to_y; k++)
      for(l = from_x; l <= to_x; l++)
      {
        ind = (loadY + k) * 512 + (loadX + l);

        phase_conf_local = phase_conf_vec[ind];
        conf1_local = phase_conf_local.z;
        conf2_local = phase_conf_local.w;
        phase_1_local = phase_conf_local.x;
        phase_2_local = phase_conf_local.y;

        gauss = gauss_filt_array[k + KDE_NEIGBORHOOD_SIZE] * gauss_filt_array[l + KDE_NEIGBORHOOD_SIZE];
        sum_gauss += gauss * (conf1_local + conf2_local);
        diff11 = phase_1_local - phase_local.x;
        diff21 = phase_2_local - phase_local.x;
        diff12 = phase_1_local - phase_local.y;
        diff22 = phase_2_local - phase_local.y;
        sum_1 += gauss * (conf1_local * exp(-diff11 * diff11 / (2 * KDE_SIGMA_SQR)) + conf2_local * exp(-diff21 * diff21 / (2 * KDE_SIGMA_SQR)));
        sum_2 += gauss * (conf1_local * exp(-diff12 * diff12 / (2 * KDE_SIGMA_SQR)) + conf2_local * exp(-diff22 * diff22 / (2 * KDE_SIGMA_SQR)));
      }
    kde_val_1 = sum_gauss > 0.5f ? sum_1/sum_gauss : sum_1 * 2.0f;
    kde_val_2 = sum_gauss > 0.5f ? sum_2/sum_gauss : sum_2 * 2.0f;
  }

  //select hypothesis
  int val_ind = kde_val_2 <= kde_val_1 ? 1: 0;

  float phase_final = val_ind ? phase_local.x: phase_local.y;
  float max_val = val_ind ? kde_val_1: kde_val_2;

  float zmultiplier = z_table[i];
  float xmultiplier = x_table[i];

  float depth_linear = zmultiplier * phase_final;
  float max_depth = phase_final * UNAMBIGIOUS_DIST * 2.0f;

  bool cond1 =  true && 0.0f < depth_linear && 0.0f < max_depth;

  xmultiplier = (xmultiplier * 90.0f) / (max_depth * max_depth * 8192.0f);

  float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1);
  depth_fit = depth_fit < 0.0f ? 0.0f : depth_fit;

  float d = cond1 ? depth_fit : depth_linear; // r1.y -> later r2.z

  max_val = d < MIN_DEPTH || d > MAX_DEPTH ? 0.0f: max_val;

  depth[i] = max_val >= KDE_THRESHOLD ? d: 0.0f;
}


/*****************************************************************
 * THREE HYPOTHESIS
 *****************************************************************/
static __device__
void phaseUnWrapper3(float t0, float t1,float t2, float* phase_first, float* phase_second, float* phase_third, float* err_w1, float* err_w2, float* err_w3)
{
  float err;
  float err1, err2, err3;

  //unwrapping weight for cost function
  float w1 = 1.0f;
  float w2 = 10.0f;
  float w3 = 1.0218f;

  //initialize residuals with som large numbers
  float err_min = 100000.0f;
  float err_min_second = 200000.0f;
  float err_min_third = 300000.0f;
  unsigned int ind_min, ind_second, ind_third;

  float k, n, m;

  for(int i = 0; i < NUM_HYPOTHESES; i++)
  {
    m = m_list[i];
    n = n_list[i];
    k = k_list[i];
    calcErr(k, n, m, t0, t1, t2, &err1, &err2, &err3);
    err = w1 * err1 * err1 + w2 * err2 * err2 + w3 * err3 * err3;
    if(err < err_min)
    {
      err_min_third = err_min_second;
      ind_third = ind_second;
      err_min_second = err_min;
      ind_second = ind_min;
      err_min = err;
      ind_min = i;

    }
    else if(err<err_min_second)
    {
      err_min_third = err_min_second;
      ind_third = ind_second;
      err_min_second = err;
      ind_second = i;
    }
    else if(err<err_min_third)
    {
      err_min_third = err;
      ind_third = i;
    }
  }

  //decode ind_min
  float mvals = m_list[ind_min];
  float nvals = n_list[ind_min];
  float kvals = k_list[ind_min];

  //Weighted phases for phase fusion weighted average
  float phi2_out = (t2 / 2.0f + mvals);
  float phi1_out = (t1 / 15.0f + kvals);
  float phi0_out = (t0 / 3.0f + nvals);

  *err_w1 = err_min;

  *phase_first = (phi2_out + phi1_out + phi0_out) / 3.0f;

  mvals = m_list[ind_second];
  nvals = n_list[ind_second];
  kvals = k_list[ind_second];

  //Weighted phases for phase fusion weighted average
  phi2_out = (t2 / 2.0f + mvals);
  phi1_out = (t1 / 15.0f + kvals);
  phi0_out = (t0 / 3.0f + nvals);

  *err_w2 = err_min_second;
  *phase_second = (phi2_out + phi1_out + phi0_out) / 3.0f;

  mvals = m_list[ind_third];
  nvals = n_list[ind_third];
  kvals = k_list[ind_third];

  //Weighted phases for phase fusion weighted average
  phi2_out = (t2 / 2.0f + mvals);
  phi1_out = (t1 / 15.0f + kvals);
  phi0_out = (t0 / 3.0f + nvals);

  *err_w3 = err_min_third;
  *phase_third = (phi2_out + phi1_out + phi0_out) / 3.0f;
}

static __global__
void processPixelStage2_phase3(const float4* __restrict__ a_in, const float4* __restrict__ b_in, float *phase_1, float *phase_2, float *phase_3, float *conf1, float *conf2, float *conf3)
{
  const uint i = get_global_id(0);

  //read complex number real (a) and imaginary part (b)
  float3 a = make_float3(a_in[i]);
  float3 b = make_float3(b_in[i]);

  //calculate complex argument
  float3 phase = atan2(b, a);
  phase = select(phase, make_float3(0.0f), isnan(phase));
  phase = select(phase, phase + 2.0f * M_PI_F, isless(phase, make_float3(0.0f)));

  //calculate amplitude or the absolute value
  float3 ir = sqrt(a * a + b * b) * AB_MULTIPLIER;

  float ir_sum = ir.x + ir.y + ir.z;
  float ir_min = min(ir.x, min(ir.y, ir.z));
  float ir_max = max(ir.x, max(ir.y, ir.z));

  float phase_first = 0.0f;
  float phase_second = 0.0f;
  float phase_third = 0.0f;
  float J_1, J_2, J_3, unwrapping_likelihood1, unwrapping_likelihood2, unwrapping_likelihood3;

  //scale with least common multiples of modulation frequencies
  float3 t = phase / (2.0f * M_PI_F) * make_float3(3.0f, 15.0f, 2.0f);

  float t0 = t.x;
  float t1 = t.y;
  float t2 = t.z;

  //rank and extract three most likely phase hypothises
  phaseUnWrapper3(t0, t1, t2, &phase_first, &phase_second, &phase_third, &J_1, &J_2, &J_3);

  phase_1[i] = phase_first;
  phase_2[i] = phase_second;
  phase_3[i] = phase_third;

  float phase_likelihood;
  //check if near saturation
  if(ir_sum < 0.4f * 65535.0f)
  {
    //calculate phase likelihood from amplitude
    float3 var;
    calculatePhaseUnwrappingVar(ir, &var);
    phase_likelihood = exp(-(var.x + var.y + var.z) / (2.0f * PHASE_CONFIDENCE_SCALE));
    phase_likelihood = isnan(phase_likelihood) ? 0.0f : phase_likelihood;
  }
  else
  {
    phase_likelihood = 0.0f;
  }

  //merge unwrapping likelihood with phase likelihood
  unwrapping_likelihood1 = phase_likelihood * exp(-J_1 / (2 * UNWRAPPING_LIKELIHOOD_SCALE));
  unwrapping_likelihood2 = phase_likelihood * exp(-J_2 / (2 * UNWRAPPING_LIKELIHOOD_SCALE));
  unwrapping_likelihood3 = phase_likelihood * exp(-J_3 / (2 * UNWRAPPING_LIKELIHOOD_SCALE));

  //suppress confidence if phase is beyond allowed range
  unwrapping_likelihood1 = phase_first > MAX_DEPTH * 9.0f / 18750.0f ? 0.0f: unwrapping_likelihood1;
  unwrapping_likelihood2 = phase_second > MAX_DEPTH * 9.0f / 18750.0f ? 0.0f: unwrapping_likelihood2;
  unwrapping_likelihood3 = phase_third > MAX_DEPTH * 9.0f / 18750.0f ? 0.0f: unwrapping_likelihood3;

  conf1[i] = unwrapping_likelihood1;
  conf2[i] = unwrapping_likelihood2;
  conf3[i] = unwrapping_likelihood3;
}


static __global__
void filter_kde3(const float *phase_1, const float *phase_2, const float *phase_3, const float* conf1, const float* conf2, const float* conf3, const float* gauss_filt_array, const float* __restrict__ x_table, const float* __restrict__ z_table, float* depth)
{
  const uint i = get_global_id(0);
  float kde_val_1, kde_val_2, kde_val_3;

  const int loadX = i % 512;
  const int loadY = i / 512;
  int k, l;
  float sum_1, sum_2, sum_3;

  //initialize neighborhood boundaries
  int from_x = (loadX > KDE_NEIGBORHOOD_SIZE ? -KDE_NEIGBORHOOD_SIZE : -loadX + 1);
  int from_y = (loadY > KDE_NEIGBORHOOD_SIZE ? -KDE_NEIGBORHOOD_SIZE : -loadY + 1);
  int to_x = (loadX < 511 - KDE_NEIGBORHOOD_SIZE-1 ? KDE_NEIGBORHOOD_SIZE: 511 - loadX - 1);
  int to_y = (loadY < 423 - KDE_NEIGBORHOOD_SIZE ? KDE_NEIGBORHOOD_SIZE: 423 - loadY);

  kde_val_1 = 0.0f;
  kde_val_2 = 0.0f;
  kde_val_3 = 0.0f;
  float phase_first = phase_1[i];
  float phase_second = phase_2[i];
  float phase_third = phase_3[i];
  if(loadX >= 1 && loadX < 511 && loadY >= 0 && loadY<424)
  {
    //Filter kernel
    sum_1 = 0.0f;
    sum_2 = 0.0f;
    sum_3 = 0.0f;
    float gauss;
    float sum_gauss = 0.0f;

    float phase_1_local;
    float phase_2_local;
    float phase_3_local;
    float conf1_local;
    float conf2_local;
    float conf3_local;
    float diff11, diff12, diff13, diff21, diff22, diff23, diff31, diff32, diff33;
    uint ind;

    //calculate KDE for all hypothesis within the neigborhood
    for(k = from_y; k <= to_y; k++)
      for(l = from_x; l <= to_x; l++)
      {
        ind = (loadY + k) * 512+ (loadX + l);
        conf1_local = conf1[ind];
        conf2_local = conf2[ind];
        conf3_local = conf3[ind];
        phase_1_local = phase_1[ind];
        phase_2_local = phase_2[ind];
        phase_3_local = phase_3[ind];
        diff11 = phase_1_local - phase_first;
        diff12 = phase_1_local - phase_second;
        diff13 = phase_1_local - phase_third;
        diff21 = phase_2_local - phase_first;
        diff22 = phase_2_local - phase_second;
        diff23 = phase_2_local - phase_third;
        diff31 = phase_3_local - phase_first;
        diff32 = phase_3_local - phase_second;
        diff33 = phase_3_local - phase_third;
        gauss = gauss_filt_array[k + KDE_NEIGBORHOOD_SIZE] * gauss_filt_array[l + KDE_NEIGBORHOOD_SIZE];
        sum_gauss += gauss * (conf1_local + conf2_local+conf3_local);
        sum_1 += gauss*(conf1_local * exp(-diff11 * diff11 / (2 * KDE_SIGMA_SQR)) + conf2_local * exp(-diff21 * diff21 / (2 * KDE_SIGMA_SQR)) + conf3_local * exp(-diff31 * diff31 /(2 * KDE_SIGMA_SQR)));
        sum_2 += gauss * (conf1_local * exp(-diff12 * diff12 / (2 * KDE_SIGMA_SQR)) + conf2_local * exp(-diff22 * diff22 / (2 * KDE_SIGMA_SQR)) + conf3_local * exp(-diff32 * diff32 / (2 * KDE_SIGMA_SQR)));
        sum_3 += gauss * (conf1_local * exp(-diff13 * diff13 / (2 * KDE_SIGMA_SQR)) + conf2_local * exp(-diff23 * diff23 / (2 * KDE_SIGMA_SQR)) + conf3_local * exp(-diff33 * diff33 / (2 * KDE_SIGMA_SQR)));
      }
    kde_val_1 = sum_gauss > 0.5f ? sum_1 / sum_gauss : sum_1 * 2.0f;
    kde_val_2 = sum_gauss > 0.5f ? sum_2 / sum_gauss : sum_2 * 2.0f;
    kde_val_3 = sum_gauss > 0.5f ? sum_3 / sum_gauss : sum_3 * 2.0f;
  }

  //select hypothesis
  float phase_final, max_val;
  if(kde_val_2 > kde_val_1 || kde_val_3 > kde_val_1)
  {
    if(kde_val_3 > kde_val_2)
    {
      phase_final = phase_third;
      max_val = kde_val_3;
    }
    else
    {
      phase_final = phase_second;
      max_val = kde_val_2;
    }
  }
  else
  {
    phase_final = phase_first;
    max_val = kde_val_1;
  }

  float zmultiplier = z_table[i];
  float xmultiplier = x_table[i];

  float depth_linear = zmultiplier * phase_final;
  float max_depth = phase_final * UNAMBIGIOUS_DIST * 2.0f;

  bool cond1 =  true && 0.0f < depth_linear && 0.0f < max_depth;

  xmultiplier = (xmultiplier * 90.0f) / (max_depth * max_depth * 8192.0f);

  float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1);
  depth_fit = depth_fit < 0.0f ? 0.0f : depth_fit;

  float d = cond1 ? depth_fit : depth_linear; // r1.y -> later r2.z

  max_val = depth_linear < MIN_DEPTH || depth_linear > MAX_DEPTH ? 0.0f: max_val;

  //set to zero if confidence is low
  depth[i] = max_val >= KDE_THRESHOLD ? d: 0.0f;
}


#define CHECK_CUDA(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { LOG_ERROR << #expr ": " << cudaGetErrorString(err); return false; } } while(0)
#define CALL_CUDA(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { LOG_ERROR << #expr ": " << cudaGetErrorString(err); } } while(0)

namespace libfreenect2
{

class CudaKdeFrame: public Frame
{
public:
  CudaKdeFrame(Buffer *buffer):
    Frame(512, 424, 4, (unsigned char*)-1)
  {
    data = buffer->data;
    rawdata = reinterpret_cast<unsigned char *>(buffer);
  }

  virtual ~CudaKdeFrame()
  {
    Buffer *buffer = reinterpret_cast<Buffer*>(rawdata);
    buffer->allocator->free(buffer);
    rawdata = NULL;
  }
};

class CudaKdeAllocator: public Allocator
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
  CudaKdeAllocator(bool input): input(input) {}

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

class CudaKdeDepthPacketProcessorImpl: public WithPerfLogging
{
public:
  static const size_t IMAGE_SIZE = 512*424;
  static const size_t LUT_SIZE = 2048;

  size_t d_lut_size;
  size_t d_xtable_size;
  size_t d_ztable_size;
  size_t d_p0table_size;
  size_t d_gauss_kernel_size;

  short *d_lut;
  float *d_xtable;
  float *d_ztable;
  float4 *d_p0table;
  float4 h_p0table[IMAGE_SIZE];
  float* d_gauss_kernel;

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
  //float *d_ir_sum;
  float *d_phase_1;
  float *d_phase_2;
  float *d_phase_3;
  float *d_conf_1;
  float *d_conf_2;
  float *d_conf_3;
  float4 *d_phase_conf_vec;

  size_t block_size;
  size_t grid_size;

  DepthPacketProcessor::Config config;
  DepthPacketProcessor::Parameters params;

  Frame *ir_frame, *depth_frame;

  Allocator *input_allocator;
  Allocator *ir_allocator;
  Allocator *depth_allocator;

  bool good;

  CudaKdeDepthPacketProcessorImpl(const int deviceId):
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

    input_allocator = new PoolAllocator(new CudaKdeAllocator(true));
    ir_allocator = new PoolAllocator(new CudaKdeAllocator(false));
    depth_allocator = new PoolAllocator(new CudaKdeAllocator(false));

    newIrFrame();
    newDepthFrame();
  }

  ~CudaKdeDepthPacketProcessorImpl()
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


    COPY(KDE_SIGMA_SQR, kde_sigma_sqr);
    COPY(KDE_NEIGBORHOOD_SIZE, kde_neigborhood_size);
    COPY(UNWRAPPING_LIKELIHOOD_SCALE, unwrapping_likelihood_scale);
    COPY(PHASE_CONFIDENCE_SCALE, phase_confidence_scale);
    COPY(KDE_THRESHOLD, kde_threshold);
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
    d_gauss_kernel_size = (2*params.kde_neigborhood_size+1)*sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_p0table, d_p0table_size));
    CHECK_CUDA(cudaMalloc(&d_xtable, d_xtable_size));
    CHECK_CUDA(cudaMalloc(&d_ztable, d_ztable_size));
    CHECK_CUDA(cudaMalloc(&d_lut, d_lut_size));
    CHECK_CUDA(cudaMalloc(&d_gauss_kernel, d_gauss_kernel_size));

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
    size_t d_phase_conf_vec_size = IMAGE_SIZE * sizeof(float4);

    CHECK_CUDA(cudaMalloc(&d_a, d_a_size));
    CHECK_CUDA(cudaMalloc(&d_b, d_b_size));
    CHECK_CUDA(cudaMalloc(&d_n, d_n_size));
    CHECK_CUDA(cudaMalloc(&d_ir, d_ir_size));
    CHECK_CUDA(cudaMalloc(&d_a_filtered, d_a_filtered_size));
    CHECK_CUDA(cudaMalloc(&d_b_filtered, d_b_filtered_size));
    CHECK_CUDA(cudaMalloc(&d_edge_test, d_edge_test_size));
    CHECK_CUDA(cudaMalloc(&d_depth, d_depth_size));
    CHECK_CUDA(cudaMalloc(&d_phase_1, d_depth_size));
    CHECK_CUDA(cudaMalloc(&d_phase_2, d_depth_size));
    CHECK_CUDA(cudaMalloc(&d_conf_1, d_depth_size));
    CHECK_CUDA(cudaMalloc(&d_conf_2, d_depth_size));
    CHECK_CUDA(cudaMalloc(&d_phase_conf_vec, d_phase_conf_vec_size));

    if(params.num_hyps == 3)
    {
      CHECK_CUDA(cudaMalloc(&d_phase_3, d_depth_size));
      CHECK_CUDA(cudaMalloc(&d_conf_3, d_depth_size));
    }

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
    CALL_CUDA(cudaFree(d_gauss_kernel));

    CALL_CUDA(cudaFree(d_packet));

    CALL_CUDA(cudaFree(d_a));
    CALL_CUDA(cudaFree(d_b));
    CALL_CUDA(cudaFree(d_n));
    CALL_CUDA(cudaFree(d_ir));
    CALL_CUDA(cudaFree(d_a_filtered));
    CALL_CUDA(cudaFree(d_b_filtered));
    CALL_CUDA(cudaFree(d_edge_test));
    CALL_CUDA(cudaFree(d_depth));
    //CALL_CUDA(cudaFree(d_ir_sum));
    CALL_CUDA(cudaFree(d_phase_1));
    CALL_CUDA(cudaFree(d_phase_2));
    CALL_CUDA(cudaFree(d_conf_1));
    CALL_CUDA(cudaFree(d_conf_2));
    CALL_CUDA(cudaFree(d_phase_conf_vec));

    if(params.num_hyps == 3)
    {
      CALL_CUDA(cudaFree(d_phase_3));
      CALL_CUDA(cudaFree(d_conf_3));
    }
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

    if(params.num_hyps == 3)
    {
      processPixelStage2_phase3<<<grid_size, block_size>>>(
        config.EnableBilateralFilter ? d_a_filtered : d_a,
        config.EnableBilateralFilter ? d_b_filtered : d_b,
        d_phase_1, d_phase_2, d_phase_3, d_conf_1, d_conf_2, d_conf_3);

      filter_kde3<<<grid_size, block_size>>>(
        d_phase_1,
        d_phase_2,
        d_phase_3,
        d_conf_1,
        d_conf_2,
        d_conf_3,
        d_gauss_kernel,
        d_xtable,
        d_ztable,
        d_depth);
    }
    else
    {
      processPixelStage2_phase<<<grid_size, block_size>>>(
        config.EnableBilateralFilter ? d_a_filtered : d_a,
        config.EnableBilateralFilter ? d_b_filtered : d_b,
        d_phase_conf_vec);

      filter_kde<<<grid_size, block_size>>>(
        d_phase_conf_vec,
        d_gauss_kernel,
        d_xtable,
        d_ztable,
        d_depth);
    }

    cudaMemcpyAsync(depth_frame->data, d_depth, depth_frame_size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaGetLastError());
    return true;
  }

  void newIrFrame()
  {
    ir_frame = new CudaKdeFrame(ir_allocator->allocate(IMAGE_SIZE*sizeof(float)));
    ir_frame->format = Frame::Float;
  }

  void newDepthFrame()
  {
    depth_frame = new CudaKdeFrame(depth_allocator->allocate(IMAGE_SIZE*sizeof(float)));
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

  //initialize spatial weights
  void createGaussianKernel(float** kernel, int size)
  {
    *kernel = new float[2*size+1];
    float sigma = 0.5f*(float)size;

    for(int i = -size; i <= size; i++)
    {
      (*kernel)[i+size] = exp(-0.5f*i*i/(sigma*sigma));
    }
  }
};

CudaKdeDepthPacketProcessor::CudaKdeDepthPacketProcessor(const int deviceId):
  impl_(new CudaKdeDepthPacketProcessorImpl(deviceId))
{
}

CudaKdeDepthPacketProcessor::~CudaKdeDepthPacketProcessor()
{
  delete impl_;
}

void CudaKdeDepthPacketProcessor::setConfiguration(const DepthPacketProcessor::Config &config)
{
  DepthPacketProcessor::setConfiguration(config);

  impl_->good = impl_->setConfiguration(config);
}

void CudaKdeDepthPacketProcessor::loadP0TablesFromCommandResponse(unsigned char *buffer, size_t buffer_length)
{
  impl_->fill_trig_table((protocol::P0TablesResponse *)buffer);
  cudaMemcpy(impl_->d_p0table, impl_->h_p0table, impl_->d_p0table_size, cudaMemcpyHostToDevice);
}

void CudaKdeDepthPacketProcessor::loadXZTables(const float *xtable, const float *ztable)
{
  cudaMemcpy(impl_->d_xtable, xtable, impl_->d_xtable_size, cudaMemcpyHostToDevice);
  cudaMemcpy(impl_->d_ztable, ztable, impl_->d_ztable_size, cudaMemcpyHostToDevice);
  float* gauss_kernel;
  impl_->createGaussianKernel(&gauss_kernel, impl_->params.kde_neigborhood_size);
  cudaMemcpy(impl_->d_gauss_kernel, gauss_kernel, impl_->d_gauss_kernel_size, cudaMemcpyHostToDevice);
}

void CudaKdeDepthPacketProcessor::loadLookupTable(const short *lut)
{
  cudaMemcpy(impl_->d_lut, lut, impl_->d_lut_size, cudaMemcpyHostToDevice);
}

bool CudaKdeDepthPacketProcessor::good()
{
  return impl_->good;
}

void CudaKdeDepthPacketProcessor::process(const DepthPacket &packet)
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

Allocator *CudaKdeDepthPacketProcessor::getAllocator()
{
  return impl_->input_allocator;
}
} // namespace libfreenect2
