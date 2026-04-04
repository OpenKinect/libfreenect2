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

/** @file metal_depth_packet_processor.metal Metal compute kernels for depth processing. */

#include <metal_stdlib>
using namespace metal;

/** Parameters struct passed as a constant buffer to all kernels.
 * Mirrors libfreenect2::DepthPacketProcessor::Parameters. */
struct MetalDepthParams
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
  float joint_bilateral_threshold; /* precomputed: (ab_threshold^2)/(ab_multiplier^2) */

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

/*******************************************************************************
 * Decode a packed 11-bit pixel measurement from the raw IR packet buffer.
 ******************************************************************************/
static float decodePixelMeasurement(
    device const ushort *data,
    device const short *lut11to16,
    const uint sub,
    const uint x,
    const uint y)
{
  uint row_idx = (424u * sub + y) * 352u;
  uint idx = (((x >> 2u) + ((x << 7u) & 0x180u)) * 11u) & 0xffffffffu;

  uint col_idx = idx >> 4u;
  uint upper_bytes = idx & 15u;
  uint lower_bytes = 16u - upper_bytes;

  uint data_idx0 = row_idx + col_idx;
  uint data_idx1 = row_idx + col_idx + 1u;

  uint packed = (x < 1u || 510u < x || col_idx > 352u) ? 0u :
      ((uint(data[data_idx0]) >> upper_bytes) | (uint(data[data_idx1]) << lower_bytes)) & 2047u;

  return float(lut11to16[packed]);
}

/*******************************************************************************
 * Process pixel stage 1: phase unwrapping from raw IR data.
 *
 * One thread per pixel (linear index i = y * 512 + x).
 * Reads raw packet data and p0 calibration tables, outputs complex IR vectors
 * (a, b) and amplitude (n) for three modulation frequencies, plus a quick IR
 * image for monitoring.
 ******************************************************************************/
kernel void processPixelStage1(
    device const short   *lut11to16   [[ buffer(0) ]],
    device const float   *z_table     [[ buffer(1) ]],
    device const float3  *p0_table    [[ buffer(2) ]],
    device const ushort  *data        [[ buffer(3) ]],
    device float3        *a_out       [[ buffer(4) ]],
    device float3        *b_out       [[ buffer(5) ]],
    device float3        *n_out       [[ buffer(6) ]],
    device float         *ir_out      [[ buffer(7) ]],
    constant MetalDepthParams &params [[ buffer(8) ]],
    uint i [[ thread_position_in_grid ]])
{
  const uint x = i % 512u;
  const uint y = i / 512u;

  /* The raw frame rows are stored in a rearranged order:
   * bottom half of the sensor is stored first, top half second.
   * y_in maps the output row index to the correct input row. */
  const uint y_tmp = 423u - y;
  const uint y_in = (y_tmp < 212u) ? y_tmp + 212u : 423u - y_tmp;

  /* Pixel validity: z_table[i] <= 0 means no calibration data. */
  const bool invalid = (0.0f >= z_table[i]);

  /* Per-pixel phase offsets from calibration tables. */
  const float3 p0 = p0_table[i];

  /* Phase vector for the three modulation frequencies. */
  const float3 phase = float3(params.phase_in_rad0, params.phase_in_rad1, params.phase_in_rad2);

  /* Compute sin/cos for each frequency combined with the p0 offset. */
  float3 p0x_cos, p0y_cos, p0z_cos;
  float3 p0x_sin = -sincos(phase + p0.x, p0x_cos);
  float3 p0y_sin = -sincos(phase + p0.y, p0y_cos);
  float3 p0z_sin = -sincos(phase + p0.z, p0z_cos);

  /* Decode the nine raw measurements (3 sub-frames per frequency). */
  const float3 v0 = float3(decodePixelMeasurement(data, lut11to16, 0u, x, y_in),
                           decodePixelMeasurement(data, lut11to16, 1u, x, y_in),
                           decodePixelMeasurement(data, lut11to16, 2u, x, y_in));
  const float3 v1 = float3(decodePixelMeasurement(data, lut11to16, 3u, x, y_in),
                           decodePixelMeasurement(data, lut11to16, 4u, x, y_in),
                           decodePixelMeasurement(data, lut11to16, 5u, x, y_in));
  const float3 v2 = float3(decodePixelMeasurement(data, lut11to16, 6u, x, y_in),
                           decodePixelMeasurement(data, lut11to16, 7u, x, y_in),
                           decodePixelMeasurement(data, lut11to16, 8u, x, y_in));

  /* Per-frequency multipliers for the ab (amplitude-bias) computation. */
  const float3 ab_mult = float3(params.ab_multiplier_per_frq0,
                                params.ab_multiplier_per_frq1,
                                params.ab_multiplier_per_frq2);

  /* Compute complex IR vectors a (real) and b (imaginary). */
  float3 a = float3(dot(v0, p0x_cos), dot(v1, p0y_cos), dot(v2, p0z_cos)) * ab_mult;
  float3 b = float3(dot(v0, p0x_sin), dot(v1, p0y_sin), dot(v2, p0z_sin)) * ab_mult;

  /* Zero out invalid pixels. */
  a = select(a, float3(0.0f), invalid);
  b = select(b, float3(0.0f), invalid);
  float3 n = sqrt(a * a + b * b);

  /* Detect saturated measurements (raw value == 32767 in any sub-frame). */
  const bool sat0 = any(v0 == float3(32767.0f));
  const bool sat1 = any(v1 == float3(32767.0f));
  const bool sat2 = any(v2 == float3(32767.0f));
  const bool3 saturated = bool3(sat0, sat1, sat2);

  /* Zero a/b for saturated frequencies; IR reports saturation as 65535. */
  a_out[i] = select(a, float3(0.0f), saturated);
  b_out[i] = select(b, float3(0.0f), saturated);
  n_out[i] = n;

  float3 n_or_sat = select(n, float3(65535.0f), saturated);
  ir_out[i] = min(dot(n_or_sat, float3(0.333333333f * params.ab_multiplier * params.ab_output_multiplier)), 65535.0f);
}

/*******************************************************************************
 * Filter pixel stage 1: joint bilateral filter on the complex IR vectors.
 *
 * Smooths the (a, b) outputs from stage 1 while preserving edges.
 * Also computes the max_edge_test flag used by stage 2 edge filtering.
 ******************************************************************************/
kernel void filterPixelStage1(
    device const float3 *a           [[ buffer(0) ]],
    device const float3 *b           [[ buffer(1) ]],
    device const float3 *n           [[ buffer(2) ]],
    device float3       *a_out       [[ buffer(3) ]],
    device float3       *b_out       [[ buffer(4) ]],
    device uchar        *max_edge_test [[ buffer(5) ]],
    constant MetalDepthParams &params [[ buffer(6) ]],
    uint i [[ thread_position_in_grid ]])
{
  const uint x = i % 512u;
  const uint y = i / 512u;

  const float3 self_a = a[i];
  const float3 self_b = b[i];

  const float gaussian[9] = {
    params.gaussian_kernel0, params.gaussian_kernel1, params.gaussian_kernel2,
    params.gaussian_kernel3, params.gaussian_kernel4, params.gaussian_kernel5,
    params.gaussian_kernel6, params.gaussian_kernel7, params.gaussian_kernel8
  };

  /* Border pixels: pass through without filtering and mark as valid edge. */
  if(x < 1u || y < 1u || x > 510u || y > 422u)
  {
    a_out[i] = self_a;
    b_out[i] = self_b;
    max_edge_test[i] = 1;
    return;
  }

  float3 threshold = float3(params.joint_bilateral_threshold);
  float3 joint_bilateral_exp = float3(params.joint_bilateral_exp);

  const float3 self_norm = n[i];
  const float3 self_normalized_a = self_a / self_norm;
  const float3 self_normalized_b = self_b / self_norm;

  /* If the centre pixel's signal is too weak, disable distance weighting. */
  const bool3 c0 = self_norm * self_norm < threshold;
  threshold = select(threshold, float3(0.0f), c0);
  joint_bilateral_exp = select(joint_bilateral_exp, float3(0.0f), c0);

  float3 weight_acc = float3(0.0f);
  float3 weighted_a_acc = float3(0.0f);
  float3 weighted_b_acc = float3(0.0f);
  float3 dist_acc = float3(0.0f);

  for(int yi = -1, j = 0; yi < 2; ++yi)
  {
    uint i_other = uint(int(y) + yi) * 512u + x - 1u;

    for(int xi = -1; xi < 2; ++xi, ++j, ++i_other)
    {
      const float3 other_a = a[i_other];
      const float3 other_b = b[i_other];
      const float3 other_norm = n[i_other];
      const float3 other_normalized_a = other_a / other_norm;
      const float3 other_normalized_b = other_b / other_norm;

      const bool3 c1 = other_norm * other_norm < threshold;

      const float3 dist = 0.5f * (1.0f - (self_normalized_a * other_normalized_a +
                                           self_normalized_b * other_normalized_b));
      const float3 weight = select(gaussian[j] * exp(-1.442695f * joint_bilateral_exp * dist),
                                   float3(0.0f), c1);

      weighted_a_acc += weight * other_a;
      weighted_b_acc += weight * other_b;
      weight_acc += weight;
      dist_acc += select(dist, float3(0.0f), c1);
    }
  }

  const bool3 c2 = weight_acc > float3(0.0f);
  a_out[i] = select(float3(0.0f), weighted_a_acc / weight_acc, c2);
  b_out[i] = select(float3(0.0f), weighted_b_acc / weight_acc, c2);

  max_edge_test[i] = all(dist_acc < float3(params.joint_bilateral_max_edge)) ? 1u : 0u;
}

/*******************************************************************************
 * Process pixel stage 2: depth calculation from unwrapped phase.
 *
 * Implements three-frequency phase disambiguation and converts the final phase
 * to depth in millimetres using the x/z calibration tables.
 ******************************************************************************/
kernel void processPixelStage2(
    device const float3 *a_in        [[ buffer(0) ]],
    device const float3 *b_in        [[ buffer(1) ]],
    device const float  *x_table     [[ buffer(2) ]],
    device const float  *z_table     [[ buffer(3) ]],
    device float        *depth       [[ buffer(4) ]],
    device float        *ir_sums     [[ buffer(5) ]],
    constant MetalDepthParams &params [[ buffer(6) ]],
    uint i [[ thread_position_in_grid ]])
{
  float3 a = a_in[i];
  float3 b = b_in[i];

  float3 phase = atan2(b, a);
  phase = select(phase, phase + 2.0f * M_PI_F, phase < float3(0.0f));
  phase = select(phase, float3(0.0f), isnan(phase));
  float3 ir = sqrt(a * a + b * b) * params.ab_multiplier;

  float ir_sum = ir.x + ir.y + ir.z;
  float ir_min = min(ir.x, min(ir.y, ir.z));

  float phase_final = 0.0f;

  if(ir_min >= params.individual_ab_threshold && ir_sum >= params.ab_threshold)
  {
    float3 t = phase / (2.0f * M_PI_F) * float3(3.0f, 15.0f, 2.0f);

    float t0 = t.x;
    float t1 = t.y;
    float t2 = t.z;

    float t5 = (floor((t1 - t0) * 0.333333f + 0.5f) * 3.0f + t0);
    float t3 = (-t2 + t5);
    float t4 = t3 * 2.0f;

    bool c1 = t4 >= -t4;

    float f1 = c1 ? 2.0f : -2.0f;
    float f2 = c1 ? 0.5f : -0.5f;
    t3 *= f2;
    t3 = (t3 - floor(t3)) * f1;

    bool c2 = 0.5f < abs(t3) && abs(t3) < 1.5f;

    float t6 = c2 ? t5 + 15.0f : t5;
    float t7 = c2 ? t1 + 15.0f : t1;

    float t8 = (floor((-t2 + t6) * 0.5f + 0.5f) * 2.0f + t2) * 0.5f;

    t6 *= 0.333333f;
    t7 *= 0.066667f;

    float t9 = (t8 + t6 + t7);
    float t10 = t9 * 0.333333f;

    t6 *= 2.0f * M_PI_F;
    t7 *= 2.0f * M_PI_F;
    t8 *= 2.0f * M_PI_F;

    float t8_new = t7 * 0.826977f - t8 * 0.110264f;
    float t6_new = t8 * 0.551318f - t6 * 0.826977f;
    float t7_new = t6 * 0.110264f - t7 * 0.551318f;

    t8 = t8_new;
    t6 = t6_new;
    t7 = t7_new;

    float norm = t8 * t8 + t6 * t6 + t7 * t7;
    float mask = t9 >= 0.0f ? 1.0f : 0.0f;
    t10 *= mask;

    bool slope_positive = 0 < params.ab_confidence_slope;

    float ir_max = max(ir.x, max(ir.y, ir.z));
    float ir_x = slope_positive ? ir_min : ir_max;

    ir_x = log(ir_x);
    ir_x = (ir_x * params.ab_confidence_slope * 0.301030f + params.ab_confidence_offset) * 3.321928f;
    ir_x = exp(ir_x);
    ir_x = clamp(ir_x, params.min_dealias_confidence, params.max_dealias_confidence);
    ir_x *= ir_x;

    float mask2 = ir_x >= norm ? 1.0f : 0.0f;
    float t11 = t10 * mask2;

    float mask3 = params.max_dealias_confidence * params.max_dealias_confidence >= norm ? 1.0f : 0.0f;
    t10 *= mask3;
    phase_final = t11;
  }

  float zmultiplier = z_table[i];
  float xmultiplier = x_table[i];

  phase_final = 0.0f < phase_final ? phase_final + params.phase_offset : phase_final;

  float depth_linear = zmultiplier * phase_final;
  float max_depth = phase_final * params.unambigious_dist * 2.0f;

  bool cond1 = 0.0f < depth_linear && 0.0f < max_depth;

  xmultiplier = (xmultiplier * 90.0f) / (max_depth * max_depth * 8192.0f);

  float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1.0f);
  depth_fit = depth_fit < 0.0f ? 0.0f : depth_fit;

  float d = cond1 ? depth_fit : depth_linear;
  depth[i] = d;
  ir_sums[i] = ir_sum;
}

/*******************************************************************************
 * Filter pixel stage 2: edge-aware depth filter.
 *
 * Removes depth measurements at depth discontinuities where the IR signal
 * variance indicates an unreliable reading.
 ******************************************************************************/
kernel void filterPixelStage2(
    device const float *depth        [[ buffer(0) ]],
    device const float *ir_sums      [[ buffer(1) ]],
    device const uchar *max_edge_test [[ buffer(2) ]],
    device float       *filtered     [[ buffer(3) ]],
    constant MetalDepthParams &params [[ buffer(4) ]],
    uint i [[ thread_position_in_grid ]])
{
  const uint x = i % 512u;
  const uint y = i / 512u;

  const float raw_depth = depth[i];
  const float ir_sum = ir_sums[i];
  const uchar edge_test = max_edge_test[i];

  if(raw_depth >= params.min_depth && raw_depth <= params.max_depth)
  {
    if(x < 1u || y < 1u || x > 510u || y > 422u)
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
        uint i_other = uint(int(y) + yi) * 512u + x - 1u;

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
      float edge_avg = max(ir_sum_acc / 9.0f, params.edge_ab_avg_min_value);
      tmp0 /= edge_avg;

      float abs_min_diff = abs(raw_depth - min_depth);
      float abs_max_diff = abs(raw_depth - max_depth);

      float avg_diff = (abs_min_diff + abs_max_diff) * 0.5f;
      float max_abs_diff = max(abs_min_diff, abs_max_diff);

      bool cond0 =
          0.0f < raw_depth &&
          tmp0 >= params.edge_ab_std_dev_threshold &&
          params.edge_close_delta_threshold < abs_min_diff &&
          params.edge_far_delta_threshold < abs_max_diff &&
          params.edge_max_delta_threshold < max_abs_diff &&
          params.edge_avg_delta_threshold < avg_diff;

      if(!cond0)
      {
        if(edge_test != 0)
        {
          /* tmp1 and edge_count would be used for a more sophisticated edge
           * count filter; currently edge_count is always 0 so this path
           * always passes depth through. */
          float edge_count = 0.0f;
          filtered[i] = edge_count > params.max_edge_count ? 0.0f : raw_depth;
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
