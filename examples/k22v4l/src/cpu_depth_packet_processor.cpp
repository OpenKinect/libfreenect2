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

#if defined(WIN32)
#define _USE_MATH_DEFINES
#include <math.h>
#endif

namespace libfreenect2
{

bool loadBufferFromFile2(const std::string& filename, unsigned char *buffer, size_t n)
{
  bool success = true;
  std::ifstream in(filename.c_str());

  in.read(reinterpret_cast<char*>(buffer), n);
  success = in.gcount() == n;

  in.close();

  return success;
}

inline int bfi(int width, int offset, int src2, int src3)
{
  int bitmask = (((1 << width)-1) << offset) & 0xffffffff;
  return ((src2 << offset) & bitmask) | (src3 & ~bitmask);
}

class CpuDepthPacketProcessorImpl
{
public:
  cv::Mat p0_table0, p0_table1, p0_table2, x_table, z_table;

  int16_t lut11to16[2048];

  float trig_table0[512*424][6];
  float trig_table1[512*424][6];
  float trig_table2[512*424][6];

  double timing_acc;
  double timing_acc_n;

  double timing_current_start;

  bool enable_bilateral_filter, enable_edge_filter;
  DepthPacketProcessor::Parameters params;

  Frame *ir_frame, *depth_frame;

  bool flip_ptables;

  CpuDepthPacketProcessorImpl()
  {
    newIrFrame();
    newDepthFrame();

    timing_acc = 0.0;
    timing_acc_n = 0.0;
    timing_current_start = 0.0;

    enable_bilateral_filter = true;
    enable_edge_filter = true;

    flip_ptables = true;
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
      std::cout << "[CpuDepthPacketProcessor] avg. time: " << (avg * 1000) << "ms -> ~" << (1.0/avg) << "Hz" << std::endl;
      timing_acc = 0.0;
      timing_acc_n = 0.0;
    }
  }

  void newIrFrame()
  {
    ir_frame = new Frame(512, 424, 4);
    //ir_frame = new Frame(512, 424, 12);
  }

  void newDepthFrame()
  {
    depth_frame = new Frame(512, 424, 4);
  }

  int32_t decodePixelMeasurement(unsigned char* data, int sub, int x, int y)
  {
    // 298496 = 512 * 424 * 11 / 8 = number of bytes per sub image
    uint16_t *ptr = reinterpret_cast<uint16_t *>(data + 298496 * sub);
    int i = y < 212 ? y + 212 : 423 - y;
    ptr += 352*i;

    /**
     r1.yz = r2.xxyx < l(0, 1, 0, 0) // ilt
     r1.y = r1.z | r1.y // or
     */
    bool r1y = x < 1 || y < 0;
    /*
    r1.zw = l(0, 0, 510, 423) < r2.xxxy // ilt
    r1.z = r1.w | r1.z // or
    */
    bool r1z = 510 < x || 423 < y;
    /*
    r1.y = r1.z | r1.y // or
    */
    r1y = r1y || r1z;
    /*
    r1.y = r1.y & l(0x1fffffff) // and
    */
    int r1yi = r1y ? 0xffffffff : 0x0;
    r1yi &= 0x1fffffff;

    /*
    bfi r1.z, l(2), l(7), r2.x, l(0)
    ushr r1.w, r2.x, l(2)
    r1.z = r1.w + r1.z // iadd
    */
    int r1zi = bfi(2, 7, x, 0);
    int r1wi = x >> 2;
    r1zi = r1wi + r1zi;

    /*
    imul null, r1.z, r1.z, l(11)
    ushr r1.w, r1.z, l(4)
    r1.y = r1.w + r1.y // iadd
    r1.w = r1.y + l(1) // iadd
    r1.z = r1.z & l(15) // and
    r4.w = -r1.z + l(16) // iadd
     */
    r1zi = (r1zi * 11L) & 0xffffffff;
    r1wi = r1zi >> 4;
    r1yi = r1yi + r1wi;
    r1zi = r1zi & 15;
    int r4wi = -r1zi + 16;

    if(r1yi > 352)
    {
      return lut11to16[0];
    }

    int i1 = ptr[r1yi];
    int i2 = ptr[r1yi + 1];
    i1 = i1 >> r1zi;
    i2 = i2 << r4wi;

    return lut11to16[((i1 | i2) & 2047)];
  }

  void fill_trig_tables(cv::Mat& p0table, float trig_table[512*424][6])
  {
    for (int i = 0; i < 512*424; i++)
    {
      float p0 = -((float)p0table.at<uint16_t>(i)) * 0.000031 * M_PI;

      float tmp0 = p0 + params.phase_in_rad[0];
      float tmp1 = p0 + params.phase_in_rad[1];
      float tmp2 = p0 + params.phase_in_rad[2];

      trig_table[i][0] = std::cos(tmp0);
      trig_table[i][1] = std::cos(tmp1);
      trig_table[i][2] = std::cos(tmp2);

      trig_table[i][3] = std::sin(-tmp0);
      trig_table[i][4] = std::sin(-tmp1);
      trig_table[i][5] = std::sin(-tmp2);
    }
  }

  void processMeasurementTriple(float trig_table[512*424][6], float abMultiplierPerFrq, int x, int y, const int32_t* m, float* m_out)
  {
    int offset = y * 512 + x;
    float cos_tmp0 = trig_table[offset][0];
    float cos_tmp1 = trig_table[offset][1];
    float cos_tmp2 = trig_table[offset][2];

    float sin_negtmp0 = trig_table[offset][3];
    float sin_negtmp1 = trig_table[offset][4];
    float sin_negtmp2 = trig_table[offset][5];

    float zmultiplier = z_table.at<float>(y, x);
    bool cond0 = 0 < zmultiplier;
    bool cond1 = (m[0] == 32767 || m[1] == 32767 || m[2] == 32767) && cond0;

    // formula given in Patent US 8,587,771 B2
    float tmp3 = cos_tmp0 * m[0] + cos_tmp1 * m[1] + cos_tmp2 * m[2];
    float tmp4 = sin_negtmp0 * m[0] + sin_negtmp1 * m[1] + sin_negtmp2 * m[2];

    // only if modeMask & 32 != 0;
    if(true)//(modeMask & 32) != 0)
    {
        tmp3 *= abMultiplierPerFrq;
        tmp4 *= abMultiplierPerFrq;
    }
    float tmp5 = std::sqrt(tmp3 * tmp3 + tmp4 * tmp4) * params.ab_multiplier;

    // invalid pixel because zmultiplier < 0 ??
    tmp3 = cond0 ? tmp3 : 0;
    tmp4 = cond0 ? tmp4 : 0;
    tmp5 = cond0 ? tmp5 : 0;

    // invalid pixel because saturated?
    tmp3 = !cond1 ? tmp3 : 0;
    tmp4 = !cond1 ? tmp4 : 0;
    tmp5 = !cond1 ? tmp5 : 65535.0; // some kind of norm calculated from tmp3 and tmp4

    m_out[0] = tmp3; // ir image a
    m_out[1] = tmp4; // ir image b
    m_out[2] = tmp5; // ir amplitude
  }

  void transformMeasurements(float* m)
  {
    float tmp0 = std::atan2((m[1]), (m[0]));
    tmp0 = tmp0 < 0 ? tmp0 + M_PI * 2.0f : tmp0;
    tmp0 = (tmp0 != tmp0) ? 0 : tmp0;

    float tmp1 = std::sqrt(m[0] * m[0] + m[1] * m[1]) * params.ab_multiplier;

    m[0] = tmp0; // phase
    m[1] = tmp1; // ir amplitude - (possibly bilateral filtered)
  }

  void processPixelStage1(int x, int y, unsigned char* data, float *m0_out, float *m1_out, float *m2_out)
  {
    int32_t m0_raw[3], m1_raw[3], m2_raw[3];

    m0_raw[0] = decodePixelMeasurement(data, 0, x, y);
    m0_raw[1] = decodePixelMeasurement(data, 1, x, y);
    m0_raw[2] = decodePixelMeasurement(data, 2, x, y);
    m1_raw[0] = decodePixelMeasurement(data, 3, x, y);
    m1_raw[1] = decodePixelMeasurement(data, 4, x, y);
    m1_raw[2] = decodePixelMeasurement(data, 5, x, y);
    m2_raw[0] = decodePixelMeasurement(data, 6, x, y);
    m2_raw[1] = decodePixelMeasurement(data, 7, x, y);
    m2_raw[2] = decodePixelMeasurement(data, 8, x, y);

    processMeasurementTriple(trig_table0, params.ab_multiplier_per_frq[0], x, y, m0_raw, m0_out);
    processMeasurementTriple(trig_table1, params.ab_multiplier_per_frq[1], x, y, m1_raw, m1_out);
    processMeasurementTriple(trig_table2, params.ab_multiplier_per_frq[2], x, y, m2_raw, m2_out);
  }

  void filterPixelStage1(int x, int y, const cv::Mat& m, float* m_out, bool& bilateral_max_edge_test)
  {
    const float *m_ptr = m.ptr<float>(y, x);
    bilateral_max_edge_test = true;

    if(x < 1 || y < 1 || x > 510 || y > 422)
    {
      for(int i = 0; i < 9; ++i)
        m_out[i] = m_ptr[i];
    }
    else
    {
      float m_normalized[2];
      float other_m_normalized[2];

      int offset = 0;

      for(int i = 0; i < 3; ++i, m_ptr += 3, m_out += 3, offset += 3)
      {
        float norm2 = m_ptr[0] * m_ptr[0] + m_ptr[1] * m_ptr[1];
        float inv_norm = 1.0f / std::sqrt(norm2);
        inv_norm = (inv_norm == inv_norm) ? inv_norm : std::numeric_limits<float>::infinity();

        m_normalized[0] = m_ptr[0] * inv_norm;
        m_normalized[1] = m_ptr[1] * inv_norm;

        int j = 0;

        float weight_acc = 0.0f;
        float weighted_m_acc[2] = {0.0f, 0.0f};

        float threshold = (params.joint_bilateral_ab_threshold * params.joint_bilateral_ab_threshold) / (params.ab_multiplier * params.ab_multiplier);
        float joint_bilateral_exp = params.joint_bilateral_exp;

        if(norm2 < threshold)
        {
          threshold = 0.0f;
          joint_bilateral_exp = 0.0f;
        }

        float dist_acc = 0.0f;

        for(int yi = -1; yi < 2; ++yi)
        {
          for(int xi = -1; xi < 2; ++xi, ++j)
          {
            if(yi == 0 && xi == 0)
            {
              weight_acc += params.gaussian_kernel[j];

              weighted_m_acc[0] += params.gaussian_kernel[j] * m_ptr[0];
              weighted_m_acc[1] += params.gaussian_kernel[j] * m_ptr[1];
              continue;
            }

            const float *other_m_ptr = m.ptr<float>(y + yi, x + xi) + offset;
            float other_norm2 = other_m_ptr[0] * other_m_ptr[0] + other_m_ptr[1] * other_m_ptr[1];
            // TODO: maybe fix numeric problems when norm = 0 - original code uses reciprocal square root, which returns +inf for +0
            float other_inv_norm = 1.0f / std::sqrt(other_norm2);
            other_inv_norm = (other_inv_norm == other_inv_norm) ? other_inv_norm : std::numeric_limits<float>::infinity();

            other_m_normalized[0] = other_m_ptr[0] * other_inv_norm;
            other_m_normalized[1] = other_m_ptr[1] * other_inv_norm;

            float dist = -(other_m_normalized[0] * m_normalized[0] + other_m_normalized[1] * m_normalized[1]);
            dist += 1.0f;
            dist *= 0.5f;

            float weight = 0.0f;

            if(other_norm2 >= threshold)
            {
              weight = (params.gaussian_kernel[j] * std::exp(-1.442695f * joint_bilateral_exp * dist));
              dist_acc += dist;
            }

            weighted_m_acc[0] += weight * other_m_ptr[0];
            weighted_m_acc[1] += weight * other_m_ptr[1];

            weight_acc += weight;
          }
        }

        bilateral_max_edge_test = bilateral_max_edge_test && dist_acc < params.joint_bilateral_max_edge;

        m_out[0] = 0.0f < weight_acc ? weighted_m_acc[0] / weight_acc : 0.0f;
        m_out[1] = 0.0f < weight_acc ? weighted_m_acc[1] / weight_acc : 0.0f;
        m_out[2] = m_ptr[2];
      }
    }
  }

  void processPixelStage2(int x, int y, float *m0, float *m1, float *m2, float *ir_out, float *depth_out, float *ir_sum_out)
  {
    //// 10th measurement
    //float m9 = 1; // decodePixelMeasurement(data, 9, x, y);
    //
    //// WTF?
    //bool cond0 = zmultiplier == 0 || (m9 >= 0 && m9 < 32767);
    //m9 = std::max(-m9, m9);
    //// if m9 is positive or pixel is invalid (zmultiplier) we set it to 0 otherwise to its absolute value O.o
    //m9 = cond0 ? 0 : m9;

    transformMeasurements(m0);
    transformMeasurements(m1);
    transformMeasurements(m2);

    float ir_sum = m0[1] + m1[1] + m2[1];

    float phase;
    // if(DISABLE_DISAMBIGUATION)
    if(false)
    {
        //r0.yz = r3.zx + r4.zx // add
        //r0.yz = r5.xz + r0.zy // add
        float phase = m0[0] + m1[0] + m2[0]; // r0.y
        float tmp1 = m0[2] + m1[2] + m2[2];  // r0.z

        //r7.xyz = r3.zxy + r4.zxy // add
        //r4.xyz = r5.zyx + r7.xzy // add
        float tmp2 = m0[0] + m1[0] + m2[0]; // r4.z
        //r3.zw = r4.xy // mov
        float tmp3 = m0[2] + m1[2] + m2[2]; // r3.z
        float tmp4 = m0[1] + m1[1] + m2[1]; // r3.w
    }
    else
    {
      float ir_min = std::min(std::min(m0[1], m1[1]), m2[1]);

      if (ir_min < params.individual_ab_threshold || ir_sum < params.ab_threshold)
      {
        phase = 0;
      }
      else
      {
        float t0 = m0[0] / (2.0f * M_PI) * 3.0f;
        float t1 = m1[0] / (2.0f * M_PI) * 15.0f;
        float t2 = m2[0] / (2.0f * M_PI) * 2.0f;

        float t5 = (std::floor((t1 - t0) * 0.333333f + 0.5f) * 3.0f + t0);
        float t3 = (-t2 + t5);
        float t4 = t3 * 2.0f;

        bool c1 = t4 >= -t4; // true if t4 positive

        float f1 = c1 ? 2.0f : -2.0f;
        float f2 = c1 ? 0.5f : -0.5f;
        t3 *= f2;
        t3 = (t3 - std::floor(t3)) * f1;

        bool c2 = 0.5f < std::abs(t3) && std::abs(t3) < 1.5f;

        float t6 = c2 ? t5 + 15.0f : t5;
        float t7 = c2 ? t1 + 15.0f : t1;

        float t8 = (std::floor((-t2 + t6) * 0.5f + 0.5f) * 2.0f + t2) * 0.5f;

        t6 *= 0.333333f; // = / 3
        t7 *= 0.066667f; // = / 15

        float t9 = (t8 + t6 + t7); // transformed phase measurements (they are transformed and divided by the values the original values were multiplied with)
        float t10 = t9 * 0.333333f; // some avg

        t6 *= 2.0f * M_PI;
        t7 *= 2.0f * M_PI;
        t8 *= 2.0f * M_PI;

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

        bool slope_positive = 0 < params.ab_confidence_slope;

        float ir_min_ = std::min(std::min(m0[1], m1[1]), m2[1]);
        float ir_max_ = std::max(std::max(m0[1], m1[1]), m2[1]);

        float ir_x = slope_positive ? ir_min_ : ir_max_;

        ir_x = std::log(ir_x);
        ir_x = (ir_x * params.ab_confidence_slope * 0.301030f + params.ab_confidence_offset) * 3.321928f;
        ir_x = std::exp(ir_x);
        ir_x = std::min(params.max_dealias_confidence, std::max(params.min_dealias_confidence, ir_x));
        ir_x *= ir_x;

        float mask2 = ir_x >= norm ? 1.0f : 0.0f;

        float t11 = t10 * mask2;

        float mask3 = params.max_dealias_confidence * params.max_dealias_confidence >= norm ? 1.0f : 0.0f;
        t10 *= mask3;
        phase = true/*(modeMask & 2) != 0*/ ? t11 : t10;
      }
    }

    // this seems to be the phase to depth mapping :)
    float zmultiplier = z_table.at<float>(y, x);
    float xmultiplier = x_table.at<float>(y, x);

    phase = 0 < phase ? phase + params.phase_offset : phase;

    float depth_linear = zmultiplier * phase;
    float max_depth = phase * params.unambigious_dist * 2;

    bool cond1 = /*(modeMask & 32) != 0*/ true && 0 < depth_linear && 0 < max_depth;

    xmultiplier = (xmultiplier * 90) / (max_depth * max_depth * 8192.0);

    float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1);

    depth_fit = depth_fit < 0 ? 0 : depth_fit;
    float depth = cond1 ? depth_fit : depth_linear; // r1.y -> later r2.z

    // depth
    *depth_out = depth;
    if(ir_sum_out != 0)
    {
      *ir_sum_out = ir_sum;
    }

    // ir
    //*ir_out = std::min((m1[2]) * ab_output_multiplier, 65535.0f);
    // ir avg
    *ir_out = std::min((m0[2] + m1[2] + m2[2]) * 0.3333333f * params.ab_output_multiplier, 65535.0f);
    //ir_out[0] = std::min(m0[2] * ab_output_multiplier, 65535.0f);
    //ir_out[1] = std::min(m1[2] * ab_output_multiplier, 65535.0f);
    //ir_out[2] = std::min(m2[2] * ab_output_multiplier, 65535.0f);
  }

  void filterPixelStage2(int x, int y, cv::Mat &m, bool max_edge_test_ok, float *depth_out)
  {
    cv::Vec3f &depth_and_ir_sum = m.at<cv::Vec3f>(y, x);
    float &raw_depth = depth_and_ir_sum.val[0], &ir_sum = depth_and_ir_sum.val[2];

    if(raw_depth >= params.min_depth && raw_depth <= params.max_depth)
    {
      if(x < 1 || y < 1 || x > 510 || y > 422)
      {
        *depth_out = raw_depth;
      }
      else
      {
        float ir_sum_acc = ir_sum, squared_ir_sum_acc = ir_sum * ir_sum, min_depth = raw_depth, max_depth = raw_depth;

        for(int yi = -1; yi < 2; ++yi)
        {
          for(int xi = -1; xi < 2; ++xi)
          {
            if(yi == 0 && xi == 0) continue;

            cv::Vec3f &other = m.at<cv::Vec3f>(y + yi, x + xi);

            ir_sum_acc += other.val[2];
            squared_ir_sum_acc += other.val[2] * other.val[2];

            if(0.0f < other.val[1])
            {
              min_depth = std::min(min_depth, other.val[1]);
              max_depth = std::max(max_depth, other.val[1]);
            }
          }
        }

        float tmp0 = std::sqrt(squared_ir_sum_acc * 9.0f - ir_sum_acc * ir_sum_acc) / 9.0f;
        float edge_avg = std::max(ir_sum_acc / 9.0f, params.edge_ab_avg_min_value);
        tmp0 /= edge_avg;

        float abs_min_diff = std::abs(raw_depth - min_depth);
        float abs_max_diff = std::abs(raw_depth - max_depth);

        float avg_diff = (abs_min_diff + abs_max_diff) * 0.5f;
        float max_abs_diff = std::max(abs_min_diff, abs_max_diff);

        bool cond0 =
            0.0f < raw_depth &&
            tmp0 >= params.edge_ab_std_dev_threshold &&
            params.edge_close_delta_threshold < abs_min_diff &&
            params.edge_far_delta_threshold < abs_max_diff &&
            params.edge_max_delta_threshold < max_abs_diff &&
            params.edge_avg_delta_threshold < avg_diff;

        *depth_out = cond0 ? 0.0f : raw_depth;

        if(!cond0)
        {
          if(max_edge_test_ok)
          {
            float tmp1 = 1500.0f > raw_depth ? 30.0f : 0.02f * raw_depth;
            float edge_count = 0.0f;

            *depth_out = edge_count > params.max_edge_count ? 0.0f : raw_depth;
          }
          else
          {
            *depth_out = !max_edge_test_ok ? 0.0f : raw_depth;
            *depth_out = true ? *depth_out : raw_depth;
          }
        }
      }
    }
    else
    {
      *depth_out = 0.0f;
    }

    // override raw depth
    depth_and_ir_sum.val[0] = depth_and_ir_sum.val[1];
  }
};

CpuDepthPacketProcessor::CpuDepthPacketProcessor() :
    impl_(new CpuDepthPacketProcessorImpl())
{
}

CpuDepthPacketProcessor::~CpuDepthPacketProcessor()
{
  delete impl_;
}

void CpuDepthPacketProcessor::setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config)
{
  DepthPacketProcessor::setConfiguration(config);
  
  impl_->params.min_depth = config.MinDepth * 1000.0f;
  impl_->params.max_depth = config.MaxDepth * 1000.0f;
  impl_->enable_bilateral_filter = config.EnableBilateralFilter;
  impl_->enable_edge_filter = config.EnableEdgeAwareFilter;
}

void CpuDepthPacketProcessor::loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length)
{
  // TODO: check known header fields (headersize, tablesize)
  libfreenect2::protocol::P0TablesResponse* p0table = (libfreenect2::protocol::P0TablesResponse*)buffer;

  if(buffer_length < sizeof(libfreenect2::protocol::P0TablesResponse))
  {
    std::cerr << "[CpuDepthPacketProcessor::loadP0TablesFromCommandResponse] P0Table response too short!" << std::endl;
    return;
  }

  if(impl_->flip_ptables)
  {
    cv::flip(cv::Mat(424, 512, CV_16UC1, p0table->p0table0), impl_->p0_table0, 0);
    cv::flip(cv::Mat(424, 512, CV_16UC1, p0table->p0table1), impl_->p0_table1, 0);
    cv::flip(cv::Mat(424, 512, CV_16UC1, p0table->p0table2), impl_->p0_table2, 0);

    impl_->fill_trig_tables(impl_->p0_table0, impl_->trig_table0);
    impl_->fill_trig_tables(impl_->p0_table1, impl_->trig_table1);
    impl_->fill_trig_tables(impl_->p0_table2, impl_->trig_table2);
  }
  else
  {
    cv::Mat(424, 512, CV_16UC1, p0table->p0table0).copyTo(impl_->p0_table0);
    cv::Mat(424, 512, CV_16UC1, p0table->p0table1).copyTo(impl_->p0_table1);
    cv::Mat(424, 512, CV_16UC1, p0table->p0table2).copyTo(impl_->p0_table2);
  }
}
void CpuDepthPacketProcessor::loadP0TablesFromFiles(const char* p0_filename, const char* p1_filename, const char* p2_filename)
{
  cv::Mat p0_table0(424, 512, CV_16UC1);
  if(!loadBufferFromFile2(p0_filename, p0_table0.data, p0_table0.total() * p0_table0.elemSize()))
  {
    std::cerr << "[CpuDepthPacketProcessor::loadP0TablesFromFiles] Loading p0table 0 from '" << p0_filename << "' failed!" << std::endl;
  }

  cv::Mat p0_table1(424, 512, CV_16UC1);
  if(!loadBufferFromFile2(p1_filename, p0_table1.data, p0_table1.total() * p0_table1.elemSize()))
  {
    std::cerr << "[CpuDepthPacketProcessor::loadP0TablesFromFiles] Loading p0table 1 from '" << p1_filename << "' failed!" << std::endl;
  }

  cv::Mat p0_table2(424, 512, CV_16UC1);
  if(!loadBufferFromFile2(p2_filename, p0_table2.data, p0_table2.total() * p0_table2.elemSize()))
  {
    std::cerr << "[CpuDepthPacketProcessor::loadP0TablesFromFiles] Loading p0table 2 from '" << p2_filename << "' failed!" << std::endl;
  }

  if(impl_->flip_ptables)
  {
    cv::flip(p0_table0, impl_->p0_table0, 0);
    cv::flip(p0_table1, impl_->p0_table1, 0);
    cv::flip(p0_table2, impl_->p0_table2, 0);

    impl_->fill_trig_tables(impl_->p0_table0, impl_->trig_table0);
    impl_->fill_trig_tables(impl_->p0_table1, impl_->trig_table1);
    impl_->fill_trig_tables(impl_->p0_table2, impl_->trig_table2);
  }
  else
  {
    impl_->fill_trig_tables(p0_table0, impl_->trig_table0);
    impl_->fill_trig_tables(p0_table1, impl_->trig_table1);
    impl_->fill_trig_tables(p0_table2, impl_->trig_table2);
  }
}

void CpuDepthPacketProcessor::loadXTableFromFile(const char* filename)
{
  impl_->x_table.create(424, 512, CV_32FC1);
  const unsigned char *data;
  size_t length;

  if(loadResource("xTable.bin", &data, &length))
  {
    std::copy(data, data + length, impl_->x_table.data);
  }
  else
  {
    std::cerr << "[CpuDepthPacketProcessor::loadXTableFromFile] Loading xtable from resource 'xTable.bin' failed!" << std::endl;
  }
}

void CpuDepthPacketProcessor::loadZTableFromFile(const char* filename)
{
  impl_->z_table.create(424, 512, CV_32FC1);

  const unsigned char *data;
  size_t length;

  if(loadResource("zTable.bin", &data, &length))
  {
    std::copy(data, data + length, impl_->z_table.data);
  }
  else
  {
    std::cerr << "[CpuDepthPacketProcessor::loadZTableFromFile] Loading ztable from resource 'zTable.bin' failed!" << std::endl;
  }
}

void CpuDepthPacketProcessor::load11To16LutFromFile(const char* filename)
{
  const unsigned char *data;
  size_t length;

  if(loadResource("11to16.bin", &data, &length))
  {
    std::copy(data, data + length, reinterpret_cast<unsigned char*>(impl_->lut11to16));
  }
  else
  {
    std::cerr << "[CpuDepthPacketProcessor::load11To16LutFromFile] Loading 11to16 lut from resource '11to16.bin' failed!" << std::endl;
  }
}

void CpuDepthPacketProcessor::process(const DepthPacket &packet)
{
  if(listener_ == 0) return;

  impl_->startTiming();

  cv::Mat m = cv::Mat::zeros(424, 512, CV_32FC(9)), m_filtered = cv::Mat::zeros(424, 512, CV_32FC(9)), m_max_edge_test = cv::Mat::ones(424, 512, CV_8UC1);

  float *m_ptr = m.ptr<float>();

  for(int y = 0; y < 424; ++y)
    for(int x = 0; x < 512; ++x, m_ptr += 9)
    {
      impl_->processPixelStage1(x, y, packet.buffer, m_ptr + 0, m_ptr + 3, m_ptr + 6);
    }

  // bilateral filtering
  if(impl_->enable_bilateral_filter)
  {
    float *m_filtered_ptr = m_filtered.ptr<float>();
    unsigned char *m_max_edge_test_ptr = m_max_edge_test.ptr<unsigned char>();

    for(int y = 0; y < 424; ++y)
      for(int x = 0; x < 512; ++x, m_filtered_ptr += 9, ++m_max_edge_test_ptr)
      {
        bool max_edge_test_val = true;
        impl_->filterPixelStage1(x, y, m, m_filtered_ptr, max_edge_test_val);
        *m_max_edge_test_ptr = max_edge_test_val ? 1 : 0;
      }

    m_ptr = m_filtered.ptr<float>();
  }
  else
  {
    m_ptr = m.ptr<float>();
  }

  cv::Mat out_ir(424, 512, CV_32FC1, impl_->ir_frame->data), out_depth(424, 512, CV_32FC1, impl_->depth_frame->data);

  if(impl_->enable_edge_filter)
  {
    cv::Mat depth_ir_sum(424, 512, CV_32FC3);
    cv::Vec3f *depth_ir_sum_ptr = depth_ir_sum.ptr<cv::Vec3f>();
    unsigned char *m_max_edge_test_ptr = m_max_edge_test.ptr<unsigned char>();

    for(int y = 0; y < 424; ++y)
      for(int x = 0; x < 512; ++x, m_ptr += 9, ++m_max_edge_test_ptr, ++depth_ir_sum_ptr)
      {
        float raw_depth, ir_sum;

        impl_->processPixelStage2(x, y, m_ptr + 0, m_ptr + 3, m_ptr + 6, out_ir.ptr<float>(423 - y, x), &raw_depth, &ir_sum);

        depth_ir_sum_ptr->val[0] = raw_depth;
        depth_ir_sum_ptr->val[1] = *m_max_edge_test_ptr == 1 ? raw_depth : 0;
        depth_ir_sum_ptr->val[2] = ir_sum;
      }

    m_max_edge_test_ptr = m_max_edge_test.ptr<unsigned char>();

    for(int y = 0; y < 424; ++y)
      for(int x = 0; x < 512; ++x, ++m_max_edge_test_ptr)
      {
        impl_->filterPixelStage2(x, y, depth_ir_sum, *m_max_edge_test_ptr == 1, out_depth.ptr<float>(423 - y, x));
      }
  }
  else
  {
    for(int y = 0; y < 424; ++y)
      for(int x = 0; x < 512; ++x, m_ptr += 9)
      {
        impl_->processPixelStage2(x, y, m_ptr + 0, m_ptr + 3, m_ptr + 6, out_ir.ptr<float>(423 - y, x), out_depth.ptr<float>(423 - y, x), 0);
      }
  }

  if(listener_->onNewFrame(Frame::Ir, impl_->ir_frame))
  {
    impl_->newIrFrame();
  }

  if(listener_->onNewFrame(Frame::Depth, impl_->depth_frame))
  {
    impl_->newDepthFrame();
  }

  impl_->stopTiming();
}

} /* namespace libfreenect2 */

