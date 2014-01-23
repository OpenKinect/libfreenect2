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

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

namespace libfreenect2
{

template<typename T>
cv::Mat loadTableFromFile(const std::string& filename)
{
  std::ifstream file(filename.c_str());

  size_t h =  424, w = 512;
  size_t n = w * h * sizeof(T);

  cv::Mat r(h, w, cv::DataType<T>::type), r_final;

  file.read(reinterpret_cast<char*>(r.data), n);

  if(file.gcount() != n)
  {
    std::cerr << "file '" << filename << "' too short!" << std::endl;
    return cv::Mat();
  }

  file.close();

  return r;
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

  float phase_in_rad0, phase_in_rad1, phase_in_rad2;
  float ab_multiplier, ab_multiplier_per_frq0, ab_multiplier_per_frq1, ab_multiplier_per_frq2;
  float phase_offset;
  float unambigious_dist;
  float ab_output_multiplier;

  int16_t lut11to16[2048];

  cv::Mat out_ir;
  cv::Mat out_depth;

  double timing_acc;
  double timing_acc_n;

  double timing_current_start;

  CpuDepthPacketProcessorImpl()
  {
    phase_in_rad0 = 0.0f;
    phase_in_rad1 = 2.094395f;
    phase_in_rad2 = 4.18879f;
    ab_multiplier = 0.6666667f;
    ab_multiplier_per_frq0 = 1.322581f;
    ab_multiplier_per_frq1 = 1.0f;
    ab_multiplier_per_frq2 = 1.612903f;
    phase_offset = 0.0f;
    unambigious_dist = 2083.333f;
    ab_output_multiplier = 16.0f;

    out_ir = cv::Mat(424, 512, CV_32FC1);
    out_depth = cv::Mat(424, 512, CV_32FC1);

    timing_acc = 0.0;
    timing_acc_n = 0.0;
    timing_current_start = 0.0;
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

  void processMeasurementTriple(cv::Mat& P0Table, float abMultiplierPerFrq, int x, int y, const int32_t* m, float* m_out)
  {
    float p0 = -((float)P0Table.at<uint16_t>(y, x)) * 0.000031 * M_PI;

    float tmp0 = p0 + phase_in_rad0;
    float tmp1 = p0 + phase_in_rad1;
    float tmp2 = p0 + phase_in_rad2;

    float cos_tmp0 = std::cos(tmp0);
    float cos_tmp1 = std::cos(tmp1);
    float cos_tmp2 = std::cos(tmp2);

    float sin_negtmp0 = std::sin(-tmp0);
    float sin_negtmp1 = std::sin(-tmp1);
    float sin_negtmp2 = std::sin(-tmp2);

    float zmultiplier = z_table.at<float>(y, x);
    bool cond0 = 0 < zmultiplier;
    bool cond1 = (m[0] == 32767 || m[1] == 32767 || m[2] == 32767) && cond0;

    float tmp3 = cos_tmp0 * m[0] + cos_tmp1 * m[1] + cos_tmp2 * m[2];
    float tmp4 = sin_negtmp0 * m[0] + sin_negtmp1 * m[1] + sin_negtmp2 * m[2];

    // only if modeMask & 32 != 0;
    if(true)//(modeMask & 32) != 0)
    {
        tmp3 *= abMultiplierPerFrq;
        tmp4 *= abMultiplierPerFrq;
    }
    float tmp5 = std::sqrt(tmp3 * tmp3 + tmp4 * tmp4) * ab_multiplier;

    // invalid pixel because zmultiplier < 0 ??
    tmp3 = cond0 ? tmp3 : 0;
    tmp4 = cond0 ? tmp4 : 0;
    tmp5 = cond0 ? tmp5 : 0;

    // invalid pixel because saturated?
    tmp3 = !cond1 ? tmp3 : 0;
    tmp4 = !cond1 ? tmp4 : 0;
    tmp5 = !cond1 ? tmp5 : 65535.0; // some kind of norm calculated from tmp3 and tmp4

    m_out[0] = tmp3;
    m_out[1] = tmp4;
    m_out[2] = tmp5;
  }

  void transformMeasurements(float* m)
  {
    float tmp0 = std::atan2((m[0]), (m[1]));
    tmp0 = tmp0 < 0 ? tmp0 + M_PI * 2.0f : tmp0;
    tmp0 = (tmp0 != tmp0) ? 0 : tmp0;

    float tmp1 = std::sqrt(m[0] * m[0] + m[1] * m[1]) * ab_multiplier;

    m[0] = tmp0; // depth ?
    m[1] = tmp1; // ir ?
  }

  void processPixel(unsigned char* data, int x, int y, float* ir_out, float* depth_out)
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

    float m0[3], m1[3], m2[3];
    processMeasurementTriple(p0_table0, ab_multiplier_per_frq0, x, y, m0_raw, m0);
    processMeasurementTriple(p0_table1, ab_multiplier_per_frq1, x, y, m1_raw, m1);
    processMeasurementTriple(p0_table2, ab_multiplier_per_frq2, x, y, m2_raw, m2);

    float zmultiplier = z_table.at<float>(y, x);

    // 10th measurement
    float m9 = 1; // decodePixelMeasurement(data, 9, x, y);

    // WTF?
    bool cond0 = zmultiplier == 0 || (m9 >= 0 && m9 < 32767);
    m9 = std::max(-m9, m9);
    // if m9 is positive or pixel is invalid (zmultiplier) we set it to 0 otherwise to its absolute value O.o
    m9 = cond0 ? 0 : m9;

    // TODO: bilateral filtering

    transformMeasurements(m0);
    transformMeasurements(m1);
    transformMeasurements(m2);

    // if(DISABLE_DISAMBIGUATION)
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
    // else
        // TODO: disambiguation


    // this seems to be the phase to depth mapping :)
    //zmultiplier = (float)load<float>(zTable, x, y);
    float xmultiplier = x_table.at<float>(y, x);

    phase = 0 < phase ? phase + phase_offset : phase;

    float depth_linear = zmultiplier * phase;
    float max_depth = phase * unambigious_dist * 2;

    bool cond1 = /*(modeMask & 32) != 0*/ true && 0 < depth_linear && 0 < max_depth;

    xmultiplier = (xmultiplier * 90) / (max_depth * max_depth * 8192.0);

    float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1);

    depth_fit = depth_fit < 0 ? 0 : depth_fit;
    float depth = cond1 ? depth_fit : depth_linear; // r1.y -> later r2.z

    // TODO: edge aware bilateral filter

    if (x >= 0 && y >= 0 && x < 512 && y < 424)
    {
        // output depth
        // output (tmp2 + tmp3 + tmp4) * 0.3333333

        // output m1[2]
    }

    // depth
    *depth_out = depth;
    // ir
    //*ir_out = std::min((m1[2]) * ab_output_multiplier, 65535.0f);
    // ir avg
    *ir_out = std::min((m0[2] + m1[2] + m2[2]) * 0.3333333f * ab_output_multiplier, 65535.0f);
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

void CpuDepthPacketProcessor::loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length)
{
  int t0 = 34, t1 = 434214, t2 = 868394;

  size_t table_size = 424 * 512 * sizeof(uint16_t); // 434176

  if(buffer_length < t2 + table_size)
  {
    std::cerr << "[CpuDepthPacketProcessor::loadP0TablesFromCommandResponse] P0Table response too short!" << std::endl;
    return;
  }

  cv::Mat(424, 512, CV_16UC1, buffer + t0).copyTo(impl_->p0_table0);
  cv::Mat(424, 512, CV_16UC1, buffer + t1).copyTo(impl_->p0_table1);
  cv::Mat(424, 512, CV_16UC1, buffer + t2).copyTo(impl_->p0_table2);
}

void CpuDepthPacketProcessor::loadXTableFromFile(const char* filename)
{
  impl_->x_table = loadTableFromFile<float>(filename);
}

void CpuDepthPacketProcessor::loadZTableFromFile(const char* filename)
{
  impl_->z_table = loadTableFromFile<float>(filename);
}

void CpuDepthPacketProcessor::load11To16LutFromFile(const char* filename)
{
  size_t n = 2048 * sizeof(int16_t);

  std::ifstream file(filename);
  file.read(reinterpret_cast<char *>(impl_->lut11to16), n);

  if(file.gcount() != n)
  {
    std::cerr << "file '" << filename << "' too short!" << std::endl;
  }

  file.close();
}

void CpuDepthPacketProcessor::doProcess(DepthPacket* packet, size_t buffer_length)
{
  impl_->startTiming();

  for(int y = 0; y < 424; ++y)
    for(int x = 0; x < 512; ++x)
      impl_->processPixel(packet->buffer, x, y, impl_->out_ir.ptr<float>(423 - y, x), impl_->out_depth.ptr<float>(423 - y, x));

  cv::imshow("ir_out", impl_->out_ir / 20000.0f);
  // TODO: depth has wrong scale -.-
  cv::imshow("depth_out", impl_->out_depth / 30000.0f);
  cv::waitKey(1);

  impl_->stopTiming();
}

} /* namespace libfreenect2 */

