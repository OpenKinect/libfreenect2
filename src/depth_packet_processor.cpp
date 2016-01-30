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

/** @file depth_packet_processor.cpp Generic part of the depth processors (configuration and parameters). */

#include <libfreenect2/depth_packet_processor.h>
#include <libfreenect2/async_packet_processor.h>

#include <cstring>

namespace libfreenect2
{

DepthPacketProcessor::Parameters::Parameters()
{
  ab_multiplier = 0.6666667f;
  ab_multiplier_per_frq[0] = 1.322581f;
  ab_multiplier_per_frq[1] = 1.0f;
  ab_multiplier_per_frq[2] = 1.612903f;
  ab_output_multiplier = 16.0f;

  phase_in_rad[0] = 0.0f;
  phase_in_rad[1] = 2.094395f;
  phase_in_rad[2] = 4.18879f;

  joint_bilateral_ab_threshold = 3.0f;
  joint_bilateral_max_edge = 2.5f;
  joint_bilateral_exp = 5.0f;

  gaussian_kernel[0] = 0.1069973f;
  gaussian_kernel[1] = 0.1131098f;
  gaussian_kernel[2] = 0.1069973f;
  gaussian_kernel[3] = 0.1131098f;
  gaussian_kernel[4] = 0.1195716f;
  gaussian_kernel[5] = 0.1131098f;
  gaussian_kernel[6] = 0.1069973f;
  gaussian_kernel[7] = 0.1131098f;
  gaussian_kernel[8] = 0.1069973f;

  phase_offset = 0.0f;
  unambigious_dist = 2083.333f;
  individual_ab_threshold  = 3.0f;
  ab_threshold = 10.0f;
  ab_confidence_slope = -0.5330578f;
  ab_confidence_offset = 0.7694894f;
  min_dealias_confidence = 0.3490659f;
  max_dealias_confidence = 0.6108653f;

  edge_ab_avg_min_value = 50.0f;
  edge_ab_std_dev_threshold = 0.05f;
  edge_close_delta_threshold = 50.0f;
  edge_far_delta_threshold = 30.0f;
  edge_max_delta_threshold = 100.0f;
  edge_avg_delta_threshold = 0.0f;
  max_edge_count  = 5.0f;

  min_depth = 500.0f;
  max_depth = 4500.0f;
}

DepthPacketProcessor::DepthPacketProcessor() :
    listener_(0)
{
}

DepthPacketProcessor::~DepthPacketProcessor()
{
}

void DepthPacketProcessor::setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config)
{
  config_ = config;
}

void DepthPacketProcessor::setFrameListener(libfreenect2::FrameListener *listener)
{
  listener_ = listener;
}

DumpDepthPacketProcessor::DumpDepthPacketProcessor()
  : p0table_(NULL), xtable_(NULL), ztable_(NULL), lut_(NULL) {
}

DumpDepthPacketProcessor::~DumpDepthPacketProcessor(){
  delete[] p0table_;
  delete[] xtable_;
  delete[] ztable_;
  delete[] lut_;
}

void DumpDepthPacketProcessor::process(const DepthPacket &packet) {
  Frame* depth_frame = new Frame(1, 1, packet.buffer_length);
  
  depth_frame->timestamp = packet.timestamp;
  depth_frame->sequence = packet.sequence;
  depth_frame->format = Frame::Raw;
  std::memcpy(depth_frame->data, packet.buffer, packet.buffer_length);

  Frame* ir_frame = new Frame(1, 1, packet.buffer_length, depth_frame->data);
  ir_frame->timestamp = packet.timestamp;
  ir_frame->sequence = packet.sequence;
  ir_frame->data = packet.buffer;
  ir_frame->format = Frame::Raw;

  if (!listener_->onNewFrame(Frame::Ir, ir_frame)) {
    delete ir_frame;
  }
  ir_frame = NULL;
  if (!listener_->onNewFrame(Frame::Depth, depth_frame)) {
    delete depth_frame;
  }
  depth_frame = NULL;
}
  
const unsigned char* DumpDepthPacketProcessor::getP0Tables() { return p0table_; }
  
const float* DumpDepthPacketProcessor::getXTable() { return xtable_; }
const float* DumpDepthPacketProcessor::getZTable() { return ztable_; }

const short* DumpDepthPacketProcessor::getLookupTable() { return lut_; }

void DumpDepthPacketProcessor::loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length) {
  p0table_ = new unsigned char[buffer_length];
  std::memcpy(p0table_, buffer, buffer_length);
}
  
void DumpDepthPacketProcessor::loadXZTables(const float *xtable, const float *ztable) {
  xtable_ = new float[TABLE_SIZE];
  std::memcpy(xtable_, xtable, TABLE_SIZE * sizeof(float));

  ztable_ = new float[TABLE_SIZE];
  std::memcpy(ztable_, ztable, TABLE_SIZE * sizeof(float));
}
  
void DumpDepthPacketProcessor::loadLookupTable(const short *lut) {
  lut_ = new short[LUT_SIZE];
  std::memcpy(lut_, lut, LUT_SIZE * sizeof(short));
}
} /* namespace libfreenect2 */
