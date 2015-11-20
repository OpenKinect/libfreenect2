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

/** @file depth_packet_processor.h Depth processor definitions. */

#ifndef DEPTH_PACKET_PROCESSOR_H_
#define DEPTH_PACKET_PROCESSOR_H_

#include <stddef.h>
#include <stdint.h>

#include <libfreenect2/config.h>
#include <libfreenect2/frame_listener.hpp>
#include <libfreenect2/packet_processor.h>

namespace libfreenect2
{

/** Data packet with depth information. */
struct LIBFREENECT2_API DepthPacket
{
  uint32_t sequence;
  uint32_t timestamp;
  unsigned char *buffer; ///< Depth data.
  size_t buffer_length;  ///< Size of depth data.
};

/** Class for processing depth information. */
typedef PacketProcessor<DepthPacket> BaseDepthPacketProcessor;

class LIBFREENECT2_API DepthPacketProcessor : public BaseDepthPacketProcessor
{
public:
  /** Configuration of depth processing. */
  struct LIBFREENECT2_API Config
  {
    float MinDepth;
    float MaxDepth;

    bool EnableBilateralFilter; ///< Whether to run the bilateral filter.
    bool EnableEdgeAwareFilter; ///< Whether to run the edge aware filter.

    Config();
  };

  /** Parameters of depth processing. */
  struct LIBFREENECT2_API Parameters
  {
    float ab_multiplier;
    float ab_multiplier_per_frq[3];
    float ab_output_multiplier;

    float phase_in_rad[3];

    float joint_bilateral_ab_threshold;
    float joint_bilateral_max_edge;
    float joint_bilateral_exp;
    float gaussian_kernel[9];

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

    float min_depth;
    float max_depth;

    Parameters();
  };

  DepthPacketProcessor();
  virtual ~DepthPacketProcessor();

  virtual void setFrameListener(libfreenect2::FrameListener *listener);
  virtual void setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config);

  virtual void loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length) = 0;

  static const size_t TABLE_SIZE = 512*424;
  static const size_t LUT_SIZE = 2048;
  virtual void loadXZTables(const float *xtable, const float *ztable) = 0;
  virtual void loadLookupTable(const short *lut) = 0;

protected:
  libfreenect2::DepthPacketProcessor::Config config_;
  libfreenect2::FrameListener *listener_;
};

#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
class OpenGLDepthPacketProcessorImpl;

/** Depth packet processor using OpenGL. */
class LIBFREENECT2_API OpenGLDepthPacketProcessor : public DepthPacketProcessor
{
public:
  OpenGLDepthPacketProcessor(void *parent_opengl_context_ptr, bool debug);
  virtual ~OpenGLDepthPacketProcessor();
  virtual void setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config);

  virtual void loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length);

  virtual void loadXZTables(const float *xtable, const float *ztable);
  virtual void loadLookupTable(const short *lut);

  virtual void process(const DepthPacket &packet);
private:
  OpenGLDepthPacketProcessorImpl *impl_;
};
#endif // LIBFREENECT2_WITH_OPENGL_SUPPORT

// TODO: push this to some internal namespace
class CpuDepthPacketProcessorImpl;

/** Depth packet processor using the CPU. */
class LIBFREENECT2_API CpuDepthPacketProcessor : public DepthPacketProcessor
{
public:
  CpuDepthPacketProcessor();
  virtual ~CpuDepthPacketProcessor();
  virtual void setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config);

  virtual void loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length);

  virtual void loadXZTables(const float *xtable, const float *ztable);
  virtual void loadLookupTable(const short *lut);

  virtual void process(const DepthPacket &packet);
private:
  CpuDepthPacketProcessorImpl *impl_;
};

#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
class OpenCLDepthPacketProcessorImpl;

/** Depth packet processor using OpenCL. */
class LIBFREENECT2_API OpenCLDepthPacketProcessor : public DepthPacketProcessor
{
public:
  OpenCLDepthPacketProcessor(const int deviceId = -1);
  virtual ~OpenCLDepthPacketProcessor();
  virtual void setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config);

  virtual void loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length);

  virtual void loadXZTables(const float *xtable, const float *ztable);
  virtual void loadLookupTable(const short *lut);

  virtual void process(const DepthPacket &packet);
private:
  OpenCLDepthPacketProcessorImpl *impl_;
};
#endif // LIBFREENECT2_WITH_OPENCL_SUPPORT
} /* namespace libfreenect2 */
#endif /* DEPTH_PACKET_PROCESSOR_H_ */
