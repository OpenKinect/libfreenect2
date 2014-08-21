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

#ifndef DEPTH_PACKET_PROCESSOR_H_
#define DEPTH_PACKET_PROCESSOR_H_

#include <stddef.h>
#include <stdint.h>

#include <libfreenect2/frame_listener.hpp>

namespace libfreenect2
{

struct DepthPacket
{
  uint32_t sequence;
  unsigned char *buffer;
  size_t buffer_length;
};

class DepthPacketProcessor
{
public:
  struct Config
  {
    float MinDepth;
    float MaxDepth;
    
    bool EnableBilateralFilter;
    bool EnableEdgeAwareFilter;
    
    Config();
  };

  struct Parameters
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
  virtual void process(const DepthPacket &packet) = 0;

  virtual void loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length) = 0;

protected:
  libfreenect2::DepthPacketProcessor::Config config_;
  libfreenect2::FrameListener *listener_;
};

class OpenGLDepthPacketProcessorImpl;

class OpenGLDepthPacketProcessor : public DepthPacketProcessor
{
public:
  OpenGLDepthPacketProcessor(void *parent_opengl_context_ptr);
  virtual ~OpenGLDepthPacketProcessor();
  virtual void setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config);

  virtual void loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length);

  void loadP0TablesFromFiles(const char* p0_filename, const char* p1_filename, const char* p2_filename);

  /**
   * GUESS: the x and z table follow some polynomial, until we know the exact polynom formula and its coefficients
   * just load them from a memory dump - although they probably vary per camera
   */
  void loadXTableFromFile(const char* filename);

  void loadZTableFromFile(const char* filename);

  void load11To16LutFromFile(const char* filename);

  virtual void process(const DepthPacket &packet);
private:
  OpenGLDepthPacketProcessorImpl *impl_;
};

// TODO: push this to some internal namespace
// use pimpl to hide opencv dependency
class CpuDepthPacketProcessorImpl;

class CpuDepthPacketProcessor : public DepthPacketProcessor
{
public:
  CpuDepthPacketProcessor();
  virtual ~CpuDepthPacketProcessor();
  virtual void setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config);

  virtual void loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length);

  void loadP0TablesFromFiles(const char* p0_filename, const char* p1_filename, const char* p2_filename);

  /**
   * GUESS: the x and z table follow some polynomial, until we know the exact polynom formula and its coefficients
   * just load them from a memory dump - although they probably vary per camera
   */
  void loadXTableFromFile(const char* filename);

  void loadZTableFromFile(const char* filename);

  void load11To16LutFromFile(const char* filename);

  virtual void process(const DepthPacket &packet);
private:
  CpuDepthPacketProcessorImpl *impl_;
};

class OpenCLDepthPacketProcessorImpl;

class OpenCLDepthPacketProcessor : public DepthPacketProcessor
{
public:
  OpenCLDepthPacketProcessor();
  virtual ~OpenCLDepthPacketProcessor();
  virtual void setConfiguration(const libfreenect2::DepthPacketProcessor::Config &config);

  virtual void loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length);

  /**
   * GUESS: the x and z table follow some polynomial, until we know the exact polynom formula and its coefficients
   * just load them from a memory dump - although they probably vary per camera
   */
  void loadXTableFromFile(const char* filename);

  void loadZTableFromFile(const char* filename);

  void load11To16LutFromFile(const char* filename);

  virtual void process(const DepthPacket &packet);
private:
  OpenCLDepthPacketProcessorImpl *impl_;
};

} /* namespace libfreenect2 */
#endif /* DEPTH_PACKET_PROCESSOR_H_ */
