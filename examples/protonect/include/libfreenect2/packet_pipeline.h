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

#ifndef PACKET_PIPELINE_H_
#define PACKET_PIPELINE_H_

#include <libfreenect2/config.h>
#include <libfreenect2/data_callback.h>
#include <libfreenect2/rgb_packet_stream_parser.h>
#include <libfreenect2/depth_packet_stream_parser.h>
#include <libfreenect2/depth_packet_processor.h>
#include <libfreenect2/rgb_packet_processor.h>

namespace libfreenect2
{

class LIBFREENECT2_API PacketPipeline
{
public:
  typedef DataCallback PacketParser;
  virtual ~PacketPipeline();

  virtual PacketParser *getRgbPacketParser() const = 0;
  virtual PacketParser *getIrPacketParser() const = 0;

  virtual RgbPacketProcessor *getRgbPacketProcessor() const = 0;
  virtual DepthPacketProcessor *getDepthPacketProcessor() const = 0;
};

class LIBFREENECT2_API BasePacketPipeline : public PacketPipeline
{
protected:
  RgbPacketStreamParser *rgb_parser_;
  DepthPacketStreamParser *depth_parser_;

  RgbPacketProcessor *rgb_processor_;
  BaseRgbPacketProcessor *async_rgb_processor_;
  DepthPacketProcessor *depth_processor_;
  BaseDepthPacketProcessor *async_depth_processor_;

  virtual void initialize();
  virtual DepthPacketProcessor *createDepthPacketProcessor() = 0;
public:
  virtual ~BasePacketPipeline();

  virtual PacketParser *getRgbPacketParser() const;
  virtual PacketParser *getIrPacketParser() const;

  virtual RgbPacketProcessor *getRgbPacketProcessor() const;
  virtual DepthPacketProcessor *getDepthPacketProcessor() const;
};

class LIBFREENECT2_API CpuPacketPipeline : public BasePacketPipeline
{
protected:
  virtual DepthPacketProcessor *createDepthPacketProcessor();
public:
  CpuPacketPipeline();
  virtual ~CpuPacketPipeline();
};

#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
class LIBFREENECT2_API OpenGLPacketPipeline : public BasePacketPipeline
{
protected:
  void *parent_opengl_context_;
  bool debug_;
  virtual DepthPacketProcessor *createDepthPacketProcessor();
public:
  OpenGLPacketPipeline(void *parent_opengl_context = 0, bool debug = false);
  virtual ~OpenGLPacketPipeline();
};
#endif // LIBFREENECT2_WITH_OPENGL_SUPPORT

#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
class LIBFREENECT2_API OpenCLPacketPipeline : public BasePacketPipeline
{
protected:
  const int deviceId;
  virtual DepthPacketProcessor *createDepthPacketProcessor();
public:
  OpenCLPacketPipeline(const int deviceId = -1);
  virtual ~OpenCLPacketPipeline();
};
#endif // LIBFREENECT2_WITH_OPENCL_SUPPORT


} /* namespace libfreenect2 */
#endif /* PACKET_PIPELINE_H_ */
