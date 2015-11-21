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

/** @file packet_pipeline.h Packet pipe line definitions. */

#ifndef PACKET_PIPELINE_H_
#define PACKET_PIPELINE_H_

#include <libfreenect2/config.h>

namespace libfreenect2
{

class DataCallback;
class RgbPacketProcessor;
class DepthPacketProcessor;
class PacketPipelineComponents;

class ConfigPacketProcessor;
#ifndef LIBFREENECT2_SETCONFIGURATION_COMPAT_INTERNAL
#define DepthPacketProcessor ConfigPacketProcessor
#endif

/** Front of the pipeline, RGB and Depth parsing and processing. */
class LIBFREENECT2_API PacketPipeline
{
public:
  typedef DataCallback PacketParser;

  PacketPipeline();
  virtual ~PacketPipeline();

  virtual PacketParser *getRgbPacketParser() const;
  virtual PacketParser *getIrPacketParser() const;

  virtual RgbPacketProcessor *getRgbPacketProcessor() const;
  virtual DepthPacketProcessor *getDepthPacketProcessor() const;
protected:
  PacketPipelineComponents *comp_;
};

/** Complete pipe line with depth processing by the CPU. */
class LIBFREENECT2_API CpuPacketPipeline : public PacketPipeline
{
public:
  CpuPacketPipeline();
  virtual ~CpuPacketPipeline();
};

#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
/** Complete pipe line with depth processing with OpenGL. */
class LIBFREENECT2_API OpenGLPacketPipeline : public PacketPipeline
{
protected:
  void *parent_opengl_context_;
  bool debug_;
public:
  OpenGLPacketPipeline(void *parent_opengl_context = 0, bool debug = false);
  virtual ~OpenGLPacketPipeline();
};
#endif // LIBFREENECT2_WITH_OPENGL_SUPPORT

#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
/** Complete pipe line with depth processing with OpenCL. */
class LIBFREENECT2_API OpenCLPacketPipeline : public PacketPipeline
{
protected:
  const int deviceId;
public:
  OpenCLPacketPipeline(const int deviceId = -1);
  virtual ~OpenCLPacketPipeline();
};
#endif // LIBFREENECT2_WITH_OPENCL_SUPPORT


} /* namespace libfreenect2 */
#endif /* PACKET_PIPELINE_H_ */
