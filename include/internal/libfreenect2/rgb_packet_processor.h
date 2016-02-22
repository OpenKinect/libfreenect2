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

/** @file rgb_packet_processor.h JPEG decoder processors. */

#ifndef RGB_PACKET_PROCESSOR_H_
#define RGB_PACKET_PROCESSOR_H_

#include <stddef.h>
#include <stdint.h>

#include <libfreenect2/config.h>
#include <libfreenect2/frame_listener.hpp>
#include <libfreenect2/packet_processor.h>

namespace libfreenect2
{

/** Packet with JPEG data. */
struct RgbPacket
{
  uint32_t sequence;

  uint32_t timestamp;
  unsigned char *jpeg_buffer; ///< JPEG data.
  size_t jpeg_buffer_length;  ///< Length of the JPEG data.
  float exposure;
  float gain;
  float gamma;

  Buffer *memory;
};

typedef PacketProcessor<RgbPacket> BaseRgbPacketProcessor;

/** JPEG processor. */
class RgbPacketProcessor : public BaseRgbPacketProcessor
{
public:
  RgbPacketProcessor();
  virtual ~RgbPacketProcessor();

  virtual void setFrameListener(libfreenect2::FrameListener *listener);
protected:
  libfreenect2::FrameListener *listener_;
};

/** Class for dumping the JPEG information, eg to file. */
class DumpRgbPacketProcessor : public RgbPacketProcessor
{
public:
  DumpRgbPacketProcessor();
  virtual ~DumpRgbPacketProcessor();
  virtual void process(const libfreenect2::RgbPacket &packet);
};

#ifdef LIBFREENECT2_WITH_TURBOJPEG_SUPPORT
class TurboJpegRgbPacketProcessorImpl;

/** Processor to decode JPEG to image, using TurboJpeg. */
class TurboJpegRgbPacketProcessor : public RgbPacketProcessor
{
public:
  TurboJpegRgbPacketProcessor();
  virtual ~TurboJpegRgbPacketProcessor();
  virtual void process(const libfreenect2::RgbPacket &packet);
  virtual const char *name() { return "TurboJPEG"; }
private:
  TurboJpegRgbPacketProcessorImpl *impl_; ///< Decoder implementation.
};
#endif

#ifdef LIBFREENECT2_WITH_VT_SUPPORT
class VTRgbPacketProcessorImpl;

class VTRgbPacketProcessor : public RgbPacketProcessor
{
public:
  VTRgbPacketProcessor();
  virtual ~VTRgbPacketProcessor();
  virtual void process(const libfreenect2::RgbPacket &packet);
  virtual const char *name() { return "VideoToolbox"; }
private:
  VTRgbPacketProcessorImpl *impl_;
};
#endif

#ifdef LIBFREENECT2_WITH_VAAPI_SUPPORT
class VaapiRgbPacketProcessorImpl;

class VaapiRgbPacketProcessor : public RgbPacketProcessor
{
public:
  VaapiRgbPacketProcessor();
  virtual ~VaapiRgbPacketProcessor();
  virtual bool good();
  virtual const char *name() { return "VAAPI"; }
  virtual void process(const libfreenect2::RgbPacket &packet);
protected:
  virtual Allocator *getAllocator();
private:
  VaapiRgbPacketProcessorImpl *impl_;
};
#endif //LIBFREENECT2_WITH_VAAPI_SUPPORT

#ifdef LIBFREENECT2_WITH_TEGRAJPEG_SUPPORT
class TegraJpegRgbPacketProcessorImpl;

class TegraJpegRgbPacketProcessor : public RgbPacketProcessor
{
public:
  TegraJpegRgbPacketProcessor();
  virtual ~TegraJpegRgbPacketProcessor();
  virtual bool good();
  virtual const char *name() { return "TegraJPEG"; }
  virtual void process(const libfreenect2::RgbPacket &packet);
private:
  TegraJpegRgbPacketProcessorImpl *impl_;
};
#endif //LIBFREENECT2_WITH_TEGRAJPEG_SUPPORT

} /* namespace libfreenect2 */
#endif /* RGB_PACKET_PROCESSOR_H_ */
