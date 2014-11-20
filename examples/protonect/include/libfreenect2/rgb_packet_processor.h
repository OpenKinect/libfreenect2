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

#ifndef RGB_PACKET_PROCESSOR_H_
#define RGB_PACKET_PROCESSOR_H_

#include <stddef.h>
#include <stdint.h>

#include <libfreenect2/config.h>
#include <libfreenect2/frame_listener.hpp>
#include <libfreenect2/packet_processor.h>

namespace libfreenect2
{

struct LIBFREENECT2_API RgbPacket
{
  uint32_t sequence;

  unsigned char *jpeg_buffer;
  size_t jpeg_buffer_length;
};

// explicit instantiation and export to make vsc++ happy
template class LIBFREENECT2_API PacketProcessor<RgbPacket>;
typedef PacketProcessor<RgbPacket> BaseRgbPacketProcessor;

class LIBFREENECT2_API RgbPacketProcessor : public BaseRgbPacketProcessor
{
public:
  RgbPacketProcessor();
  virtual ~RgbPacketProcessor();

  virtual void setFrameListener(libfreenect2::FrameListener *listener);
protected:
  libfreenect2::FrameListener *listener_;
};

class LIBFREENECT2_API DumpRgbPacketProcessor : public RgbPacketProcessor
{
public:
  DumpRgbPacketProcessor();
  virtual ~DumpRgbPacketProcessor();
protected:
  virtual void process(const libfreenect2::RgbPacket &packet);
};

class TurboJpegRgbPacketProcessorImpl;

class LIBFREENECT2_API TurboJpegRgbPacketProcessor : public RgbPacketProcessor
{
public:
  TurboJpegRgbPacketProcessor();
  virtual ~TurboJpegRgbPacketProcessor();
protected:
  virtual void process(const libfreenect2::RgbPacket &packet);
private:
  TurboJpegRgbPacketProcessorImpl *impl_;
};

} /* namespace libfreenect2 */
#endif /* RGB_PACKET_PROCESSOR_H_ */
