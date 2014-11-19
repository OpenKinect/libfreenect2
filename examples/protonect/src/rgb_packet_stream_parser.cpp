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

#include <libfreenect2/config.h>
#include <libfreenect2/rgb_packet_stream_parser.h>
#include <memory.h>
#include <iostream>

namespace libfreenect2
{


LIBFREENECT2_PACK(struct RawRgbPacket
{
  uint32_t sequence;
  uint32_t unknown0;

  unsigned char jpeg_buffer[0];
});

RgbPacketStreamParser::RgbPacketStreamParser() :
    processor_(noopProcessor<RgbPacket>())
{
  buffer_.allocate(1920*1080*3+sizeof(RgbPacket));
}

RgbPacketStreamParser::~RgbPacketStreamParser()
{
}

void RgbPacketStreamParser::setPacketProcessor(BaseRgbPacketProcessor *processor)
{
  processor_ = (processor != 0) ? processor : noopProcessor<RgbPacket>();
}

void RgbPacketStreamParser::onDataReceived(unsigned char* buffer, size_t length)
{
  Buffer &fb = buffer_.front();

  // package containing data
  if(length > 0)
  {
    if(fb.length + length <= fb.capacity)
    {
      memcpy(fb.data + fb.length, buffer, length);
      fb.length += length;
    }
    else
    {
      std::cerr << "[RgbPacketStreamParser::handleNewData] buffer overflow!" << std::endl;
    }

    // not full transfer buffer and we already have some data -> signals end of rgb image packet
    // TODO: better method, is unknown0 a magic? detect JPEG magic?
    if(length < 0x4000 && fb.length > sizeof(RgbPacket))
    {
      // can the processor handle the next image?
      if(processor_->ready())
      {
        buffer_.swap();
        Buffer &bb = buffer_.back();

        RawRgbPacket *raw_packet = reinterpret_cast<RawRgbPacket *>(bb.data);
        RgbPacket rgb_packet;
        rgb_packet.sequence = raw_packet->sequence;
        rgb_packet.jpeg_buffer = raw_packet->jpeg_buffer;
        rgb_packet.jpeg_buffer_length = bb.length - sizeof(RawRgbPacket);

        // call the processor
        processor_->process(rgb_packet);
      }
      else
      {
        std::cerr << "[RgbPacketStreamParser::handleNewData] skipping rgb packet!" << std::endl;
      }

      // reset front buffer
      buffer_.front().length = 0;
    }
  }
}

} /* namespace libfreenect2 */
