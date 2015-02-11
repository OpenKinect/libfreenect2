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
  uint32_t magic_header; // is 'BBBB' equal 0x42424242

  unsigned char jpeg_buffer[0];
});

LIBFREENECT2_PACK(struct RgbPacketFooter {
    uint32_t magic_header; // is '9999' equal 0x39393939
    uint32_t sequence;
    int32_t filler_length; // Filler length of filler before footer
    int32_t unknown2; // seems 0 always
    int32_t unknown3; // seems 0 always
    uint32_t timestamp;
    float exposure; // ranges from 0.5 to about 60.0 with powerfull light at camera or totally covered
    float gain; // ranges from 1.0 when camera is clear to 1.5 when camera is covered.
    uint32_t magic_footer; // is 'BBBB' equal 0x42424242
    uint32_t packet_size;
    float unknown4; // consistenly 1.0
    uint8_t unknown5[12]; // 0 all the time.
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
  // if no data return
  if (length == 0)
    return;

  Buffer &fb = buffer_.front();

  // clear buffer if overflow of data
  if (fb.length + length > fb.capacity)
  {
    std::cerr << "[RgbPacketStreamParser::handleNewData] Buffer overflow - resetting." << std::endl;
    fb.length = 0;
    return;
  }

  memcpy(&fb.data[fb.length], buffer, length);
  fb.length += length;

  if (fb.length < sizeof(RgbPacketFooter))
    return;

  RgbPacketFooter* footer = reinterpret_cast<RgbPacketFooter *>(&fb.data[fb.length - sizeof(RgbPacketFooter)]);

  // if magic markers match
  if (footer->magic_header == 0x39393939 && footer->magic_footer == 0x42424242)
  {
    RawRgbPacket *raw_packet = reinterpret_cast<RawRgbPacket *>(fb.data);

    if (fb.length == footer->packet_size && raw_packet->sequence == footer->sequence)
    {
      if (processor_->ready())
      {
        buffer_.swap();
        Buffer &bb = buffer_.back();

        RgbPacket rgb_packet;
        rgb_packet.sequence = raw_packet->sequence;
        rgb_packet.timestamp = footer->timestamp;
        rgb_packet.jpeg_buffer = raw_packet->jpeg_buffer;
        rgb_packet.jpeg_buffer_length = bb.length - sizeof(RawRgbPacket) - sizeof(RgbPacketFooter) - footer->filler_length;

        // call the processor
        processor_->process(rgb_packet);
      }
      else
      {
        std::cerr << "[RgbPacketStreamParser::handleNewData] skipping rgb packet!" << std::endl;
      }
      // clear buffer and return
      buffer_.front().length = 0;
      return;
    }
    else
    {
      std::cerr << "[RgbPacketStreamParser::handleNewData] packetsize or sequence doesn't match!" << std::endl;
      fb.length = 0;
      return;
    }
  }
}

} /* namespace libfreenect2 */
