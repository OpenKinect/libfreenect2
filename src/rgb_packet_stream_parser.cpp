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

/** @file rgb_packet_stream_parser.cpp Parser implementation for retrieving JPEG data from a stream. */

#include <libfreenect2/config.h>
#include <libfreenect2/rgb_packet_stream_parser.h>
#include <libfreenect2/logging.h>
#include <memory.h>

namespace libfreenect2
{


LIBFREENECT2_PACK(struct RawRgbPacket
{
  uint32_t sequence;
  uint32_t magic_header; // is 'BBBB' equal 0x42424242

  unsigned char jpeg_buffer[0];
});

// starting from JPEG EOI: 0xff 0xd9
// char pad_0xa5[]; //0-3 bytes alignment of 0xa5
// char filler[filler_length] = "ZZZZ...";
LIBFREENECT2_PACK(struct RgbPacketFooter {
  uint32_t magic_header; // is '9999' equal 0x39393939
  uint32_t sequence;
  uint32_t filler_length;
  uint32_t unknown1; // seems 0 always
  uint32_t unknown2; // seems 0 always
  uint32_t timestamp;
  float exposure; // ? ranges from 0.5 to about 60.0 with powerfull light at camera or totally covered
  float gain; // ? ranges from 1.0 when camera is clear to 1.5 when camera is covered.
  uint32_t magic_footer; // is 'BBBB' equal 0x42424242
  uint32_t packet_size;
  float unknown3; // seems 1.0f always
  uint32_t unknown4[3]; // seems to be 0 all the time.
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
      LOG_ERROR << "buffer overflow!";
      fb.length = 0;
      return;
    }

    // not enough data to do anything
    if (fb.length <= sizeof(RawRgbPacket) + sizeof(RgbPacketFooter))
      return;

    RgbPacketFooter* footer = reinterpret_cast<RgbPacketFooter *>(&fb.data[fb.length - sizeof(RgbPacketFooter)]);

    if (footer->magic_header == 0x39393939 && footer->magic_footer == 0x42424242)
    {
      RawRgbPacket *raw_packet = reinterpret_cast<RawRgbPacket *>(fb.data);

      if (fb.length != footer->packet_size || raw_packet->sequence != footer->sequence)
      {
        LOG_ERROR << "packetsize or sequence doesn't match!";
        fb.length = 0;
        return;
      }

      if (fb.length - sizeof(RawRgbPacket) - sizeof(RgbPacketFooter) < footer->filler_length)
      {
        LOG_ERROR << "not enough space for packet filler!";
        fb.length = 0;
        return;
      }

      size_t jpeg_length = 0;
      //check for JPEG EOI 0xff 0xd9 within 0 to 3 alignment bytes
      size_t length_no_filler = fb.length - sizeof(RawRgbPacket) - sizeof(RgbPacketFooter) - footer->filler_length;
      for (size_t i = 0; i < 4; i++)
      {
        if (length_no_filler < i + 2)
          break;
        size_t eoi = length_no_filler - i;

        if (raw_packet->jpeg_buffer[eoi - 2] == 0xff && raw_packet->jpeg_buffer[eoi - 1] == 0xd9)
          jpeg_length = eoi;
      }

      if (jpeg_length == 0)
      {
        LOG_ERROR << "no JPEG detected!";
        fb.length = 0;
        return;
      }

      // can the processor handle the next image?
      if(processor_->ready())
      {
        buffer_.swap();
        Buffer &bb = buffer_.back();

        RawRgbPacket *raw_packet = reinterpret_cast<RawRgbPacket *>(bb.data);
        RgbPacket rgb_packet;
        rgb_packet.sequence = raw_packet->sequence;
        rgb_packet.timestamp = footer->timestamp;
        rgb_packet.jpeg_buffer = raw_packet->jpeg_buffer;
        rgb_packet.jpeg_buffer_length = jpeg_length;

        // call the processor
        processor_->process(rgb_packet);
      }
      else
      {
        LOG_DEBUG << "skipping rgb packet!";
      }

      // reset front buffer
      buffer_.front().length = 0;
    }
  }
}

} /* namespace libfreenect2 */
