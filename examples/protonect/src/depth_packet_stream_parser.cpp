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

#include <libfreenect2/depth_packet_stream_parser.h>
#include <iostream>
#include <memory.h>

namespace libfreenect2
{

DepthPacketStreamParser::DepthPacketStreamParser() :
    processor_(noopProcessor<DepthPacket>()),
    current_sequence_(0),
    current_subsequence_(0)
{
  size_t single_image = 512*424*11/8;

  work_buffer_.data = new unsigned char[single_image];
  work_buffer_.capacity = single_image;
  work_buffer_.length = 0;
}

DepthPacketStreamParser::~DepthPacketStreamParser()
{
}

void DepthPacketStreamParser::setPacketProcessor(libfreenect2::BaseDepthPacketProcessor *processor)
{
  processor_ = (processor != 0) ? processor : noopProcessor<DepthPacket>();

  DoubleBuffer &buffer_ = *processor_->getPacketBuffer();
  size_t single_image = 512*424*11/8;

  buffer_.allocate((single_image) * 10);
  buffer_.front().length = buffer_.front().capacity;
  buffer_.back().length = buffer_.back().capacity;
}

void DepthPacketStreamParser::onDataReceived(unsigned char* buffer, size_t in_length)
{
  DoubleBuffer &buffer_ = *processor_->getPacketBuffer();
  Buffer &wb = work_buffer_;

  if(in_length == 0)
  {
    //synchronize to subpacket boundary
    wb.length = 0;
  }
  else
  {
    DepthSubPacketFooter *footer = 0;
    bool footer_found = false;

    if(wb.length + in_length == wb.capacity + sizeof(DepthSubPacketFooter))
    {
      in_length -= sizeof(DepthSubPacketFooter);
      footer = reinterpret_cast<DepthSubPacketFooter *>(&buffer[in_length]);
      footer_found = true;
    }

    if(wb.length + in_length > wb.capacity)
    {
      std::cerr << "[DepthPacketStreamParser::onDataReceived] subpacket too large" << std::endl;
      wb.length = 0;
      return;
    }

    memcpy(wb.data + wb.length, buffer, in_length);
    wb.length += in_length;

    if(footer_found)
    {
      if(footer->length != wb.length)
      {
        std::cerr << "[DepthPacketStreamParser::onDataReceived] image data too short!" << std::endl;
      }
      else
      {
        if(current_sequence_ != footer->sequence)
        {
          if(current_subsequence_ == 0x3ff)
          {
            if(processor_->ready())
            {
              buffer_.swap();

              DepthPacket packet;
              packet.sequence = current_sequence_;
              packet.timestamp = footer->timestamp;
              packet.buffer = buffer_.back().data;
              packet.buffer_length = buffer_.back().length;

              processor_->process(packet);
            }
            else
            {
              std::cerr << "[DepthPacketStreamParser::onDataReceived] skipping depth packet" << std::endl;
            }
          }
          else
          {
            std::cerr << "[DepthPacketStreamParser::onDataReceived] not all subsequences received " << current_subsequence_ << std::endl;
          }

          current_sequence_ = footer->sequence;
          current_subsequence_ = 0;
        }

        Buffer &fb = buffer_.front();

        // set the bit corresponding to the subsequence number to 1
        current_subsequence_ |= 1 << footer->subsequence;

        if(footer->subsequence * footer->length > fb.length)
        {
          std::cerr << "[DepthPacketStreamParser::onDataReceived] front buffer too short! subsequence number is " << footer->subsequence << std::endl;
        }
        else
        {
          memcpy(fb.data + (footer->subsequence * footer->length), wb.data + (wb.length - footer->length), footer->length);
        }
      }

      // reset working buffer
      wb.length = 0;
    }
  }
}

} /* namespace libfreenect2 */
