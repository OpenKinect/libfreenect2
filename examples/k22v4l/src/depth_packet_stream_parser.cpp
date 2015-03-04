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
#include <algorithm>

namespace libfreenect2
{

DepthPacketStreamParser::DepthPacketStreamParser() :
    processor_(noopProcessor<DepthPacket>()),
    current_sequence_(0),
    current_subsequence_(0)
{
  size_t single_image = 512*424*11/8;

  buffer_.allocate((single_image) * 10);
  buffer_.front().length = buffer_.front().capacity;
  buffer_.back().length = buffer_.back().capacity;

  work_buffer_.data = new unsigned char[single_image * 2];
  work_buffer_.capacity = single_image * 2;
  work_buffer_.length = 0;
}

DepthPacketStreamParser::~DepthPacketStreamParser()
{
}

void DepthPacketStreamParser::setPacketProcessor(libfreenect2::BaseDepthPacketProcessor *processor)
{
  processor_ = (processor != 0) ? processor : noopProcessor<DepthPacket>();
}

void DepthPacketStreamParser::onDataReceived(unsigned char* buffer, size_t in_length)
{
  // TODO: simplify this crap (so code, such unreadable, wow ;)
  Buffer &wb = work_buffer_;

  size_t in_offset = 0;

  while(in_offset < in_length)
  {
    unsigned char *ptr_in = buffer + in_offset, *ptr_out = wb.data + wb.length;
    DepthSubPacketFooter *footer = 0;
    bool footer_found = false;

    size_t max_length = std::min<size_t>(wb.capacity - wb.length, in_length - 8);

    for(; in_offset < max_length; ++in_offset)
    {
      footer = reinterpret_cast<DepthSubPacketFooter *>(ptr_in);

      if(footer->magic0 == 0x0 && footer->magic1 == 0x9)
      {
        footer_found = true;
        break;
      }

      *ptr_out = *ptr_in;
      ++ptr_in;
      ++ptr_out;
    }

    wb.length = ptr_out - wb.data;

    if(footer_found)
    {
      if((in_length - in_offset) < sizeof(DepthSubPacketFooter))
      {
        std::cerr << "[DepthPacketStreamParser::handleNewData] incomplete footer detected!" << std::endl;
      }
      else if(footer->length > wb.length)
      {
        std::cerr << "[DepthPacketStreamParser::handleNewData] image data too short!" << std::endl;
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
              packet.buffer = buffer_.back().data;
              packet.buffer_length = buffer_.back().length;

              processor_->process(packet);
            }
            else
            {
              //std::cerr << "[DepthPacketStreamParser::handleNewData] skipping depth packet!" << std::endl;
            }
          }
          else
          {
            std::cerr << "[DepthPacketStreamParser::handleNewData] not all subsequences received " << current_subsequence_ << std::endl;
          }

          current_sequence_ = footer->sequence;
          current_subsequence_ = 0;
        }

        Buffer &fb = buffer_.front();

        // set the bit corresponding to the subsequence number to 1
        current_subsequence_ |= 1 << footer->subsequence;

        if(footer->subsequence * footer->length > fb.length)
        {
          std::cerr << "[DepthPacketStreamParser::handleNewData] front buffer too short! subsequence number is " << footer->subsequence << std::endl;
        }
        else
        {
          memcpy(fb.data + (footer->subsequence * footer->length), wb.data + (wb.length - footer->length), footer->length);
        }
      }

      // reset working buffer
      wb.length = 0;
      // skip header
      in_offset += sizeof(DepthSubPacketFooter);
    }
    else
    {
      if((wb.length + 8) >= wb.capacity)
      {
        std::cerr << "[DepthPacketStreamParser::handleNewData] working buffer full, resetting it!" << std::endl;
        wb.length = 0;
        ptr_out = wb.data;
      }

      // copy remaining 8 bytes
      if((in_length - in_offset) != 8)
      {
        std::cerr << "[DepthPacketStreamParser::handleNewData] remaining data should be 8 bytes, but is " << (in_length - in_offset) << std::endl;
      }

      for(; in_offset < in_length; ++in_offset)
      {
        *ptr_out = *ptr_in;
        ++ptr_in;
        ++ptr_out;
      }

      wb.length = ptr_out - wb.data;
    }
  }
}

} /* namespace libfreenect2 */
