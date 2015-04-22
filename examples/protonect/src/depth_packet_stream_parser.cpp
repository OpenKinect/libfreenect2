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
  next_subsequence_(0)
{
  size_t single_image = 512 * 424 * 11 / 8;

  buffer_.allocate((single_image)* 10);
  buffer_.front().length = 0;
  buffer_.back().length = 0;
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
  if (in_length == 0)
    return;

  DepthSubPacketFooter *footer = 0;

  Buffer &fb = buffer_.front();

  for (size_t i = 0; i < in_length; i++)
  {
    footer = reinterpret_cast<DepthSubPacketFooter *>(&buffer[i]);

    if (footer->magic0 == 0x0 && footer->magic1 == 0x9 && footer->subsequence != 9)
    {
      if (next_subsequence_ == footer->subsequence)
      {
        // last part of current subsequence so copy up to where footer is found.
        memcpy(&fb.data[fb.length], buffer, i);
        fb.length += i;
        next_subsequence_ = footer->subsequence + 1;
      }
      else
      {
        // reset buffer if we get a sequence out of order.
        std::cerr << "[DepthPacketStreamParser::handleNewData] Subsequence out of order. Got: " << footer->subsequence << " expected: " << next_subsequence_ << std::endl;
        fb.length = 0;
        next_subsequence_ = 0;
      }
      return;
    }
    else if (footer->magic0 == 0x0 && footer->magic1 == 0x9 && footer->subsequence == 9)
    {
      // got the last subsequence so copy up to where footer is found
      next_subsequence_ = 0;

      memcpy(&fb.data[fb.length], buffer, i);
      fb.length += i;

      // does the received amount of data match expected
      if (fb.length == fb.capacity)
      {
        if (processor_->ready())
        {
          buffer_.swap();

          DepthPacket packet;
          packet.sequence = footer->sequence;
          packet.timestamp = footer->timestamp;
          packet.buffer = buffer_.back().data;
          packet.buffer_length = buffer_.back().length;

          processor_->process(packet);

        }
        else
        {
          std::cerr << "[DepthPacketStreamParser::handleNewData] skipping depth packet!" << std::endl;
        }

        // if a complete packet is processed or skipped, reset buffer.
        fb.length = 0;
        return;
      }
      else
      {
        // if data amount doesn't match, reset buffer.
        std::cerr << "[DepthPacketStreamParser::handleNewData] Depth packet not complete - resetting buffer" << std::endl;
        fb.length = 0;
        return;
      }
    }
  }
  // copy data if space
  if (fb.length + in_length > fb.capacity)
  {
    std::cerr << "[DepthPacketStreamParser::handleNewData] Buffer full - reseting" << std::endl;
    fb.length = 0;
  }

  memcpy(&fb.data[fb.length], buffer, in_length);
  fb.length += in_length;
}

} /* namespace libfreenect2 */
