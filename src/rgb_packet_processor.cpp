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

/** @file rgb_packet_processor.cpp Implementation of generic color packet processors. */

#include <libfreenect2/rgb_packet_processor.h>
#include <libfreenect2/async_packet_processor.h>

#include <cstring>
#include <fstream>
#include <string>

namespace libfreenect2
{

RgbPacketProcessor::RgbPacketProcessor() :
    listener_(0)
{
}

RgbPacketProcessor::~RgbPacketProcessor()
{
}

void RgbPacketProcessor::setFrameListener(libfreenect2::FrameListener *listener)
{
  listener_ = listener;
}

DumpRgbPacketProcessor::DumpRgbPacketProcessor() {}
DumpRgbPacketProcessor::~DumpRgbPacketProcessor() {}

void DumpRgbPacketProcessor::process(const RgbPacket &packet)
{
  Frame *frame = new Frame(1, 1, 1920*1080*4);
  frame->sequence = packet.sequence;
  frame->timestamp = packet.timestamp;
  frame->exposure = packet.exposure;
  frame->gain = packet.gain;
  frame->gamma = packet.gamma;
  frame->format = Frame::Raw;
  frame->bytes_per_pixel = packet.jpeg_buffer_length;

  std::memcpy(frame->data, packet.jpeg_buffer, packet.jpeg_buffer_length);

  if (!listener_->onNewFrame(Frame::Color, frame)) {
    delete frame;
  }
  frame = NULL;
}

} /* namespace libfreenect2 */
