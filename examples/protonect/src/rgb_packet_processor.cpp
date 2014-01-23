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

#include <libfreenect2/rgb_packet_processor.h>

#include <fstream>
#include <string>

namespace libfreenect2
{

RgbPacketProcessor::RgbPacketProcessor()
{
}

RgbPacketProcessor::~RgbPacketProcessor()
{
}

AsyncRgbPacketProcessor::AsyncRgbPacketProcessor() :
    current_packet_available_(false),
    current_packet_(0),
    current_packet_jpeg_buffer_length_(0),
    shutdown_(false),
    thread_(boost::bind(&AsyncRgbPacketProcessor::execute, this))
{

}

AsyncRgbPacketProcessor::~AsyncRgbPacketProcessor()
{
  shutdown_ = true;
  packet_condition_.notify_one();

  thread_.join();
}

bool AsyncRgbPacketProcessor::ready()
{
  // try to aquire lock, if we succeed the background thread is waiting on packet_condition_
  bool locked = packet_mutex_.try_lock();

  if(locked)
  {
    packet_mutex_.unlock();
  }

  return locked;
}

void AsyncRgbPacketProcessor::process(RgbPacket *packet, size_t jpeg_buffer_length)
{
  {
    boost::mutex::scoped_lock l(packet_mutex_);
    current_packet_ = packet;
    current_packet_jpeg_buffer_length_ = jpeg_buffer_length;
    current_packet_available_ = true;
  }
  packet_condition_.notify_one();
}

void AsyncRgbPacketProcessor::execute()
{
  boost::mutex::scoped_lock l(packet_mutex_);

  while(!shutdown_)
  {
    packet_condition_.wait(l);

    if(current_packet_available_)
    {
      // invoke process impl
      doProcess(current_packet_, current_packet_jpeg_buffer_length_);

      current_packet_ = 0;
      current_packet_jpeg_buffer_length_ = 0;
      current_packet_available_ = false;
    }
  }
}

DumpRgbPacketProcessor::DumpRgbPacketProcessor()
{
}

DumpRgbPacketProcessor::~DumpRgbPacketProcessor()
{
}

void DumpRgbPacketProcessor::doProcess(RgbPacket *packet, size_t jpeg_buffer_length)
{
  //std::stringstream name;
  //name << packet->sequence << "_" << packet->unknown0 << "_" << jpeg_buffer_length << ".jpeg";
  //
  //std::ofstream file(name.str().c_str());
  //file.write(reinterpret_cast<char *>(packet->jpeg_buffer), jpeg_buffer_length);
  //file.close();
}

} /* namespace libfreenect2 */
