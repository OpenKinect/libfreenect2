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

#ifndef ASYNC_PACKET_PROCESSOR_H_
#define ASYNC_PACKET_PROCESSOR_H_

#include <boost/thread.hpp>

namespace libfreenect2
{

template<typename PacketT, typename PacketProcessorT>
class AsyncPacketProcessor
{
public:
  typedef PacketProcessorT* PacketProcessorPtr;

  AsyncPacketProcessor(PacketProcessorPtr processor) :
    processor_(processor),
    current_packet_available_(false),
    shutdown_(false),
    thread_(boost::bind(&AsyncPacketProcessor<PacketT, PacketProcessorT>::execute, this))
  {
  }
  virtual ~AsyncPacketProcessor()
  {
    shutdown_ = true;
    packet_condition_.notify_one();

    thread_.join();
  }

  bool ready()
  {
    // try to aquire lock, if we succeed the background thread is waiting on packet_condition_
    bool locked = packet_mutex_.try_lock();

    if(locked)
    {
      packet_mutex_.unlock();
    }

    return locked;
  }

  void process(const PacketT &packet)
  {
    {
      boost::mutex::scoped_lock l(packet_mutex_);
      current_packet_ = packet;
      current_packet_available_ = true;
    }
    packet_condition_.notify_one();
  }
private:
  PacketProcessorPtr processor_;
  bool current_packet_available_;
  PacketT current_packet_;

  bool shutdown_;
  boost::mutex packet_mutex_;
  boost::condition_variable packet_condition_;
  boost::thread thread_;

  void execute()
  {
    boost::mutex::scoped_lock l(packet_mutex_);

    while(!shutdown_)
    {
      packet_condition_.wait(l);

      if(current_packet_available_)
      {
        // invoke process impl
        processor_->process(current_packet_);

        current_packet_available_ = false;
      }
    }
  }
};

} /* namespace libfreenect2 */
#endif /* ASYNC_PACKET_PROCESSOR_H_ */
