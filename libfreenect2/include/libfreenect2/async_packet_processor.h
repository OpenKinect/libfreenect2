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

#include <libfreenect2/threading.h>
#include <libfreenect2/packet_processor.h>

namespace libfreenect2
{

template<typename PacketT>
class AsyncPacketProcessor : public PacketProcessor<PacketT>
{
public:
  typedef PacketProcessor<PacketT>* PacketProcessorPtr;

  AsyncPacketProcessor(PacketProcessorPtr processor) :
    processor_(processor),
    current_packet_available_(false),
    shutdown_(false),
    thread_(&AsyncPacketProcessor<PacketT>::static_execute, this)
  {
  }
  virtual ~AsyncPacketProcessor()
  {
    shutdown_ = true;
    packet_condition_.notify_one();

    thread_.join();
  }

  virtual bool ready()
  {
    // try to aquire lock, if we succeed the background thread is waiting on packet_condition_
    bool locked = packet_mutex_.try_lock();

    if(locked)
    {
      packet_mutex_.unlock();
    }

    return locked;
  }

  virtual void process(const PacketT &packet)
  {
    {
      libfreenect2::lock_guard l(packet_mutex_);
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
  libfreenect2::mutex packet_mutex_;
  libfreenect2::condition_variable packet_condition_;
  libfreenect2::thread thread_;

  static void static_execute(void *data)
  {
    static_cast<AsyncPacketProcessor<PacketT> *>(data)->execute();
  }

  void execute()
  {
    libfreenect2::unique_lock l(packet_mutex_);

    while(!shutdown_)
    {
      WAIT_CONDITION(packet_condition_, packet_mutex_, l);

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
