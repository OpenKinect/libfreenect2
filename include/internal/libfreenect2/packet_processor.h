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

/** @file packet_processor.h Packet processor definitions. */

#ifndef PACKET_PROCESSOR_H_
#define PACKET_PROCESSOR_H_

#include "libfreenect2/allocator.h"

namespace libfreenect2
{

/**
 * Processor node in the pipeline.
 * @tparam PacketT Type of the packet being processed.
 */
template<typename PacketT>
class PacketProcessor
{
public:
  virtual ~PacketProcessor() {}

  /**
   * Test whether the processor is idle.
   * @return True if the processor is idle, else false.
   */
  virtual bool ready() { return true; }

  virtual bool good() { return true; }

  virtual const char *name() { return "a packet processor"; }

  /**
   * A new packet has arrived, process it.
   * @param packet Packet to process.
   */
  virtual void process(const PacketT &packet) = 0;

  virtual void allocateBuffer(PacketT &p, size_t size)
  {
    Allocator *a = getAllocator();
    if (a)
      p.memory = a->allocate(size);
  }

  virtual void releaseBuffer(PacketT &p)
  {
    Allocator *a = getAllocator();
    if (a)
      a->free(p.memory);
    p.memory = NULL;
  }

protected:
  virtual Allocator *getAllocator() { return &default_allocator_; }

private:
  PoolAllocator default_allocator_;
};

/**
 * Dummy processor class.
 * @tparam PacketT Type of the packet being processed.
 */
template<typename PacketT>
class NoopPacketProcessor : public PacketProcessor<PacketT>
{
public:
  NoopPacketProcessor() {}
  virtual ~NoopPacketProcessor() {}

  virtual void process(const PacketT &packet) {}
};

/**
 * Factory function for creating a dummy packet processor.
 * @tparam PacketT Type of the packet being processed.
 * @return The dummy (noop) processor.
 */
template<typename PacketT>
PacketProcessor<PacketT> *noopProcessor()
{
  static NoopPacketProcessor<PacketT> noop_processor_;
  return &noop_processor_;
}

} /* namespace libfreenect2 */
#endif /* PACKET_PROCESSOR_H_ */
