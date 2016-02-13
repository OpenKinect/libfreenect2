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

#include "libfreenect2/allocator.h"
#include "libfreenect2/threading.h"

namespace libfreenect2
{
class NewAllocator: public Allocator
{
public:
  virtual Buffer *allocate(size_t size)
  {
    Buffer *b = new Buffer;
    b->data = new unsigned char[size];
    b->length = 0;
    b->capacity = size;
    return b;
  }

  virtual void free(Buffer *b)
  {
    if (b == NULL)
      return;
    delete[] b->data;
    delete b;
  }
};

class PoolAllocatorImpl
{
private:
  Allocator *allocator;
  Buffer *buffers[2];
  mutex locks[2];
  int next;
public:
  PoolAllocatorImpl(Allocator *a): allocator(a), buffers(), next(0) {}

  Buffer *allocate(size_t size)
  {
    locks[next].lock();
    if (buffers[next] == NULL)
      buffers[next] = allocator->allocate(size);
    buffers[next]->length = 0;
    Buffer *b = buffers[next];
    next = !next;
    return b;
  }

  void free(Buffer *b)
  {
    if (b == buffers[0])
      locks[0].unlock();
    else if (b == buffers[1])
      locks[1].unlock();
  }

  ~PoolAllocatorImpl()
  {
    allocator->free(buffers[0]);
    allocator->free(buffers[1]);
    delete allocator;
  }
};

PoolAllocator::PoolAllocator():
  impl_(new PoolAllocatorImpl(new NewAllocator))
{
}

PoolAllocator::PoolAllocator(Allocator *a):
  impl_(new PoolAllocatorImpl(a))
{
}

PoolAllocator::~PoolAllocator()
{
  delete impl_;
}

Buffer *PoolAllocator::allocate(size_t size)
{
  return impl_->allocate(size);
}

void PoolAllocator::free(Buffer *b)
{
  impl_->free(b);
}
} // namespace libfreenect2
