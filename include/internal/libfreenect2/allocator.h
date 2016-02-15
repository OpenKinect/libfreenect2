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

/** @file allocator.h Allocator definitions. */

#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_

#include <cstddef>

namespace libfreenect2
{

class Allocator;

class Buffer
{
public:
  size_t capacity; ///< Capacity of the buffer.
  size_t length;   ///< Used length of the buffer.
  unsigned char* data; ///< Start address of the buffer.
  Allocator *allocator;
};

class Allocator
{
public:
  /* If the inner allocator fails to allocate, it MUST not return NULL.
   * Instead it MUST return a Buffer with data being NULL.
   * The stdlib will throw std::bad_alloc if new Buffer fails.
   */
  virtual Buffer *allocate(size_t size) = 0;
  virtual void free(Buffer *b) = 0;
  virtual ~Allocator() {}
};

class PoolAllocatorImpl;

class PoolAllocator: public Allocator
{
public:
  /* Use new as the inner allocator. */
  PoolAllocator();

  /* This inner allocator will be freed by PoolAllocator. */
  PoolAllocator(Allocator *inner);

  virtual ~PoolAllocator();

  /* allocate() will block until an allocation is possible.
   * It should be called as late as possible before the memory is required
   * for write access.
   *
   * This allocate() never returns NULL as required by Allocator.
   *
   * All calls to allocate() MUST have the same size.
   *
   * allocate() MUST be called from the same thread.
   */
  virtual Buffer *allocate(size_t size);

  /* free() will unblock pending allocation.
   * It should be called as early as possible after the memory is no longer
   * required for read access.
   *
   * The inner free() can be called with NULL.
   *
   * free() can be called from different threads than allocate().
   */
  virtual void free(Buffer *b);
private:
  PoolAllocatorImpl *impl_;
};

} /* namespace libfreenect2 */
#endif /* ALLOCATOR_H_ */
