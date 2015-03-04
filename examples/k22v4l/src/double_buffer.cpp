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

#include <libfreenect2/double_buffer.h>

namespace libfreenect2
{

DoubleBuffer::DoubleBuffer() :
    front_buffer_index_(0),
    buffer_data_(0)
{
}

DoubleBuffer::~DoubleBuffer()
{
  if(buffer_data_ != 0)
  {
    buffer_[0].data = 0;
    buffer_[1].data = 0;
    delete[] buffer_data_;
  }
}

void DoubleBuffer::allocate(size_t buffer_size)
{
  size_t total_buffer_size = 2 * buffer_size;
  buffer_data_ = new unsigned char[total_buffer_size];

  buffer_[0].capacity = buffer_size;
  buffer_[0].length = 0;
  buffer_[0].data = buffer_data_;

  buffer_[1].capacity = buffer_size;
  buffer_[1].length = 0;
  buffer_[1].data = buffer_data_ + buffer_size;
}

void DoubleBuffer::swap()
{
  front_buffer_index_ = (front_buffer_index_ + 1) & 1;
}

Buffer& DoubleBuffer::front()
{
  return buffer_[front_buffer_index_ & 1];
}

Buffer& DoubleBuffer::back()
{
  return buffer_[(front_buffer_index_ + 1) & 1];
}

} /* namespace libfreenect2 */
