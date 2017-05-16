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

/** @file frame_listener_impl.cpp Implementation classes for frame listeners. */

#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/threading.h>

namespace libfreenect2
{

Frame::Frame(size_t width, size_t height, size_t bytes_per_pixel, unsigned char *data_) :
  width(width),
  height(height),
  bytes_per_pixel(bytes_per_pixel),
  data(data_),
  exposure(0.f),
  gain(0.f),
  gamma(0.f),
  status(0),
  format(Frame::Invalid),
  rawdata(NULL)
{
  if (data_)
    return;
  const size_t alignment = 64;
  size_t space = width * height * bytes_per_pixel + alignment;
  rawdata = new unsigned char[space];
  uintptr_t ptr = reinterpret_cast<uintptr_t>(rawdata);
  uintptr_t aligned = (ptr - 1u + alignment) & -alignment;
  data = reinterpret_cast<unsigned char *>(aligned);
}

Frame::~Frame()
{
  delete[] rawdata;
}

FrameListener::~FrameListener() {}

/** Implementation class for synchronizing different types of frames. */
class SyncMultiFrameListenerImpl
{
public:
  libfreenect2::mutex mutex_;
  libfreenect2::condition_variable condition_;
  FrameMap next_frame_;

  const unsigned int subscribed_frame_types_;
  unsigned int ready_frame_types_;
  bool current_frame_released_;

  SyncMultiFrameListenerImpl(unsigned int frame_types) :
    subscribed_frame_types_(frame_types),
    ready_frame_types_(0),
    current_frame_released_(true)
  {
  }

  bool hasNewFrame() const
  {
    return ready_frame_types_ == subscribed_frame_types_;
  }
};

SyncMultiFrameListener::SyncMultiFrameListener(unsigned int frame_types) :
    impl_(new SyncMultiFrameListenerImpl(frame_types))
{
}

SyncMultiFrameListener::~SyncMultiFrameListener()
{
  release(impl_->next_frame_);
  delete impl_;
}

bool SyncMultiFrameListener::hasNewFrame() const
{
  libfreenect2::unique_lock l(impl_->mutex_);

  return impl_->hasNewFrame();
}

bool SyncMultiFrameListener::waitForNewFrame(FrameMap &frame, int milliseconds)
{
#ifdef LIBFREENECT2_THREADING_STDLIB
  libfreenect2::unique_lock l(impl_->mutex_);

  auto predicate = std::bind(&SyncMultiFrameListenerImpl::hasNewFrame, impl_);

  if(impl_->condition_.wait_for(l, std::chrono::milliseconds(milliseconds), predicate))
  {
    frame = impl_->next_frame_;
    impl_->next_frame_.clear();
    impl_->ready_frame_types_ = 0;

    return true;
  }
  else
  {
    return false;
  }
#else
  waitForNewFrame(frame);
  return true;
#endif // LIBFREENECT2_THREADING_STDLIB
}

void SyncMultiFrameListener::waitForNewFrame(FrameMap &frame)
{
  libfreenect2::unique_lock l(impl_->mutex_);

  while(!impl_->hasNewFrame())
  {
    WAIT_CONDITION(impl_->condition_, impl_->mutex_, l)
  }

  frame = impl_->next_frame_;
  impl_->next_frame_.clear();
  impl_->ready_frame_types_ = 0;
  impl_->current_frame_released_ = false;
}

void SyncMultiFrameListener::release(FrameMap &frame)
{
  for(FrameMap::iterator it = frame.begin(); it != frame.end(); ++it)
  {
    delete it->second;
    it->second = 0;
  }

  frame.clear();

  {
    libfreenect2::lock_guard l(impl_->mutex_);
    impl_->current_frame_released_ = true;
  }
}

bool SyncMultiFrameListener::onNewFrame(Frame::Type type, Frame *frame)
{
  if((impl_->subscribed_frame_types_ & type) == 0) return false;

  {
    libfreenect2::lock_guard l(impl_->mutex_);

    if (!impl_->current_frame_released_)
      return false;

    FrameMap::iterator it = impl_->next_frame_.find(type);

    if(it != impl_->next_frame_.end())
    {
      // replace frame
      delete it->second;
      it->second = frame;
    }
    else
    {
      impl_->next_frame_[type] = frame;
    }

    impl_->ready_frame_types_ |= type;
  }

  impl_->condition_.notify_one();

  return true;
}

} /* namespace libfreenect2 */
