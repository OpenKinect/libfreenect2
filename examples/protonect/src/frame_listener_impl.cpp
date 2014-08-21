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

#include <libfreenect2/frame_listener_impl.h>

namespace libfreenect2
{

FrameListener::~FrameListener() {}

FrameListener* FrameListener::create(unsigned int frame_types)
{
  return new SyncMultiFrameListener(frame_types);
}

SyncMultiFrameListener::SyncMultiFrameListener(unsigned int frame_types) :
    subscribed_frame_types_(frame_types),
    ready_frame_types_(0)
{
}

SyncMultiFrameListener::~SyncMultiFrameListener()
{
}

void SyncMultiFrameListener::waitForNewFrame(FrameMap &frame)
{
  libfreenect2::unique_lock l(mutex_);

  while(ready_frame_types_ != subscribed_frame_types_)
  {
    WAIT_CONDITION(condition_, mutex_, l)
  }

  frame = next_frame_;
  next_frame_.clear();
  ready_frame_types_ = 0;
}

void SyncMultiFrameListener::release(FrameMap &frame)
{
  for(FrameMap::iterator it = frame.begin(); it != frame.end(); ++it)
  {
    delete it->second;
    it->second = 0;
  }

  frame.clear();
}

bool SyncMultiFrameListener::onNewFrame(Frame::Type type, Frame *frame)
{
  if((subscribed_frame_types_ & type) == 0) return false;

  {
    libfreenect2::lock_guard l(mutex_);

    FrameMap::iterator it = next_frame_.find(type);

    if(it != next_frame_.end())
    {
      // replace frame
      delete it->second;
      it->second = frame;
    }
    else
    {
      next_frame_[type] = frame;
    }

    ready_frame_types_ |= type;
  }

  condition_.notify_one();

  return true;
}

} /* namespace libfreenect2 */
