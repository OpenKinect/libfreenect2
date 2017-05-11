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

#ifndef FRAME_LISTENER_IMPL_H_
#define FRAME_LISTENER_IMPL_H_

#include <map>

#include <libfreenect2/config.h>
#include <libfreenect2/frame_listener.hpp>

namespace libfreenect2
{

typedef std::map<Frame::Type, Frame*> FrameMap;

class SyncMultiFrameListenerImpl;

class LIBFREENECT2_API SyncMultiFrameListener : public FrameListener
{
public:
  SyncMultiFrameListener(unsigned int frame_types);
  virtual ~SyncMultiFrameListener();

  bool hasNewFrame() const;

#ifdef  LIBFREENECT2_THREADING_STDLIB
  bool waitForNewFrame(FrameMap &frame, int milliseconds);
#endif // LIBFREENECT2_THREADING_STDLIB
  // for now the caller is responsible to release the frames when he is done
  void waitForNewFrame(FrameMap &frame);

  void release(FrameMap &frame);

  virtual bool onNewFrame(Frame::Type type, Frame *frame);
private:
  SyncMultiFrameListenerImpl *impl_;
};

} /* namespace libfreenect2 */
#endif /* FRAME_LISTENER_IMPL_H_ */
