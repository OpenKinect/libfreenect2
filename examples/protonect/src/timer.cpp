/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2015 individual OpenKinect contributors. See the CONTRIB file
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

#include <libfreenect2/timer.h>

#ifdef LIBFREENECT2_WITH_CXX11_SUPPORT
#include <chrono>
#endif

#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
#include <GLFW/glfw3.h>
#endif

namespace libfreenect2 {

#ifdef LIBFREENECT2_WITH_CXX11_SUPPORT
class TimerImpl {
 public:
  void start() {
    time_start = std::chrono::high_resolution_clock::now();
  }

  double stop() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - time_start).count();
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> time_start;
};
#elif defined(LIBFREENECT2_WITH_OPENGL_SUPPORT)
class TimerImpl {
 public:
  void start() {
    time_start = glfwGetTime();
  }

  double stop() {
    return glfwGetTime() - time_start;
  }

  double time_start;
};
#else
class TimerImpl {
 public:
  void start() {
  }

  double stop() {
    return 0;
  }
};
#endif

Timer::Timer() :
    impl_(new TimerImpl()) {
}

Timer::~Timer() {
  delete impl_;
}

void Timer::start() {
  impl_->start();
}

double Timer::stop() {
  return impl_->stop();
}

} /* namespace libfreenect2 */