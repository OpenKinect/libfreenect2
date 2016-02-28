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

/** @file time.h Timer utility. */

#ifdef LIBFREENECT2_WITH_CXX11_SUPPORT
#include <chrono>
#endif

#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
#include <GLFW/glfw3.h>
#endif

namespace libfreenect2
{

class Timer
{
 public:
  double duration;
  size_t count;

  Timer()
  {
#if defined(LIBFREENECT2_WITH_OPENGL_SUPPORT)
    glfwInit();
#endif
    reset();
  }

  void reset()
  {
    duration = 0;
    count = 0;
  }

#ifdef LIBFREENECT2_WITH_CXX11_SUPPORT
  std::chrono::time_point<std::chrono::high_resolution_clock> time_start;

  void start()
  {
    time_start = std::chrono::high_resolution_clock::now();
  }

  double elapsed()
  {
    auto time_diff = std::chrono::high_resolution_clock::now() - time_start;
    double this_duration = std::chrono::duration_cast<std::chrono::duration<double>>(time_diff).count();
    return this_duration;
  }
#elif defined(LIBFREENECT2_WITH_OPENGL_SUPPORT)
  double time_start;

  void start()
  {
    time_start = glfwGetTime();
  }

  double elapsed()
  {
    double this_duration = glfwGetTime() - time_start;
    return this_duration;
  }
#else
  void start()
  {
  }

  double elapsed()
  {
    return 0;
  }
#endif
  double stop()
  {
    double this_duration = elapsed();
    duration += this_duration;
    count++;
    return this_duration;
  }
};

} /* namespace libfreenect2 */
