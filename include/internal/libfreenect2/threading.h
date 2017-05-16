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

/** @file threading.h Threading abstraction definitions. */

#ifndef THREADING_H_
#define THREADING_H_

#include <libfreenect2/config.h>

#ifdef LIBFREENECT2_THREADING_STDLIB

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#define WAIT_CONDITION(var, mutex, lock) var.wait(lock);

namespace libfreenect2
{

typedef std::thread thread;
typedef std::mutex mutex;
typedef std::lock_guard<std::mutex> lock_guard;
typedef std::unique_lock<std::mutex> unique_lock;
typedef std::condition_variable condition_variable;

namespace chrono
{
using namespace std::chrono;
}

namespace this_thread
{
using namespace std::this_thread;
}

} /* libfreenect2 */

#endif

#ifdef LIBFREENECT2_THREADING_TINYTHREAD

#include <tinythread.h>

// TODO: work around for tinythread incompatibility
#define WAIT_CONDITION(var, mutex, lock) var.wait(mutex);

namespace libfreenect2
{

typedef tthread::thread thread;
typedef tthread::mutex mutex;
typedef tthread::lock_guard<tthread::mutex> lock_guard;
// TODO: this is not optimal
typedef tthread::lock_guard<tthread::mutex> unique_lock;
typedef tthread::condition_variable condition_variable;

namespace chrono
{
using namespace tthread::chrono;
}

namespace this_thread
{
using namespace tthread::this_thread;
}

} /* libfreenect2 */

#endif

#if defined(__linux__)
#include <sys/prctl.h>
#elif defined(__APPLE__)
#include <pthread.h>
#endif

namespace libfreenect2
{
namespace this_thread
{
  static inline void set_name(const char *name)
  {
#if defined(__linux__)
    prctl(PR_SET_NAME, name);
#elif defined(__APPLE__)
    pthread_setname_np(name);
#endif
  }
}
}

#endif /* THREADING_H_ */
