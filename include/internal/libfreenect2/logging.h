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

#ifndef LOGGING_H_
#define LOGGING_H_

#include <string>
#include <sstream>

#include <libfreenect2/config.h>
#include <libfreenect2/logger.h>

namespace libfreenect2
{

class WithPerfLoggingImpl;

class WithPerfLogging
{
public:
  WithPerfLogging();
  virtual ~WithPerfLogging();
  void startTiming();
  std::ostream &stopTiming(std::ostream &stream);
private:
  WithPerfLoggingImpl *impl_;
};

class LogMessage
{
private:
  Logger *logger_;
  Logger::Level level_;
  std::ostringstream stream_;
public:
  LogMessage(Logger *logger, Logger::Level level);
  LogMessage(Logger *logger, Logger::Level level, const char *source);
  ~LogMessage();

  std::ostream &stream();
};

} /* namespace libfreenect2 */

#if defined(__GNUC__) || defined(__clang__)
#define LOG_SOURCE __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define LOG_SOURCE __FUNCSIG__
#else
#define LOG_SOURCE ""
#endif

#define LOG(LEVEL) (::libfreenect2::LogMessage(::libfreenect2::getGlobalLogger(), ::libfreenect2::Logger::LEVEL, LOG_SOURCE).stream())
#define LOG_DEBUG LOG(Debug)
#define LOG_INFO LOG(Info)
#define LOG_WARNING LOG(Warning)
#define LOG_ERROR LOG(Error)

#endif /* LOGGING_H_ */
