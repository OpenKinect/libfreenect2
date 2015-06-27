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

namespace libfreenect2
{

class Log
{
public:
  enum Level
  {
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
  };
  static Log *console();
  virtual ~Log();

  virtual void setLevel(Level new_level);
  virtual Level level() const;

  virtual void log(Level level, const std::string &message) = 0;
protected:
  Level level_;
};

class LogMessage
{
private:
  Log *log_;
  Log::Level level_;
  std::ostringstream stream_;
public:
  LogMessage(Log *log, Log::Level level);
  ~LogMessage();

  std::ostream &stream();
};

class WithLog
{
public:
  virtual ~WithLog();
  virtual void setLog(Log *log) = 0;
  virtual Log *log() = 0;
};

class WithLogImpl : public WithLog
{
protected:
  Log *log_;

  virtual void onLogChanged(Log *log);
public:
  WithLogImpl();
  virtual ~WithLogImpl();
  virtual void setLog(Log *log);
  virtual Log *log();
};

} /* namespace libfreenect2 */

#define LOG_DEBUG (::libfreenect2::LogMessage(log(), ::libfreenect2::Log::Debug).stream())
#define LOG_INFO (::libfreenect2::LogMessage(log(), ::libfreenect2::Log::Info).stream())
#define LOG_WARNING (::libfreenect2::LogMessage(log(), ::libfreenect2::Log::Warning).stream())
#define LOG_ERROR (::libfreenect2::LogMessage(log(), ::libfreenect2::Log::Error).stream())

#endif /* LOGGING_H_ */
