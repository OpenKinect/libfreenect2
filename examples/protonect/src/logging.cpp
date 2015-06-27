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

#include <libfreenect2/logging.h>
#include <iostream>

namespace libfreenect2
{
Log::~Log() {}

void Log::setLevel(Level new_level)
{
  level_ = new_level;
}

Log::Level Log::level() const
{
  return level_;
}

std::string level2str(const Log::Level &l)
{
  switch(l)
  {
  case Log::Debug:
    return "Debug";
  case Log::Info:
    return "Info";
  case Log::Warning:
    return "Warning";
  case Log::Error:
    return "Error";
  default:
    return "";
  }
}

class ConsoleLog : public Log
{
public:
  ConsoleLog() {};
  virtual ~ConsoleLog() {};
  virtual void log(Level level, const std::string &message)
  {
    if(level < level_) return;

    (level >= Warning ? std::cerr : std::cout) << "[" << level2str(level) << "]" << message << std::endl;
  }
};

Log *createConsoleLog()
{
  return new ConsoleLog();
}

LogMessage::LogMessage(Log *log, Log::Level level) : log_(log), level_(level)
{

}

LogMessage::~LogMessage()
{
  if(log_ != 0)
  {
    log_->log(level_, stream_.str());
  }
}

std::ostream &LogMessage::stream()
{
  return stream_;
}

WithLog::~WithLog() {}

WithLogImpl::WithLogImpl() : log_(0)
{
}

WithLogImpl::~WithLogImpl() {}
void WithLogImpl::onLogChanged(Log *log) {}

void WithLogImpl::setLog(Log *log)
{
  log_ = log;
  onLogChanged(log_);
}

Log *WithLogImpl::log()
{
  return log_;
}

} /* namespace libfreenect2 */
