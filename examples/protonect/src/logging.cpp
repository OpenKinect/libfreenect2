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
#include <cstdlib>
#include <string>
#include <algorithm>

namespace libfreenect2
{
Logger::~Logger() {}


Logger::Level Logger::getDefaultLevel()
{
  Logger::Level l = Logger::Info;

  char *env_logger_level_c_str = getenv("LIBFREENECT2_LOGGER_LEVEL");

  if(env_logger_level_c_str != 0)
  {
    std::string env_logger_level_str(env_logger_level_c_str);
    std::transform(env_logger_level_str.begin(), env_logger_level_str.end(), env_logger_level_str.begin(), ::tolower);

    if(env_logger_level_str == "debug")
      l = Logger::Debug;
    else if(env_logger_level_str == "info")
      l = Logger::Info;
    else if(env_logger_level_str == "warning")
      l = Logger::Warning;
    else if(env_logger_level_str == "error")
      l = Logger::Error;
  }

  return l;
}

Logger::Level Logger::level() const
{
  return level_;
}

std::string level2str(const Logger::Level &l)
{
  switch(l)
  {
  case Logger::Debug:
    return "Debug";
  case Logger::Info:
    return "Info";
  case Logger::Warning:
    return "Warning";
  case Logger::Error:
    return "Error";
  default:
    return "";
  }
}

class ConsoleLogger : public Logger
{
public:
  ConsoleLogger(Level level)
  {
    level_ = level;
  }
  virtual ~ConsoleLogger() {}
  virtual void log(Level level, const std::string &message)
  {
    if(level < level_) return;

    (level >= Warning ? std::cerr : std::cout) << "[" << level2str(level) << "]" << message << std::endl;
  }
};

class NoopLogger : public Logger
{
public:
  NoopLogger()
  {
    level_ = Debug;
  }
  virtual ~NoopLogger() {}
  virtual void log(Level level, const std::string &message) {}
};

Logger *createConsoleLogger(Logger::Level level)
{
  return new ConsoleLogger(level);
}

Logger *createConsoleLoggerWithDefaultLevel()
{
  return new ConsoleLogger(Logger::getDefaultLevel());
}

Logger *createNoopLogger()
{
  return new NoopLogger();
}

LogMessage::LogMessage(Logger *logger, Logger::Level level) : logger_(logger), level_(level)
{

}

LogMessage::~LogMessage()
{
  if(logger_ != 0)
  {
    logger_->log(level_, stream_.str());
  }
}

std::ostream &LogMessage::stream()
{
  return stream_;
}

WithLogger::~WithLogger() {}

WithLoggerImpl::WithLoggerImpl() : logger_(0)
{
}

WithLoggerImpl::~WithLoggerImpl() {}
void WithLoggerImpl::onLoggerChanged(Logger *logger) {}

void WithLoggerImpl::setLogger(Logger *logger)
{
  logger_ = logger;
  onLoggerChanged(logger_);
}

Logger *WithLoggerImpl::logger()
{
  return logger_;
}

} /* namespace libfreenect2 */
