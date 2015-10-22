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

/** @file logging.cpp Logging message handler classes. */

#include <libfreenect2/logging.h>
#include <iostream>
#include <cstdlib>
#include <string>
#include <algorithm>

#ifdef LIBFREENECT2_WITH_CXX11_SUPPORT
#include <chrono>
#endif

#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
#include <GLFW/glfw3.h>
#endif

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
    else if(env_logger_level_str == "none")
      l = Logger::None;
  }

  return l;
}

Logger::Level Logger::level() const
{
  return level_;
}

/**
 * Convert logging level to a human-readable name.
 * @param l Logging level to convert.
 * @return Human readable name for the logging level.
 */
std::string Logger::level2str(Level l)
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

/** Logger class to the console (stderr). */
class ConsoleLogger : public Logger
{
public:
  ConsoleLogger(Level level)
  {
    level_ = level;
    //std::ios_base::unitbuf causes automatic flushing which access
    //thread local variable via std::uncaught_exception().
    //This causes deadlock with ocl-icd until its recent update.
    //Accessing TLS has a slight performance penalty.
    //log() always flush the ostream so unitbuf is unnecessary anyway.
    std::nounitbuf(std::cerr);
  }
  virtual ~ConsoleLogger() {}
  virtual void log(Level level, const std::string &message)
  {
    if(level > level_) return;

    (level <= Warning ? std::cerr : std::cout) << "[" << level2str(level) << "] " << message << std::endl;
  }
};

Logger *createConsoleLogger(Logger::Level level)
{
  return new ConsoleLogger(level);
}

Logger *createConsoleLoggerWithDefaultLevel()
{
  return new ConsoleLogger(Logger::getDefaultLevel());
}

LogMessage::LogMessage(Logger *logger, Logger::Level level) : logger_(logger), level_(level)
{

}

LogMessage::~LogMessage()
{
  if(logger_ != 0 && stream_.good())
  {
    const std::string &message = stream_.str();
    if (message.size())
      logger_->log(level_, message);
  }
}

std::ostream &LogMessage::stream()
{
  return stream_;
}

static ConsoleLogger defaultLogger_(Logger::getDefaultLevel());
static Logger *userLogger_ = &defaultLogger_;

Logger *getGlobalLogger()
{
  return userLogger_;
}

void setGlobalLogger(Logger *logger)
{
  if (userLogger_ != &defaultLogger_)
    delete userLogger_;
  userLogger_ = logger;
}

/** Timer for measuring performance. */
class Timer
{
 public:
  double duration;
  size_t count;

  Timer()
  {
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

  void stop()
  {
    duration += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - time_start).count();
    count++;
  }
#elif defined(LIBFREENECT2_WITH_OPENGL_SUPPORT)
  double time_start;

  void start()
  {
    time_start = glfwGetTime();
  }

  void stop()
  {
    duration += glfwGetTime() - time_start;
    count++;
  }
#else
  void start()
  {
  }

  void stop()
  {
  }
#endif
};

class WithPerfLoggingImpl: public Timer
{
public:
  std::ostream &stop(std::ostream &stream)
  {
    Timer::stop();
    if (count < 100)
    {
      stream.setstate(std::ios::eofbit);
      return stream;
    }
    double avg = duration / count;
    reset();
    stream << "avg. time: " << (avg * 1000) << "ms -> ~" << (1.0/avg) << "Hz";
    return stream;
  }
};

WithPerfLogging::WithPerfLogging()
  :impl_(new WithPerfLoggingImpl)
{
}

WithPerfLogging::~WithPerfLogging()
{
  delete impl_;
}

void WithPerfLogging::startTiming()
{
  impl_->start();
}

std::ostream &WithPerfLogging::stopTiming(std::ostream &stream)
{
  return impl_->stop(stream);
}

std::string getShortName(const char *func)
{
  std::string src(func);
  size_t end = src.rfind('(');
  if (end == std::string::npos)
    end = src.size();
  size_t begin = 1 + src.rfind(' ', end);
  size_t first_ns = src.find("::", begin);
  if (first_ns != std::string::npos)
    begin = first_ns + 2;
  size_t last_ns = src.rfind("::", end);
  if (last_ns != std::string::npos)
    end = last_ns;
  return src.substr(begin, end - begin);
}

} /* namespace libfreenect2 */
