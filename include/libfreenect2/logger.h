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

/** @file logger.h Declaration of logging classes. */

#ifndef LIBFREENECT2_LOGGER_H_
#define LIBFREENECT2_LOGGER_H_

#include <string>

#include <libfreenect2/config.h>

namespace libfreenect2
{

/** @defgroup logging Logging utilities
 * Specify logging level and custom logging destination. */
///@{

/** Provide interfaces to receive log messages.
 * You can inherit this class and implement your custom logger. */
class LIBFREENECT2_API Logger
{
public:
  /** Available levels of logging, higher is more output. */
  enum Level
  {
    None = 0,
    Error = 1,
    Warning = 2,
    Info = 3,
    Debug = 4,
  };

  /** Default is Info, or overridden by environment variable `LIBFREENECT2_LOGGER_LEVEL`.
   * `LIBFREENECT2_LOGGER_LEVEL` can contain a case-insensitive name of level.
   */
  static Level getDefaultLevel();

  /** Convert logging level to a human-readable name.
   */
  static std::string level2str(Level level);

  virtual ~Logger();

  /** Get the level of the logger; the level is immutable. */
  virtual Level level() const;

  /** libfreenect2 calls this function to output all log messages. */
  virtual void log(Level level, const std::string &message) = 0;
protected:
  Level level_;
};

/** Allocate a Logger instance that outputs log to standard input/output  */
LIBFREENECT2_API Logger *createConsoleLogger(Logger::Level level);

/** @copybrief Logger::getDefaultLevel
 *
 * %libfreenect2 will have an initial global logger created with createConsoleLoggerWithDefaultLevel().
 * You do not have to explicitly call this if the default is already what you want.
 */
LIBFREENECT2_API Logger *createConsoleLoggerWithDefaultLevel();

/** Get the pointer to the current logger.
 * @return Pointer to the logger. This is purely informational. You should not free the pointer.
 */
LIBFREENECT2_API Logger *getGlobalLogger();

/** Set the logger for all log output in this library.
 * @param logger Pointer to your logger, or `NULL` to disable logging. The memory will be freed automatically. You should not free the pointer.
 */
LIBFREENECT2_API void setGlobalLogger(Logger *logger);

///@}
} /* namespace libfreenect2 */
#endif /* LIBFREENECT2_LOGGER_H_ */
