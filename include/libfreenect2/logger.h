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

/** Logger class. */
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
  static Level getDefaultLevel();
  static std::string level2str(Level level);

  virtual ~Logger();

  virtual Level level() const;

  virtual void log(Level level, const std::string &message) = 0;
protected:
  Level level_;
};

LIBFREENECT2_API Logger *createConsoleLogger(Logger::Level level);
LIBFREENECT2_API Logger *createConsoleLoggerWithDefaultLevel();

//libfreenect2 frees the memory of the logger passed in.
LIBFREENECT2_API Logger *getGlobalLogger();
LIBFREENECT2_API void setGlobalLogger(Logger *logger);

} /* namespace libfreenect2 */
#endif /* LIBFREENECT2_LOGGER_H_ */
