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

// This file contains symbols that may be used by any class or don't really go anywhere else.
#pragma once

#include <iostream>
#include "Driver/OniDriverAPI.h"


// Oni helpers

static OniVideoMode makeOniVideoMode(OniPixelFormat pixel_format, int resolution_x, int resolution_y, int frames_per_second)
{
  OniVideoMode mode;
  mode.pixelFormat = pixel_format;
  mode.resolutionX = resolution_x;
  mode.resolutionY = resolution_y;
  mode.fps = frames_per_second;
  return mode;
}
static bool operator==(const OniVideoMode& left, const OniVideoMode& right)
{
  return (left.pixelFormat == right.pixelFormat && left.resolutionX == right.resolutionX
          && left.resolutionY == right.resolutionY && left.fps == right.fps);
}
static bool operator<(const OniVideoMode& left, const OniVideoMode& right)
{
  return (left.resolutionX * left.resolutionY < right.resolutionX * right.resolutionY);
}

static bool operator<(const OniDeviceInfo& left, const OniDeviceInfo& right)
{
  return (strcmp(left.uri, right.uri) < 0);
}


/// Extracts `first` from `pair`, for transforming a map into its keys.
struct ExtractKey
{
  template <typename T> typename T::first_type
  operator()(T pair) const
  {
    return pair.first;
  }
};


// holding out on C++11
template <typename T>
static std::string to_string(const T& n)
{
  std::ostringstream oss;
  oss << n;
  return oss.str();
}


// global logging
namespace Freenect2Driver
{
  // DriverServices is set in DeviceDriver.cpp so all files can call errorLoggerAppend()
  static oni::driver::DriverServices* DriverServices;

  // from XnLog.h
  typedef enum XnLogSeverity {
    XN_LOG_VERBOSE = 0,
    XN_LOG_INFO = 1,
    XN_LOG_WARNING = 2,
    XN_LOG_ERROR = 3,
    XN_LOG_SEVERITY_NONE = 10,
  } XnLogSeverity;
}
#define FN2DRV_LOG_MASK "Freenect2Driver"
#define WriteVerbose(str) do { if (DriverServices != NULL) DriverServices->log(XN_LOG_VERBOSE, __FILE__, __LINE__, FN2DRV_LOG_MASK, std::string(str).c_str()); } while(0)
#define WriteInfo(str)    do { if (DriverServices != NULL) DriverServices->log(XN_LOG_INFO,    __FILE__, __LINE__, FN2DRV_LOG_MASK, std::string(str).c_str()); } while(0)
#define WriteWarning(str) do { if (DriverServices != NULL) DriverServices->log(XN_LOG_WARNING, __FILE__, __LINE__, FN2DRV_LOG_MASK, std::string(str).c_str()); } while(0)
#define WriteError(str)   do { if (DriverServices != NULL) DriverServices->log(XN_LOG_ERROR,   __FILE__, __LINE__, FN2DRV_LOG_MASK, std::string(str).c_str()); } while(0)
#define WriteMessage(str) WriteInfo(str)
#define LogError(str)     WriteError(str)
