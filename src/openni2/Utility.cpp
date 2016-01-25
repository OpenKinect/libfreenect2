/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2014 Benn Snyder, 2015 individual OpenKinect contributors.
 * See the CONTRIB file for details.
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
#include <cstring>
#include "Driver/OniDriverAPI.h"


// Oni helpers

OniVideoMode makeOniVideoMode(OniPixelFormat pixel_format, int resolution_x, int resolution_y, int frames_per_second)
{
  OniVideoMode mode;
  mode.pixelFormat = pixel_format;
  mode.resolutionX = resolution_x;
  mode.resolutionY = resolution_y;
  mode.fps = frames_per_second;
  return mode;
}

bool operator==(const OniVideoMode& left, const OniVideoMode& right)
{
  return (left.pixelFormat == right.pixelFormat && left.resolutionX == right.resolutionX
          && left.resolutionY == right.resolutionY && left.fps == right.fps);
}

bool operator<(const OniVideoMode& left, const OniVideoMode& right)
{
  return (left.resolutionX * left.resolutionY < right.resolutionX * right.resolutionY);
}

bool operator<(const OniDeviceInfo& left, const OniDeviceInfo& right)
{
  return (strcmp(left.uri, right.uri) < 0);
}
