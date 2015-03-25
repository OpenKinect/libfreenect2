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

#ifndef REGISTRATION_H_
#define REGISTRATION_H_

#include <string>
#include <libfreenect2/config.h>
#include <libfreenect2/libfreenect2.hpp>

namespace libfreenect2
{

class LIBFREENECT2_API Registration
{
public:
  Registration(Freenect2Device::IrCameraParams *depth_p, Freenect2Device::ColorCameraParams *rgb_p);

  void apply( int dx, int dy, float dz, float& cx, float &cy);

private:
  void undistort_depth(int dx, int dy, float& mx, float& my);
  void depth_to_color(float mx, float my, float& rx, float& ry);

  Freenect2Device::IrCameraParams depth;
  Freenect2Device::ColorCameraParams color;

  float undistort_map[512][424][2];
  float depth_to_color_map[512][424][2];
};

} /* namespace libfreenect2 */
#endif /* REGISTRATION_H_ */
