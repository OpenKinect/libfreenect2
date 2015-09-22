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

/** @file registration.h Class for merging depth and color frames. */

#ifndef REGISTRATION_H_
#define REGISTRATION_H_

#include <string>
#include <libfreenect2/config.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener.hpp>

namespace libfreenect2
{

/** Combine frames of depth and color camera. */
class LIBFREENECT2_API Registration
{
public:
  Registration(Freenect2Device::IrCameraParams depth_p, Freenect2Device::ColorCameraParams rgb_p);

  // undistort/register a single depth data point
  void apply(int dx, int dy, float dz, float& cx, float &cy) const;

  // undistort/register a whole image
  void apply(const Frame* rgb, const Frame* depth, Frame* undistorted, Frame* registered, const bool enable_filter = true, Frame* bigdepth = 0) const;

  // compute point XYZ RGB from undistored and registered frames
  void getPointXYZRGB (const Frame* undistorted, const Frame* registered, int r, int c, float& x, float& y, float& z, float& rgb) const;

private:
  void distort(int mx, int my, float& dx, float& dy) const;
  void depth_to_color(float mx, float my, float& rx, float& ry) const;

  Freenect2Device::IrCameraParams depth;    ///< Depth camera parameters.
  Freenect2Device::ColorCameraParams color; ///< Color camera parameters.

  int distort_map[512 * 424];
  float depth_to_color_map_x[512 * 424];
  float depth_to_color_map_y[512 * 424];
  int depth_to_color_map_yi[512 * 424];

  const int filter_width_half;
  const int filter_height_half;
  const float filter_tolerance;
};

} /* namespace libfreenect2 */
#endif /* REGISTRATION_H_ */
