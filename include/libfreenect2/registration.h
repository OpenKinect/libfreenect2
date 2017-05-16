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

class RegistrationImpl;

/** @defgroup registration Registration and Geometry
 * Register depth to color, create point clouds. */

/** Combine frames of depth and color camera. @ingroup registration
 * Right now this class uses a reverse engineered formula that uses factory
 * preset extrinsic parameters.  We do not have a clear understanding of these
 * particular extrinsic parameters and do not know how to calibrate them by
 * hand.
 *
 * If you want to perform registration with standard camera extrinsic matrix,
 * you probably need something else.
 */
class LIBFREENECT2_API Registration
{
public:
  /**
   * @param depth_p Depth camera parameters. You can use the factory values, or use your own.
   * @param rgb_p Color camera parameters. Probably use the factory values for now.
   */
  Registration(Freenect2Device::IrCameraParams depth_p, Freenect2Device::ColorCameraParams rgb_p);
  ~Registration();

  /** Undistort and register a single depth point to color camera.
   * @param dx Distorted depth coordinate x (pixel)
   * @param dy Distorted depth coordinate y (pixel)
   * @param dz Depth value (millimeter)
   * @param[out] cx Undistorted color coordinate x (normalized)
   * @param[out] cy Undistorted color coordinate y (normalized)
   */
  void apply(int dx, int dy, float dz, float& cx, float &cy) const;

  /** Map color images onto depth images
   * @param rgb Color image (1920x1080 BGRX)
   * @param depth Depth image (512x424 float)
   * @param[out] undistorted Undistorted depth image
   * @param[out] registered Color image for the depth image (512x424)
   * @param enable_filter Filter out pixels not visible to both cameras.
   * @param[out] bigdepth If not `NULL`, return mapping of depth onto colors (1920x1082 float). **1082** not 1080, with a blank top and bottom row.
   * @param[out] color_depth_map Index of mapped color pixel for each depth pixel (512x424).
   */
  void apply(const Frame* rgb, const Frame* depth, Frame* undistorted, Frame* registered, const bool enable_filter = true, Frame* bigdepth = 0, int* color_depth_map = 0) const;

  /** Undistort depth
   * @param depth Depth image (512x424 float)
   * @param[out] undistorted Undistorted depth image
   */
  void undistortDepth(const Frame* depth, Frame* undistorted) const;

  /** Construct a 3-D point with color in a point cloud.
   * @param undistorted Undistorted depth frame from apply().
   * @param registered Registered color frame from apply().
   * @param r Row (y) index in depth image.
   * @param c Column (x) index in depth image.
   * @param[out] x X coordinate of the 3-D point (meter).
   * @param[out] y Y coordinate of the 3-D point (meter).
   * @param[out] z Z coordinate of the 3-D point (meter).
   * @param[out] rgb Color of the 3-D point (BGRX). To unpack the data, use
   *
   *     const uint8_t *p = reinterpret_cast<uint8_t*>(&rgb);
   *     uint8_t b = p[0];
   *     uint8_t g = p[1];
   *     uint8_t r = p[2];
   */
  void getPointXYZRGB (const Frame* undistorted, const Frame* registered, int r, int c, float& x, float& y, float& z, float& rgb) const;

  /** Construct a 3-D point in a point cloud.
   * @param undistorted Undistorted depth frame from apply().
   * @param r Row (y) index in depth image.
   * @param c Column (x) index in depth image.
   * @param[out] x X coordinate of the 3-D point (meter).
   * @param[out] y Y coordinate of the 3-D point (meter).
   * @param[out] z Z coordinate of the 3-D point (meter).
   */
  void getPointXYZ (const Frame* undistorted, int r, int c, float& x, float& y, float& z) const;

private:
  RegistrationImpl *impl_;

  /* Disable copy and assignment constructors */
  Registration(const Registration&);
  Registration& operator=(const Registration&);
};

} /* namespace libfreenect2 */
#endif /* REGISTRATION_H_ */
