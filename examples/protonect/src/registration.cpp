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

#include <math.h>
#include <libfreenect2/registration.h>

namespace libfreenect2
{

/*
 * most information, including the table layout in command_response.h, was
 * provided by @sh0 in https://github.com/OpenKinect/libfreenect2/issues/41
 */

// these seem to be hardcoded in the original SDK
static const float depth_q = 0.01;
static const float color_q = 0.002199;

void Registration::distort(int mx, int my, float& x, float& y) const
{
  // see http://en.wikipedia.org/wiki/Distortion_(optics) for description
  float dx = ((float)mx - depth.cx) / depth.fx;
  float dy = ((float)my - depth.cy) / depth.fy;
  float dx2 = dx * dx;
  float dy2 = dy * dy;
  float r2 = dx2 + dy2;
  float dxdy2 = 2 * dx * dy;
  float kr = 1 + ((depth.k3 * r2 + depth.k2) * r2 + depth.k1) * r2;
  x = depth.fx * (dx * kr + depth.p2 * (r2 + 2 * dx2) + depth.p1 * dxdy2) + depth.cx;
  y = depth.fy * (dy * kr + depth.p1 * (r2 + 2 * dy2) + depth.p2 * dxdy2) + depth.cy;
}

void Registration::depth_to_color(float mx, float my, float& rx, float& ry) const
{
  mx = (mx - depth.cx) * depth_q;
  my = (my - depth.cy) * depth_q;

  float wx =
    (mx * mx * mx * color.mx_x3y0) + (my * my * my * color.mx_x0y3) +
    (mx * mx * my * color.mx_x2y1) + (my * my * mx * color.mx_x1y2) +
    (mx * mx * color.mx_x2y0) + (my * my * color.mx_x0y2) + (mx * my * color.mx_x1y1) +
    (mx * color.mx_x1y0) + (my * color.mx_x0y1) + (color.mx_x0y0);

  float wy =
    (mx * mx * mx * color.my_x3y0) + (my * my * my * color.my_x0y3) +
    (mx * mx * my * color.my_x2y1) + (my * my * mx * color.my_x1y2) +
    (mx * mx * color.my_x2y0) + (my * my * color.my_x0y2) + (mx * my * color.my_x1y1) +
    (mx * color.my_x1y0) + (my * color.my_x0y1) + (color.my_x0y0);

  rx = (wx / (color.fx * color_q)) - (color.shift_m / color.shift_d);
  ry = (wy / color_q) + color.cy;
}

void Registration::apply( int dx, int dy, float dz, float& cx, float &cy) const
{
  const int index = dx + dy * 512;
  float rx = depth_to_color_map_x[index];
  cy = depth_to_color_map_y[index];

  rx += (color.shift_m / dz);
  cx = rx * color.fx + color.cx;
}

void Registration::apply(const Frame *rgb, const Frame *depth, Frame *undistorted, Frame *registered) const
{
  // Check if all frames are valid and have the correct size
  if (!undistorted || !rgb || !registered ||
      rgb->width != 1920 || rgb->height != 1080 || rgb->bytes_per_pixel != 3 ||
      depth->width != 512 || depth->height != 424 || depth->bytes_per_pixel != 4 ||
      undistorted->width != 512 || undistorted->height != 424 || undistorted->bytes_per_pixel != 4 ||
      registered->width != 512 || registered->height != 424 || registered->bytes_per_pixel != 3)
    return;

  const float *depth_data = (float*)depth->data;
  float *undistorted_data = (float*)undistorted->data;
  unsigned char *registered_data = registered->data;
  const int *map_dist = distort_map;
  const float *map_x = depth_to_color_map_x;
  const int *map_yi = depth_to_color_map_yi;
  const int size_depth = 512 * 424;
  const int size_color = 1920 * 1080 * 3;
  const float color_cx = color.cx + 0.5f; // 0.5f added for later rounding

  // iterating over all pixels from undistorted depth and registered color image
  // the three maps have the same structure as the images, so their pointers are increased each iteration as well
  for (int i = 0; i < size_depth; ++i, ++registered_data, ++undistorted_data, ++map_dist, ++map_x, ++map_yi) {
    // getting index of distorted depth pixel
    const int index = *map_dist;

    // check if distorted depth pixel is outside of the depth image
    if(index < 0){
      *undistorted_data = 0;
      *registered_data = 0;
      *++registered_data = 0;
      *++registered_data = 0;
      continue;
    }

    // getting depth value for current pixel
    const float z_raw = depth_data[index];
    *undistorted_data = z_raw;

    // checking for invalid depth value
    if (z_raw <= 0.0f) {
      *registered_data = 0;
      *++registered_data = 0;
      *++registered_data = 0;
      continue;
    }

    // calculating x offset for rgb image based on depth value
    const float rx = (*map_x + (color.shift_m / z_raw)) * color.fx + color_cx;
    const int cx = rx; // same as round for positive numbers (0.5f was already added to color_cx)
    // getting y offset for depth image
    const int cy = *map_yi;
    // combining offsets
    const int c_off = cx * 3 + cy;

    // check if c_off is outside of rgb image
    // checking rx/cx is not needed because the color image is much wider then the depth image
    if (c_off < 0 || c_off >= size_color) {
      *registered_data = 0;
      *++registered_data = 0;
      *++registered_data = 0;
      continue;
    }

    // Setting RGB or registered image
    const unsigned char *rgb_data = rgb->data + c_off;
    *registered_data = *rgb_data;
    *++registered_data = *++rgb_data;
    *++registered_data = *++rgb_data;
  }
}

Registration::Registration(Freenect2Device::IrCameraParams depth_p, Freenect2Device::ColorCameraParams rgb_p):
  depth(depth_p), color(rgb_p)
{
  float mx, my;
  int ix, iy, index;
  float rx, ry;
  int *map_dist = distort_map;
  float *map_x = depth_to_color_map_x;
  float *map_y = depth_to_color_map_y;
  int *map_yi = depth_to_color_map_yi;

  for (int y = 0; y < 424; y++) {
    for (int x = 0; x < 512; x++) {
      // compute the dirstored coordinate for current pixel
      distort(x,y,mx,my);
      // rounding the values and check if the pixel is inside the image
      ix = roundf(mx);
      iy = roundf(my);
      if(ix < 0 || ix >= 512 || iy < 0 || iy >= 424)
        index = -1;
      else
        // computing the index from the coordianted for faster access to the data
        index = iy * 512 + ix;
      *map_dist++ = index;

      // compute the depth to color mapping entries for the current pixel
      depth_to_color(x,y,rx,ry);
      *map_x++ = rx;
      *map_y++ = ry;
      // compute the y offset to minimize later computations
      *map_yi++ = roundf(ry) * 1920 * 3;
    }
  }
}

} /* namespace libfreenect2 */
