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

void Registration::undistort_depth(int x, int y, float& mx, float& my)
{
  float dx = ((float)x - depth.cx) / depth.fx;
  float dy = ((float)y - depth.cy) / depth.fy;

  float ps = (dx * dx) + (dy * dy);
  float qs = ((ps * depth.k3 + depth.k2) * ps + depth.k1) * ps + 1.0;
  for (int i = 0; i < 9; i++) {
    float qd = ps / (qs * qs);
    qs = ((qd * depth.k3 + depth.k2) * qd + depth.k1) * qd + 1.0;
  }

  mx = dx / qs;
  my = dy / qs;
}

void Registration::depth_to_color(float mx, float my, float& rx, float& ry)
{
  mx *= depth.fx * depth_q;
  my *= depth.fy * depth_q;

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

void Registration::apply(const Frame* rgb, const Frame* depth, Frame* registered) const
{
  if (!depth || !rgb || !registered ||
      depth->width != 512 || depth->height != 424 || depth->bytes_per_pixel != 4 ||
      rgb->width != 1920 || rgb->height != 1080 || rgb->bytes_per_pixel != 3 ||
      registered->width != 512 || registered->height != 424 || registered->bytes_per_pixel != 3)
    return;

  const float *depth_raw = (float*)depth->data;
  const float *map_x = depth_to_color_map_x;
  const int *map_i = depth_to_color_map_i;
  unsigned char *registered_raw = registered->data;
  const int size_depth = 512 * 424;
  const int size_color = 1920 * 1080 * 3;

  for (int i = 0; i < size_depth; ++i, ++registered_raw, ++map_x, ++map_i, ++depth_raw) {
    const float z_raw = *depth_raw;

    if (z_raw == 0.0) {
      *registered_raw = 0;
      *++registered_raw = 0;
      *++registered_raw = 0;
      continue;
    }

    const float rx = (*map_x + (color.shift_m / z_raw)) * color.fx + color.cx;
    const int cx = rx + 0.5f; // same as round for positive numbers
    const int cy = *map_i;
    const int c_off = cx * 3 + cy;

    if (c_off < 0 || c_off > size_color || rx < -0.5f) {
      *registered_raw = 0;
      *++registered_raw = 0;
      *++registered_raw = 0;
      continue;
    }

    const unsigned char *rgb_data = rgb->data + c_off;
    *registered_raw = *rgb_data;
    *++registered_raw = *++rgb_data;
    *++registered_raw = *++rgb_data;
  }
}

Registration::Registration(Freenect2Device::IrCameraParams depth_p, Freenect2Device::ColorCameraParams rgb_p):
  depth(depth_p), color(rgb_p)
{
  float mx, my;
  float rx, ry;
  float *it_undist = undistort_map;
  float *map_x = depth_to_color_map_x;
  float *map_y = depth_to_color_map_y;
  int *map_i = depth_to_color_map_i;

  for (int y = 0; y < 424; y++) {
    for (int x = 0; x < 512; x++) {

      undistort_depth(x,y,mx,my);
      *it_undist++ = mx;
      *it_undist++ = my;

      depth_to_color(mx,my,rx,ry);
      *map_x++ = rx;
      *map_y++ = ry;
      *map_i++ = round(ry) * 1920 * 3;
    }
  }
}

} /* namespace libfreenect2 */
