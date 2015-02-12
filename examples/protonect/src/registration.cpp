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

void Registration::undistort_depth(float dx, float dy, float& mx, float& my)
{
  float ps = (dx * dx) + (dy * dy);
  float qs = ((ps * depth->k3 + depth->k2) * ps + depth->k1) * ps + 1.0;
  for (int i = 0; i < 9; i++) {
    float qd = ps / (qs * qs);
    qs = ((qd * depth->k3 + depth->k2) * qd + depth->k1) * qd + 1.0;
  }
  mx = dx / qs;
  my = dy / qs;
}

void Registration::depth_to_color(float mx, float my, float& rx, float& ry)
{
  mx *= depth->fx * depth_q;
  my *= depth->fy * depth_q;

  float wx =
    (mx * mx * mx * color->mx_x3y0) + (my * my * my * color->mx_x0y3) +
    (mx * mx * my * color->mx_x2y1) + (my * my * mx * color->mx_x1y2) +
    (mx * mx * color->mx_x2y0) + (my * my * color->mx_x0y2) + (mx * my * color->mx_x1y1) +
    (mx * color->mx_x1y0) + (my * color->mx_x0y1) + (color->mx_x0y0);

  float wy =
    (mx * mx * mx * color->my_x3y0) + (my * my * my * color->my_x0y3) +
    (mx * mx * my * color->my_x2y1) + (my * my * mx * color->my_x1y2) +
    (mx * mx * color->my_x2y0) + (my * my * color->my_x0y2) + (mx * my * color->my_x1y1) +
    (mx * color->my_x1y0) + (my * color->my_x0y1) + (color->my_x0y0);

  rx = wx / (color->color_f * color_q);
  ry = wy / (color->color_f * color_q);
}

void Registration::apply( int dx, int dy, float dz, float& cx, float &cy)
{
  float rx = depth_to_color_map[dx][dy][0];
  float ry = depth_to_color_map[dx][dy][1];

  rx += (color->shift_m / dz) - (color->shift_m / color->shift_d);

  cx = rx * color->color_f + color->color_cx;
  cy = ry * color->color_f + color->color_cy;
}

Registration::Registration(protocol::DepthCameraParamsResponse *depth_p, protocol::RgbCameraParamsResponse *rgb_p):
  depth(depth_p), color(rgb_p)
{
  float mx, my;
  int rx, ry;

  for (int x = 0; x < 512; x++)
    for (int y = 0; y < 424; y++) {
      undistort_depth(x,y,mx,my);
      rx = round(mx);
      ry = round(my);
      undistort_map[rx][ry][0] = x;
      undistort_map[rx][ry][1] = y;
  }

  for (int x = 0; x < 512; x++)
    for (int y = 0; y < 424; y++) {
      depth_to_color(x,y,mx,my);
      rx = round(mx);
      ry = round(my);
      depth_to_color_map[rx][ry][0] = x;
      depth_to_color_map[rx][ry][1] = y;
  }
}

} /* namespace libfreenect2 */
