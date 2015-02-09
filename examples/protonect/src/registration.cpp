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
  float qs = ((ps * depth_k3 + depth_k2) * ps + depth_k1) * ps + 1.0;
  for (int i = 0; i < 9; i++) {
    float qd = ps / (qs * qs);
    qs = ((qd * depth_k3 + depth_k2) * qd + depth_k1) * qd + 1.0;
  }
  mx = dx / qs;
  my = dy / qs;
}

/*void kinect2_depth_to_color(float mx, float my, float z, float& rx, float& ry)
{
    mx *= depth_f * depth_q;
    my *= depth_f * depth_q;

    float wx =
        (mx * mx * mx * mx_x3y0) + (my * my * my * mx_x0y3) +
        (mx * mx * my * mx_x2y1) + (my * my * mx * mx_x1y2) +
        (mx * mx * mx_x2y0) + (my * my * mx_x0y2) + (mx * my * mx_x1y1) +
        (mx * mx_x1y0) + (my * mx_x0y1) + (mx_x0y0);
    float wy =
        (mx * mx * mx * my_x3y0) + (my * my * my * my_x0y3) +
        (mx * mx * my * my_x2y1) + (my * my * mx * my_x1y2) +
        (mx * mx * my_x2y0) + (my * my * my_x0y2) + (mx * my * my_x1y1) +
        (mx * my_x1y0) + (my * my_x0y1) + (my_x0y0);

    rx = wx / (color_f * color_q);
    ry = wy / (color_f * color_q);

    rx += (shift_m / z) - (shift_m / shift_d);

    rx = rx * color_f + color_cx;
    ry = ry * color_f + color_cy;
}*/

Registration::Registration(protocol::DepthCameraParamsResponse *depth_p, protocol::RgbCameraParamsResponse *rgb_p)
{
  float mx, my;
  int rx, ry;

  depth_k1 = depth_p->k1;
  depth_k2 = depth_p->k2;
  depth_k3 = depth_p->k3;

  for (int x = 0; x < 512; x++)
    for (int y = 0; y < 424; y++) {
      undistort_depth(x,y,mx,my);
      rx = round(mx);
      ry = round(my);
      undistort_map[rx][ry][0] = x;
      undistort_map[rx][ry][1] = y;
  }
}

} /* namespace libfreenect2 */