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

/** @file Implementation of merging depth and color images. */

#define _USE_MATH_DEFINES
#include <math.h>
#include <libfreenect2/registration.h>
#include <limits>

namespace libfreenect2
{

/*
 * most information, including the table layout in command_response.h, was
 * provided by @sh0 in https://github.com/OpenKinect/libfreenect2/issues/41
 */

// these seem to be hardcoded in the original SDK
static const float depth_q = 0.01;
static const float color_q = 0.002199;

class RegistrationImpl
{
public:
  RegistrationImpl(Freenect2Device::IrCameraParams depth_p, Freenect2Device::ColorCameraParams rgb_p);

  void apply(int dx, int dy, float dz, float& cx, float &cy) const;
  void apply(const Frame* rgb, const Frame* depth, Frame* undistorted, Frame* registered, const bool enable_filter, Frame* bigdepth, int* color_depth_map) const;
  void undistortDepth(const Frame *depth, Frame *undistorted) const;
  void getPointXYZRGB (const Frame* undistorted, const Frame* registered, int r, int c, float& x, float& y, float& z, float& rgb) const;
  void getPointXYZ (const Frame* undistorted, int r, int c, float& x, float& y, float& z) const;
  void distort(int mx, int my, float& dx, float& dy) const;
  void depth_to_color(float mx, float my, float& rx, float& ry) const;

private:
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

void RegistrationImpl::distort(int mx, int my, float& x, float& y) const
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

void RegistrationImpl::depth_to_color(float mx, float my, float& rx, float& ry) const
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
  impl_->apply(dx, dy, dz, cx, cy);
}

void RegistrationImpl::apply( int dx, int dy, float dz, float& cx, float &cy) const
{
  const int index = dx + dy * 512;
  float rx = depth_to_color_map_x[index];
  cy = depth_to_color_map_y[index];

  rx += (color.shift_m / dz);
  cx = rx * color.fx + color.cx;
}

void Registration::apply(const Frame *rgb, const Frame *depth, Frame *undistorted, Frame *registered, const bool enable_filter, Frame *bigdepth, int *color_depth_map) const
{
  impl_->apply(rgb, depth, undistorted, registered, enable_filter, bigdepth, color_depth_map);
}

void RegistrationImpl::apply(const Frame *rgb, const Frame *depth, Frame *undistorted, Frame *registered, const bool enable_filter, Frame *bigdepth, int *color_depth_map) const
{
  // Check if all frames are valid and have the correct size
  if (!rgb || !depth || !undistorted || !registered ||
      rgb->width != 1920 || rgb->height != 1080 || rgb->bytes_per_pixel != 4 ||
      depth->width != 512 || depth->height != 424 || depth->bytes_per_pixel != 4 ||
      undistorted->width != 512 || undistorted->height != 424 || undistorted->bytes_per_pixel != 4 ||
      registered->width != 512 || registered->height != 424 || registered->bytes_per_pixel != 4)
    return;

  const float *depth_data = (float*)depth->data;
  const unsigned int *rgb_data = (unsigned int*)rgb->data;
  float *undistorted_data = (float*)undistorted->data;
  unsigned int *registered_data = (unsigned int*)registered->data;
  const int *map_dist = distort_map;
  const float *map_x = depth_to_color_map_x;
  const int *map_yi = depth_to_color_map_yi;

  const int size_depth = 512 * 424;
  const int size_color = 1920 * 1080;
  const float color_cx = color.cx + 0.5f; // 0.5f added for later rounding

  // size of filter map with a border of filter_height_half on top and bottom so that no check for borders is needed.
  // since the color image is wide angle no border to the sides is needed.
  const int size_filter_map = size_color + 1920 * filter_height_half * 2;
  // offset to the important data
  const int offset_filter_map = 1920 * filter_height_half;

  // map for storing the min z values used for each color pixel
  float *filter_map = NULL;
  // pointer to the beginning of the important data
  float *p_filter_map = NULL;

  // map for storing the color offset for each depth pixel
  int *depth_to_c_off = color_depth_map ? color_depth_map : new int[size_depth];
  int *map_c_off = depth_to_c_off;

  // initializing the depth_map with values outside of the Kinect2 range
  if(enable_filter){
    filter_map = bigdepth ? (float*)bigdepth->data : new float[size_filter_map];
    p_filter_map = filter_map + offset_filter_map;

    for(float *it = filter_map, *end = filter_map + size_filter_map; it != end; ++it){
      *it = std::numeric_limits<float>::infinity();
    }
  }

  /* Fix depth distortion, and compute pixel to use from 'rgb' based on depth measurement,
   * stored as x/y offset in the rgb data.
   */

  // iterating over all pixels from undistorted depth and registered color image
  // the four maps have the same structure as the images, so their pointers are increased each iteration as well
  for(int i = 0; i < size_depth; ++i, ++undistorted_data, ++map_dist, ++map_x, ++map_yi, ++map_c_off){
    // getting index of distorted depth pixel
    const int index = *map_dist;

    // check if distorted depth pixel is outside of the depth image
    if(index < 0){
      *map_c_off = -1;
      *undistorted_data = 0;
      continue;
    }

    // getting depth value for current pixel
    const float z = depth_data[index];
    *undistorted_data = z;

    // checking for invalid depth value
    if(z <= 0.0f){
      *map_c_off = -1;
      continue;
    }

    // calculating x offset for rgb image based on depth value
    const float rx = (*map_x + (color.shift_m / z)) * color.fx + color_cx;
    const int cx = rx; // same as round for positive numbers (0.5f was already added to color_cx)
    // getting y offset for depth image
    const int cy = *map_yi;
    // combining offsets
    const int c_off = cx + cy * 1920;

    // check if c_off is outside of rgb image
    // checking rx/cx is not needed because the color image is much wider then the depth image
    if(c_off < 0 || c_off >= size_color){
      *map_c_off = -1;
      continue;
    }

    // saving the offset for later
    *map_c_off = c_off;

    if(enable_filter){
      // setting a window around the filter map pixel corresponding to the color pixel with the current z value
      int yi = (cy - filter_height_half) * 1920 + cx - filter_width_half; // index of first pixel to set
      for(int r = -filter_height_half; r <= filter_height_half; ++r, yi += 1920) // index increased by a full row each iteration
      {
        float *it = p_filter_map + yi;
        for(int c = -filter_width_half; c <= filter_width_half; ++c, ++it)
        {
          // only set if the current z is smaller
          if(z < *it)
            *it = z;
        }
      }
    }
  }

  /* Construct 'registered' image. */

  // reseting the pointers to the beginning
  map_c_off = depth_to_c_off;
  undistorted_data = (float*)undistorted->data;

  /* Filter drops duplicate pixels due to aspect of two cameras. */
  if(enable_filter){
    // run through all registered color pixels and set them based on filter results
    for(int i = 0; i < size_depth; ++i, ++map_c_off, ++undistorted_data, ++registered_data){
      const int c_off = *map_c_off;

      // check if offset is out of image
      if(c_off < 0){
        *registered_data = 0;
        continue;
      }

      const float min_z = p_filter_map[c_off];
      const float z = *undistorted_data;

      // check for allowed depth noise
      *registered_data = (z - min_z) / z > filter_tolerance ? 0 : *(rgb_data + c_off);
    }

    if (!bigdepth) delete[] filter_map;
  }
  else
  {
    // run through all registered color pixels and set them based on c_off
    for(int i = 0; i < size_depth; ++i, ++map_c_off, ++registered_data){
      const int c_off = *map_c_off;

      // check if offset is out of image
      *registered_data = c_off < 0 ? 0 : *(rgb_data + c_off);
    }
  }
  if (!color_depth_map) delete[] depth_to_c_off;
}

void Registration::undistortDepth(const Frame *depth, Frame *undistorted) const
{
  impl_->undistortDepth(depth, undistorted);
}

void RegistrationImpl::undistortDepth(const Frame *depth, Frame *undistorted) const
{
  // Check if all frames are valid and have the correct size
  if (!depth || !undistorted ||
      depth->width != 512 || depth->height != 424 || depth->bytes_per_pixel != 4 ||
      undistorted->width != 512 || undistorted->height != 424 || undistorted->bytes_per_pixel != 4)
    return;

  const float *depth_data = (float*)depth->data;
  float *undistorted_data = (float*)undistorted->data;
  const int *map_dist = distort_map;

  const int size_depth = 512 * 424;

  /* Fix depth distortion, and compute pixel to use from 'rgb' based on depth measurement,
   * stored as x/y offset in the rgb data.
   */

  // iterating over all pixels from undistorted depth and registered color image
  // the four maps have the same structure as the images, so their pointers are increased each iteration as well
  for(int i = 0; i < size_depth; ++i, ++undistorted_data, ++map_dist){
    // getting index of distorted depth pixel
    const int index = *map_dist;

    // check if distorted depth pixel is outside of the depth image
    if(index < 0){
      *undistorted_data = 0;
      continue;
    }

    // getting depth value for current pixel
    const float z = depth_data[index];
    *undistorted_data = z;
  }
}

void Registration::getPointXYZRGB (const Frame* undistorted, const Frame* registered, int r, int c, float& x, float& y, float& z, float& rgb) const
{
  impl_->getPointXYZRGB(undistorted, registered, r, c, x, y, z, rgb);
}

void RegistrationImpl::getPointXYZRGB (const Frame* undistorted, const Frame* registered, int r, int c, float& x, float& y, float& z, float& rgb) const
{
  getPointXYZ(undistorted, r, c, x, y, z);

  if(isnan(z))
  {
    rgb = 0;
  }
  else
  {
    float* registered_data = (float *)registered->data;
    rgb = registered_data[512*r+c];
  }
}

void Registration::getPointXYZ(const Frame *undistorted, int r, int c, float &x, float &y, float &z) const
{
  impl_->getPointXYZ(undistorted,r,c,x,y,z);
}

void RegistrationImpl::getPointXYZ (const Frame *undistorted, int r, int c, float &x, float &y, float &z) const
{
  const float bad_point = std::numeric_limits<float>::quiet_NaN();
  const float cx(depth.cx), cy(depth.cy);
  const float fx(1/depth.fx), fy(1/depth.fy);
  float* undistorted_data = (float *)undistorted->data;
  const float depth_val = undistorted_data[512*r+c]/1000.0f; //scaling factor, so that value of 1 is one meter.
  if (isnan(depth_val) || depth_val <= 0.001)
  {
    //depth value is not valid
    x = y = z = bad_point;
  }
  else
  {
    x = (c + 0.5 - cx) * fx * depth_val;
    y = (r + 0.5 - cy) * fy * depth_val;
    z = depth_val;
  }
}

Registration::Registration(Freenect2Device::IrCameraParams depth_p, Freenect2Device::ColorCameraParams rgb_p):
  impl_(new RegistrationImpl(depth_p, rgb_p)) {}

Registration::~Registration()
{
  delete impl_;
}

RegistrationImpl::RegistrationImpl(Freenect2Device::IrCameraParams depth_p, Freenect2Device::ColorCameraParams rgb_p):
  depth(depth_p), color(rgb_p), filter_width_half(2), filter_height_half(1), filter_tolerance(0.01f)
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
      ix = (int)(mx + 0.5f);
      iy = (int)(my + 0.5f);
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
      *map_yi++ = (int)(ry + 0.5f);
    }
  }
}

} /* namespace libfreenect2 */
