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

/** @file Implementation of merging depth and color images using cuda. */

#include <libfreenect2/cuda_registration.h>
#include "libfreenect2/logging.h"

#define MONO_ROWS 424
#define MONO_COLS 512


typedef unsigned char uchar;

#define CHECK_CUDA(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { LOG_ERROR << #expr ": " << cudaGetErrorString(err); return false; } } while(0)
#define CALL_CUDA(expr) do { cudaError_t err = (expr); if (err != cudaSuccess) { LOG_ERROR << #expr ": " << cudaGetErrorString(err); } } while(0)

static __device__
void distort(int mx, int my, float& d_x, float& d_y, const libfreenect2::Freenect2Device::IrCameraParams& d_depth)
{
	float dx = ((float)mx - d_depth.cx) / d_depth.fx;
	float dy = ((float)my - d_depth.cy) / d_depth.fy;
	float dx2 = dx * dx;
	float dy2 = dy * dy;
	float r2 = dx2 + dy2;
	float dxdy2 = 2 * dx * dy;
	float kr = 1 + ((d_depth.k3 * r2 + d_depth.k2) * r2 + d_depth.k1) * r2;
	d_x = d_depth.fx * (dx * kr + d_depth.p2 * (r2 + 2 * dx2) + d_depth.p1 * dxdy2) + d_depth.cx;
	d_y = d_depth.fy * (dy * kr + d_depth.p1 * (r2 + 2 * dy2) + d_depth.p2 * dxdy2) + d_depth.cy;
}

static __device__
void depth_to_color(float mx, float my, float& d_rx, float& d_ry,
		const libfreenect2::Freenect2Device::IrCameraParams& d_depth,
		const libfreenect2::Freenect2Device::ColorCameraParams& d_color,
		const float depth_q, const float color_q)
{
	mx = (mx - d_depth.cx) * depth_q;
	my = (my - d_depth.cy) * depth_q;

	float wx =
		(mx * mx * mx * d_color.mx_x3y0) + (my * my * my * d_color.mx_x0y3) +
		(mx * mx * my * d_color.mx_x2y1) + (my * my * mx * d_color.mx_x1y2) +
		(mx * mx * d_color.mx_x2y0) + (my * my * d_color.mx_x0y2) + (mx * my * d_color.mx_x1y1) +
		(mx * d_color.mx_x1y0) + (my * d_color.mx_x0y1) + (d_color.mx_x0y0);

	float wy =
		(mx * mx * mx * d_color.my_x3y0) + (my * my * my * d_color.my_x0y3) +
		(mx * mx * my * d_color.my_x2y1) + (my * my * mx * d_color.my_x1y2) +
		(mx * mx * d_color.my_x2y0) + (my * my * d_color.my_x0y2) + (mx * my * d_color.my_x1y1) +
		(mx * d_color.my_x1y0) + (my * d_color.my_x0y1) + (d_color.my_x0y0);

	d_rx = (wx / (d_color.fx * color_q)) - (d_color.shift_m / d_color.shift_d);
	d_ry = (wy / color_q) + d_color.cy;
}

static __global__
void dInitMaps(int* d_map_dist, float* d_map_x, float* d_map_y, float* d_map_yi,
		const libfreenect2::Freenect2Device::IrCameraParams d_depth,
		const libfreenect2::Freenect2Device::ColorCameraParams d_color,
		const float depth_q, const float color_q)
{
	// Configuration copied from cuda_depth_packet_processor.cu
	const uint i = blockIdx.x*blockDim.x + threadIdx.x;

	const uint x = i % MONO_COLS;
	const uint y = i / MONO_COLS;

	float mx, my;
	int ix, iy, index;
	float rx, ry;

	// compute the distorted coordinate for current pixel
	distort(x, y, mx, my, d_depth);

	// rounding the values and check if the pixel is inside the image
	ix = (int)(mx + 0.5f);
	iy = (int)(my + 0.5f);
	if(ix < 0 || ix >= 512 || iy < 0 || iy >= 424)
		index = -1;
	else
		// computing the index from the coordinates for faster access to the data
		index = iy * 512 + ix;
	d_map_dist[i] = index;

	// compute the depth to color mapping entries for the current pixel
	depth_to_color(x, y, rx, ry, d_depth, d_color, depth_q, color_q);
	d_map_x[i] = rx;
	d_map_y[i] = ry;
	// compute the y offset to minimize later computations
	d_map_yi[i] = (int)(ry + 0.5f);
}

namespace libfreenect2
{

/*
 * The information used here has been taken from libfreenect2::Registration source
 * code.
 */
static const float depth_q = 0.01;
static const float color_q = 0.002199;

class CudaRegistrationImpl
{
public:
	CudaRegistrationImpl(Freenect2Device::IrCameraParams depth_p, Freenect2Device::ColorCameraParams rgb_p):
		depth(depth_p), color(rgb_p),
		filter_width_half(2), filter_height_half(1), filter_tolerance(0.01f),
		block_size(128), grid_size(MONO_IMAGE_SIZE/block_size)
	{
		good = setupDevice();
		if (!good)
			return;

		good = initMaps();
		if (!good)
			return;
	}

	~CudaRegistrationImpl()
	{
		if (good)
			freeDeviceMemory();
	}

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

	const int filter_width_half;
	const int filter_height_half;
	const float filter_tolerance;

	static const size_t MONO_IMAGE_SIZE = MONO_COLS * MONO_ROWS;

	size_t block_size;
	size_t grid_size;

	bool good; 								// Memory correctly allocated

	// Maps
	int* d_distort_map;
	float* d_depth_to_color_map_x;
	float* d_depth_to_color_map_y;
	float* d_depth_to_color_map_yi;

	bool allocateDeviceMemory()
	{
		CHECK_CUDA(cudaMalloc(&d_distort_map, MONO_IMAGE_SIZE * sizeof(int)));
		CHECK_CUDA(cudaMalloc(&d_depth_to_color_map_x, MONO_IMAGE_SIZE * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&d_depth_to_color_map_y, MONO_IMAGE_SIZE * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&d_depth_to_color_map_yi, MONO_IMAGE_SIZE * sizeof(float)));

		cudaDeviceSynchronize();

		CHECK_CUDA(cudaGetLastError());
		return true;
	}

	bool setupDevice()
	{
		// Continue to use same device than cuda_depth_packet_processor?
		if (!allocateDeviceMemory())
		  return false;

		return true;
	}

	bool initMaps()
	{
		dInitMaps<<<grid_size, block_size>>>(d_distort_map, d_depth_to_color_map_x,
						d_depth_to_color_map_y, d_depth_to_color_map_yi,
						depth, color, depth_q, color_q);

		cudaDeviceSynchronize();
		CHECK_CUDA(cudaGetLastError());

		return true;
	}

	void freeDeviceMemory()
	{
		CALL_CUDA(cudaFree(d_distort_map));
		CALL_CUDA(cudaFree(d_depth_to_color_map_x));
		CALL_CUDA(cudaFree(d_depth_to_color_map_y));
		CALL_CUDA(cudaFree(d_depth_to_color_map_yi));
	}
};

CudaRegistration::CudaRegistration(Freenect2Device::IrCameraParams depth_p, Freenect2Device::ColorCameraParams rgb_p):
  impl_(new CudaRegistrationImpl(depth_p, rgb_p)) {}

CudaRegistration::~CudaRegistration()
{
  delete impl_;
}

void CudaRegistration::apply(const Frame* rgb, const Frame* depth, Frame* undistorted, Frame* registered, const bool enable_filter, Frame* bigdepth, int* color_depth_map) const
{
	impl_->apply(rgb, depth, undistorted, registered, enable_filter, bigdepth, color_depth_map);
}

void CudaRegistrationImpl::apply(const Frame *rgb, const Frame *depth, Frame *undistorted, Frame *registered, const bool enable_filter, Frame *bigdepth, int *color_depth_map) const
{
	// Check if all frames are valid and have the correct size
	if (!rgb || !depth || !undistorted || !registered ||
		rgb->width != 1920 || rgb->height != 1080 || rgb->bytes_per_pixel != 4 ||
		depth->width != 512 || depth->height != 424 || depth->bytes_per_pixel != 4 ||
		undistorted->width != 512 || undistorted->height != 424 || undistorted->bytes_per_pixel != 4 ||
		registered->width != 512 || registered->height != 424 || registered->bytes_per_pixel != 4)
	{
		LOG_ERROR << "Not applying" << std::endl;
		return;
	}
}

} /* namespace libfreenect2 */
