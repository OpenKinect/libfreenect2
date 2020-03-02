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
#include <limits>

#define MONO_ROWS 424
#define MONO_COLS 512
#define COLOR_ROWS 1080
#define COLOR_COLS 1920


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
void dInitMaps(int* d_map_dist, float* d_map_x, float* d_map_y, int* d_map_yi,
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

static __global__
void setFloat(float* devPtr, float value)
{
	// Configuration copied from cuda_depth_packet_processor.cu
	const uint i = blockIdx.x * blockDim.x + threadIdx.x;

	devPtr[i] = value;
}

/**
 * Set all values of array of floats devPtr to value.
 * This function does not call for synchronization.
 * @param devPtr pointer to memory in device
 * @param value value to set
 * @param size number of float sized elements in array
 */
void cudaMemsetFloat(float* devPtr, float value, size_t size)
{
	size_t numThreads = 512;
	size_t numBlocks = size / numThreads;
	setFloat<<<numBlocks, numThreads>>>(devPtr, value);
}

/**
 * Compares value at address with val, if val is smaller it
 * saves it at address.
 */
__device__ float atomicKeepSmaller(float* address, float val)
{
	// Implementation addapted from http://docs.nvidia.com/cuda/cuda-c-programming-guide/#atomic-functions
	int* address_as_ull = (int*)address;
	int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
				__float_as_int(val < __int_as_float(assumed) ? val : __int_as_float(assumed)));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);

	return __int_as_float(old);
}


static __global__
void undistort(int* d_depth_to_c_off,
		float* d_undistorted_data,
		float* d_filter_map,
		const float* d_depth_data, const int* d_map_dist,
		const float* d_map_x, const int* d_map_yi,
		const libfreenect2::Freenect2Device::IrCameraParams depth,
		const libfreenect2::Freenect2Device::ColorCameraParams color,
		const int filter_width_half,
		const int filter_height_half,
		const int offset_filter_map,
		const bool enable_filter)
{
	// getting index of distorted depth pixel
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int index = d_map_dist[i];

	// check if distorted depth pixel is outside of the depth image
	if(index < 0){
		d_depth_to_c_off[i] = -1;
		d_undistorted_data[i] = 0;
		return;
	}

	// getting depth value for current pixel
	const float z = d_depth_data[index];
	d_undistorted_data[i] = z;

	// checking for invalid depth value
	if(z <= 0.0f){
		d_depth_to_c_off[i] = -1;
		return;
	}

	// calculating x offset for rgb image based on depth value
	const float color_cx = color.cx + 0.5f; // 0.5f added for later rounding
	const float rx = (d_map_x[index] + (color.shift_m / z)) * color.fx + color_cx;
	const int cx = rx; // same as round for positive numbers (0.5f was already added to color_cx)
	// getting y offset for depth image
	const int cy = d_map_yi[i];
	// combining offsets
	const int c_off = cx + cy * COLOR_COLS;

	// check if c_off is outside of rgb image
	// checking rx/cx is not needed because the color image is much wider then the depth image
	if(c_off < 0 || c_off >= COLOR_ROWS * COLOR_COLS){
		d_depth_to_c_off[i] = -1;
		return;
	}

	// saving the offset for later
	d_depth_to_c_off[i] = c_off;

	// I am not sure if there won't be race conditions here due to overlap, the atomic operation should help.
	if(enable_filter){
		// setting a window around the filter map pixel corresponding to the color pixel with the current z value
		int yi = (cy - filter_height_half) * 1920 + cx - filter_width_half; // index of first pixel to set
		for(int r = -filter_height_half; r <= filter_height_half; ++r, yi += COLOR_COLS) // index increased by a full row each iteration
		{
			float *it = d_filter_map + offset_filter_map + yi;
			for(int c = -filter_width_half; c <= filter_width_half; ++c, ++it)
	        {
				// only set if the current z is smaller
				atomicKeepSmaller(it, z);
	        }
	    }
	}
}

/** Construct 'registered' image with filter.
 *  Filter drops duplicate pixels due to aspect of two cameras.
 */
static __global__
void registerImageFiltered(unsigned int *d_registered_data,
		const unsigned int * d_rgb_data,
		const int* d_depth_to_c_off,
		const float* d_undistorted_data,
		const float *d_p_filter_map,
		const float filter_tolerance)
{
	// getting index of distorted depth pixel
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	// run through all registered color pixels and set them based on filter results
	const int c_off = d_depth_to_c_off[i];

	// check if offset is out of image
	if(c_off < 0){
		d_registered_data[i] = 0;
		return;
	}

	const float min_z = d_p_filter_map[c_off];
	const float z = d_undistorted_data[i];

	// check for allowed depth noise
	d_registered_data[i] = (z - min_z) / z > filter_tolerance ? 0 : d_rgb_data[c_off];

}

/** Construct 'registered' image. */
static __global__
void registerImage(unsigned int *d_registered_data,
		const unsigned int * d_rgb_data,
		const int* d_depth_to_c_off)
{
	// getting index of distorted depth pixel
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	// run through all registered color pixels and set them based on c_off
    const int c_off = d_depth_to_c_off[i];

    // check if offset is out of image
    d_registered_data[i] = c_off < 0 ? 0 : d_rgb_data[c_off];
}

namespace libfreenect2
{

CudaDeviceFrame::CudaDeviceFrame(size_t width, size_t height, size_t bytes_per_pixel):
	Frame(width, height, bytes_per_pixel, (unsigned char*)-1)
{
	allocateMemory();
}

CudaDeviceFrame::~CudaDeviceFrame()
{
	CALL_CUDA(cudaFree(data));
}

bool CudaDeviceFrame::allocateMemory()
{
	CHECK_CUDA(cudaMalloc(&data, width * height * bytes_per_pixel * sizeof(unsigned char)));

	cudaDeviceSynchronize();

	CHECK_CUDA(cudaGetLastError());
	return true;
}

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
	bool apply(const Frame* rgb, const Frame* depth, CudaDeviceFrame* undistorted, CudaDeviceFrame* registered, const bool enable_filter, CudaDeviceFrame* bigdepth, int* color_depth_map) const;
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
	int* d_depth_to_color_map_yi;

	bool allocateDeviceMemory()
	{
		CHECK_CUDA(cudaMalloc(&d_distort_map, MONO_IMAGE_SIZE * sizeof(int)));
		CHECK_CUDA(cudaMalloc(&d_depth_to_color_map_x, MONO_IMAGE_SIZE * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&d_depth_to_color_map_y, MONO_IMAGE_SIZE * sizeof(float)));
		CHECK_CUDA(cudaMalloc(&d_depth_to_color_map_yi, MONO_IMAGE_SIZE * sizeof(int)));

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

bool CudaRegistration::apply(const Frame* rgb, const Frame* depth, CudaDeviceFrame* undistorted, CudaDeviceFrame* registered, const bool enable_filter, CudaDeviceFrame* bigdepth, int* color_depth_map) const
{
	return impl_->apply(rgb, depth, undistorted, registered, enable_filter, bigdepth, color_depth_map);
}

bool CudaRegistrationImpl::apply(const Frame *rgb, const Frame *depth, CudaDeviceFrame *undistorted, CudaDeviceFrame *registered, const bool enable_filter, CudaDeviceFrame *bigdepth, int *color_depth_map) const
{
	// Check if all frames are valid and have the correct size
	if (!rgb || !depth || !undistorted || !registered ||
		rgb->width != 1920 || rgb->height != 1080 || rgb->bytes_per_pixel != 4 ||
		depth->width != 512 || depth->height != 424 || depth->bytes_per_pixel != 4 ||
		undistorted->width != 512 || undistorted->height != 424 || undistorted->bytes_per_pixel != 4 ||
		registered->width != 512 || registered->height != 424 || registered->bytes_per_pixel != 4)
	{
		LOG_ERROR << "Not applying" << std::endl;
		return false;
	}

	// Setup memory

	float *d_depth_data;
	size_t depth_size = depth->width * depth->height * sizeof(float);
	unsigned int *d_rgb_data;
	size_t rgb_size = rgb->width * rgb->height * sizeof(unsigned int);

	CHECK_CUDA(cudaMalloc(&d_depth_data, depth_size));
	cudaMemcpy((void*)d_depth_data,
			   (const void*)depth->data, depth_size,
			   cudaMemcpyHostToDevice);

	CHECK_CUDA(cudaMalloc(&d_rgb_data, rgb_size));
	cudaMemcpy((void*)d_rgb_data,
			   (const void*)rgb->data, rgb_size,
			   cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	CHECK_CUDA(cudaGetLastError());

	float *d_undistorted_data = (float*)undistorted->data;
	unsigned int *d_registered_data = (unsigned int*)registered->data;
	const int *d_map_dist = d_distort_map;
	const float *d_map_x = d_depth_to_color_map_x;
	const int *d_map_yi = d_depth_to_color_map_yi;


	// Setup parameters

	const int size_depth = MONO_ROWS * MONO_COLS;
	const int size_color = COLOR_ROWS * COLOR_COLS;
	//const float color_cx = color.cx + 0.5f; // 0.5f added for later rounding

	// size of filter map with a border of filter_height_half on top and bottom so that no check for borders is needed.
	// since the color image is wide angle no border to the sides is needed.
	const int size_filter_map = size_color + COLOR_COLS * filter_height_half * 2;
	// offset to the important data
	const int offset_filter_map = COLOR_COLS * filter_height_half;


	// Auxiliary maps

	// map for storing the min z values used for each color pixel
	float *d_filter_map = NULL;
	// pointer to the beginning of the important data
	float *d_p_filter_map = NULL;

	// map for storing the color offset for each depth pixel
	int *d_depth_to_c_off;
	CHECK_CUDA(cudaMalloc(&d_depth_to_c_off, size_depth * sizeof(int)));
	if (color_depth_map)
	{
		// I don't know where this other color map could be coming from,
		// so for the moment I will assume it is in host memory.
		cudaMemcpy((void*)d_depth_to_c_off,
					   (const void*)color_depth_map, size_depth * sizeof(int),
					   cudaMemcpyHostToDevice);
	}
	//int *map_c_off = depth_to_c_off;

	// initializing the depth_map with values outside of the Kinect2 range
	if(enable_filter){
		if(bigdepth)
		{
			d_filter_map = (float*)bigdepth->data;
		}
		else
		{
			CHECK_CUDA(cudaMalloc(&d_filter_map, size_filter_map * sizeof(float)));
		}
	    d_p_filter_map = d_filter_map + offset_filter_map;		// works the same even on device

	    cudaMemsetFloat(d_filter_map, std::numeric_limits<float>::infinity(), size_filter_map);
	}

	/* Fix depth distortion, and compute pixel to use from 'rgb' based on depth measurement,
	 * stored as x/y offset in the rgb data.
	 */
	undistort<<<size_depth/MONO_COLS, MONO_COLS>>>(d_depth_to_c_off,
			d_undistorted_data, d_filter_map,
			d_depth_data, d_map_dist,
			d_map_x, d_map_yi,
			this->depth, this->color, filter_width_half, filter_height_half, offset_filter_map, enable_filter);
	if (enable_filter)
	{
		registerImageFiltered<<<size_depth/MONO_COLS, MONO_COLS>>>(d_registered_data,
				d_rgb_data,
				d_depth_to_c_off,
				d_undistorted_data,
				d_p_filter_map,
				filter_tolerance);
		if (!bigdepth)
		{
			CALL_CUDA(cudaFree(d_filter_map));
		}
	}
	else
	{
		registerImage<<<size_depth/MONO_COLS, MONO_COLS>>>(d_registered_data,
				d_rgb_data,
				d_depth_to_c_off);
	}

	// Finish

	// -1 represents Invalid
	//undistorted->format = undistorted->Float;
	//registered->format = registered->BGRX;


	if (color_depth_map)
	{
		// I don't know where this other color map could be coming from,
		// so for the moment I will assume it is in host memory.
		// Placing it back to where it came from
		cudaMemcpy((void*)color_depth_map,
					   (const void*)d_depth_to_c_off, size_depth * sizeof(int),
					   cudaMemcpyDeviceToHost);
	}
	CALL_CUDA(cudaFree(d_depth_to_c_off));

	CALL_CUDA(cudaFree(d_depth_data));
	CALL_CUDA(cudaFree(d_rgb_data));

	return true;
}

} /* namespace libfreenect2 */
