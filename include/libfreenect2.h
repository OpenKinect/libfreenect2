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

#pragma once

#include <stdint.h>

/* We need struct timeval */
#ifdef _WIN32
#include <winsock.h>
#else
#include <sys/time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


/// A struct used in enumeration to give access to serial numbers, so you can
/// open a particular device by serial rather than depending on index.  This
/// is most useful if you have more than one Kinect.
struct freenect2_device_attributes;
struct freenect2_device_attributes {
	const char* camera_serial; /**< Serial number of this device's camera subdevice */
};

/// Enumeration of available resolutions.
/// Not all available resolutions are actually supported for all video formats.
/// Frame modes may not perfectly match resolutions.
typedef enum {
	FREENECT2_RESOLUTION_512x424 = 0
	FREENECT2_RESOLUTION_1920x1080 = 1
	FREENECT2_RESOLUTION_DUMMY  = 2147483647, /**< Dummy value to force enum to be 32 bits wide */
} freenect2_resolution;

/// Enumeration of video frame formats
typedef enum {
	FREENECT2_VIDEO_RGB             = 0, /**< Decompressed RGB mode */
	FREENECT2_VIDEO_YUV             = 0, /**< Decompressed YUV mode */
	FREENECT2_VIDEO_RAW             = 0, /**< Raw JPEG data mode */
	FREENECT2_VIDEO_DUMMY           = 2147483647, /**< Dummy value to force enum to be 32 bits wide */
} freenect2_video_format;

/// Enumeration of ir frame formats
typedef enum {
	FREENECT2_IR_RAW           = 5, /**< raw infrared data */
	FREENECT2_IR_DUMMY         = 2147483647, /**< Dummy value to force enum to be 32 bits wide */
} freenect2_ir_format;

/// Enumeration of depth frame formats
typedef enum {
	FREENECT2_DEPTH_MM           = 5, /**< depth to each pixel in mm, but left unaligned to RGB image */
	FREENECT2_DEPTH_DUMMY        = 2147483647, /**< Dummy value to force enum to be 32 bits wide */
} freenect2_depth_format;

/// Enumeration of flags to toggle features with freenect2_set_flag()
typedef enum {
	// arbitrary bitfields to support flag combination
	FREENECT2_MIRROR_DEPTH       = 1 << 16,
	FREENECT2_MIRROR_VIDEO       = 1 << 17,
} freenect2_flag;

/// Possible values for setting each `freenect2_flag`
typedef enum {
	FREENECT2_OFF = 0,
	FREENECT2_ON  = 1,
} freenect2_flag_value;

/// Structure to give information about the width, height, bitrate,
/// framerate, and buffer size of a frame in a particular mode, as
/// well as the total number of bytes needed to hold a single frame.
typedef struct {
	uint32_t reserved;              /**< unique ID used internally.  The meaning of values may change without notice.  Don't touch or depend on the contents of this field.  We mean it. */
	freenect2_resolution resolution; /**< Resolution this freenect2_frame_mode describes, should you want to find it again with freenect2_find_*_frame_mode(). */
	union {
		int32_t dummy;
		freenect2_video_format video_format;
		freenect2_ir_format ir_format;
		freenect2_depth_format depth_format;
	};                              /**< The video or depth format that this freenect2_frame_mode describes.  The caller should know which of video_format or depth_format to use, since they called freenect2_get_*_frame_mode() */
	int32_t bytes;                  /**< Total buffer size in bytes to hold a single frame of data.  Should be equivalent to width * height * (data_bits_per_pixel+padding_bits_per_pixel) / 8 */
	int16_t width;                  /**< Width of the frame, in pixels */
	int16_t height;                 /**< Height of the frame, in pixels */
	int8_t data_bits_per_pixel;     /**< Number of bits of information needed for each pixel */
	int8_t padding_bits_per_pixel;  /**< Number of bits of padding for alignment used for each pixel */
	int8_t framerate;               /**< Approximate expected frame rate, in Hz */
	int8_t is_valid;                /**< If 0, this freenect2_frame_mode is invalid and does not describe a supported mode.  Otherwise, the frame_mode is valid. */
} freenect2_frame_mode;

struct _freenect2_context;
typedef struct _freenect2_context freenect2_context; /**< Holds information about the usb context. */

struct _freenect2_device;
typedef struct _freenect2_device freenect2_device; /**< Holds device information. */

// usb backend specific section
typedef void freenect2_usb_context; /**< Holds libusb-1.0 context */
//

/// If Win32, export all functions for DLL usage
#ifndef _WIN32
  #define FREENECT2API /**< DLLExport information for windows, set to nothing on other platforms */
#else
  /**< DLLExport information for windows, set to nothing on other platforms */
  #ifdef __cplusplus
    #define FREENECT2API extern "C" __declspec(dllexport)
  #else
    // this is required when building from a Win32 port of gcc without being
    // forced to compile all of the library files (.c) with g++...
    #define FREENECT2API __declspec(dllexport)
  #endif
#endif

/// Enumeration of message logging levels
typedef enum {
	FREENECT2_LOG_FATAL = 0,     /**< Log for crashing/non-recoverable errors */
	FREENECT2_LOG_ERROR,         /**< Log for major errors */
	FREENECT2_LOG_WARNING,       /**< Log for warning messages */
	FREENECT2_LOG_NOTICE,        /**< Log for important messages */
	FREENECT2_LOG_INFO,          /**< Log for normal messages */
	FREENECT2_LOG_DEBUG,         /**< Log for useful development messages */
	FREENECT2_LOG_SPEW,          /**< Log for slightly less useful messages */
	FREENECT2_LOG_FLOOD,         /**< Log EVERYTHING. May slow performance. */
} freenect2_loglevel;

/**
 * Initialize a freenect2 context and do any setup required for
 * platform specific USB libraries.
 *
 * @param ctx Address of pointer to freenect2 context struct to allocate and initialize
 * @param usb_ctx USB context to initialize. Can be NULL if not using multiple contexts.
 *
 * @return 0 on success, < 0 on error
 */
FREENECT2API int freenect2_init(freenect2_context **ctx, freenect2_usb_context *usb_ctx);

/**
 * Closes the device if it is open, and frees the context
 *
 * @param ctx freenect2 context to close/free
 *
 * @return 0 on success
 */
FREENECT2API int freenect2_shutdown(freenect2_context *ctx);

/// Typedef for logging callback functions
typedef void (*freenect2_log_cb)(freenect2_context *dev, freenect2_loglevel level, const char *msg);

/**
 * Set the log level for the specified freenect2 context
 *
 * @param ctx context to set log level for
 * @param level log level to use (see freenect_loglevel enum)
 */
FREENECT2API void freenect2_set_log_level(freenect2_context *ctx, freenect2_loglevel level);

/**
 * Callback for log messages (i.e. for rerouting to a file instead of
 * stdout)
 *
 * @param ctx context to set log callback for
 * @param cb callback function pointer
 */
FREENECT2API void freenect2_set_log_callback(freenect2_context *ctx, freenect2_log_cb cb);

/**
 * Scans for kinect devices and returns the number of kinect devices currently connected to the
 * system
 *
 * @param ctx Context to access device count through
 *
 * @return Number of devices connected, < 0 on error
 */
FREENECT2API int freenect2_num_devices(freenect2_context *ctx);

/**
 * Gets the attributes of a kinect device at a given index.
 *
 * @param ctx Context to access device attributes through
 * @param index Index of the kinect device
 *
 * @return Number of devices connected, < 0 on error
 */
FREENECT2API freenect2_device_attributes freenect2_get_device_attributes(freenect2_context *ctx, int index);

/**
 * Opens a kinect device via a context. Index specifies the index of
 * the device on the current state of the bus. Bus resets may cause
 * indexes to shift.
 *
 * @param ctx Context to open device through
 * @param dev Device structure to assign opened device to
 * @param index Index of the device on the bus
 *
 * @return 0 on success, < 0 on error
 */
FREENECT2API int freenect2_open_device(freenect2_context *ctx, freenect2_device **dev, int index);

/**
 * Opens a kinect device (via a context) associated with a particular camera
 * subdevice serial number.  This function will fail if no device with a
 * matching serial number is found.
 *
 * @param ctx Context to open device through
 * @param dev Device structure to assign opened device to
 * @param camera_serial Null-terminated ASCII string containing the serial number of the camera subdevice in the device to open
 *
 * @return 0 on success, < 0 on error
 */
FREENECT2API int freenect2_open_device_by_camera_serial(freenect2_context *ctx, freenect2_device **dev, const char* camera_serial);

/**
 * Closes a device that is currently open
 *
 * @param dev Device to close
 *
 * @return 0 on success
 */
FREENECT2API int freenect2_close_device(freenect2_device *dev);

/// Typedef for depth image received event callbacks
typedef void (*freenect2_depth_cb)(freenect2_device *dev, uint32_t timestamp, void *depth, void *user);
/// Typedef for ir image received event callbacks
typedef void (*freenect2_ir_cb)(freenect2_device *dev, uint32_t timestamp, void *ir, void *user);
/// Typedef for video image received event callbacks
typedef void (*freenect2_video_cb)(freenect2_device *dev, uint32_t timestamp, void *video, void *user);

/**
 * Set callback for depth information received event
 *
 * @param dev Device to set callback for
 * @param cb Function pointer for processing depth information
 * @param user Pointer to user data
 */
FREENECT2API void freenect2_set_depth_callback(freenect2_device *dev, freenect2_depth_cb cb, void *user);

/**
 * Set callback for ir information received event
 *
 * @param dev Device to set callback for
 * @param cb Function pointer for processing depth information
 * @param user Pointer to user data
 */
FREENECT2API void freenect2_set_ir_callback(freenect2_device *dev, freenect2_ir_cb cb, void *user);

/**
 * Set callback for video information received event
 *
 * @param dev Device to set callback for
 * @param cb Function pointer for processing video information
 * @param user Pointer to user data
 */
FREENECT2API void freenect2_set_video_callback(freenect2_device *dev, freenect2_video_cb cb, void *user);

/**
 * Start the depth information stream for a device.
 *
 * @param dev Device to start depth information stream for.
 *
 * @return 0 on success, < 0 on error
 */
FREENECT2API int freenect2_start_depth(freenect2_device *dev);

/**
 * Start the ir information stream for a device.
 *
 * @param dev Device to start ir information stream for.
 *
 * @return 0 on success, < 0 on error
 */
FREENECT2API int freenect2_start_ir(freenect2_device *dev);

/**
 * Start the video information stream for a device.
 *
 * @param dev Device to start video information stream for.
 *
 * @return 0 on success, < 0 on error
 */
FREENECT2API int freenect2_start_video(freenect2_device *dev);

/**
 * Stop the depth information stream for a device
 *
 * @param dev Device to stop depth information stream on.
 *
 * @return 0 on success, < 0 on error
 */
FREENECT2API int freenect2_stop_depth(freenect2_device *dev);

/**
 * Stop the ir information stream for a device
 *
 * @param dev Device to stop ir information stream on.
 *
 * @return 0 on success, < 0 on error
 */
FREENECT2API int freenect2_stop_ir(freenect2_device *dev);

/**
 * Stop the video information stream for a device
 *
 * @param dev Device to stop video information stream on.
 *
 * @return 0 on success, < 0 on error
 */
FREENECT2API int freenect2_stop_video(freenect2_device *dev);

/**
 * Get the number of video camera modes supported by the driver.
 *
 * @return Number of video modes supported by the driver
 */
FREENECT2API int freenect2_get_video_mode_count();

/**
 * Get the frame descriptor of the nth supported video mode for the
 * video camera.
 *
 * @param mode_num Which of the supported modes to return information about
 *
 * @return A freenect2_frame_mode describing the nth video mode
 */
FREENECT2API freenect2_frame_mode freenect2_get_video_mode(int mode_num);

/**
 * Get the frame descriptor of the current video mode for the specified
 * freenect device.
 *
 * @param dev Which device to return the currently-set video mode for
 *
 * @return A freenect2_frame_mode describing the current video mode of the specified device
 */
FREENECT2API freenect2_frame_mode freenect2_get_current_video_mode(freenect2_device *dev);

/**
 * Convenience function to return a mode descriptor matching the
 * specified resolution and video camera pixel format, if one exists.
 *
 * @param res Resolution desired
 * @param fmt Pixel format desired
 *
 * @return A freenect2_frame_mode that matches the arguments specified, if such a valid mode exists; otherwise, an invalid freenect2_frame_mode.
 */
FREENECT2API freenect2_frame_mode freenect2_find_video_mode(freenect2_resolution res, freenect2_video_format fmt);

/**
 * Sets the current video mode for the specified device.  If the
 * freenect2_frame_mode specified is not one provided by the driver
 * e.g. from freenect2_get_video_mode() or freenect2_find_video_mode()
 * then behavior is undefined.  The current video mode cannot be
 * changed while streaming is active.
 *
 * @param dev Device for which to set the video mode
 * @param mode Frame mode to set
 *
 * @return 0 on success, < 0 if error
 */
FREENECT2API int freenect2_set_video_mode(freenect2_device* dev, freenect2_frame_mode mode);

/**
 * Get the number of ir camera modes supported by the driver.
 *
 * @return Number of ir modes supported by the driver
 */
FREENECT2API int freenect2_get_ir_mode_count();

/**
 * Get the frame descriptor of the nth supported ir mode for the
 * ir camera.
 *
 * @param mode_num Which of the supported modes to return information about
 *
 * @return A freenect2_frame_mode describing the nth ir mode
 */
FREENECT2API freenect2_frame_mode freenect2_get_ir_mode(int mode_num);

/**
 * Get the frame descriptor of the current ir mode for the specified
 * freenect device.
 *
 * @param dev Which device to return the currently-set ir mode for
 *
 * @return A freenect2_frame_mode describing the ir video mode of the specified device
 */
FREENECT2API freenect2_frame_mode freenect2_get_current_ir_mode(freenect2_device *dev);

/**
 * Convenience function to return a mode descriptor matching the
 * specified resolution and ir camera pixel format, if one exists.
 *
 * @param res Resolution desired
 * @param fmt Pixel format desired
 *
 * @return A freenect2_frame_mode that matches the arguments specified, if such a valid mode exists; otherwise, an invalid freenect2_frame_mode.
 */
FREENECT2API freenect2_frame_mode freenect2_find_ir_mode(freenect2_resolution res, freenect2_ir_format fmt);

/**
 * Sets the current ir mode for the specified device.  If the
 * freenect2_frame_mode specified is not one provided by the driver
 * e.g. from freenect2_get_ir_mode() or freenect2_find_ir_mode()
 * then behavior is undefined.  The current ir mode cannot be
 * changed while streaming is active.
 *
 * @param dev Device for which to set the ir mode
 * @param mode Frame mode to set
 *
 * @return 0 on success, < 0 if error
 */
FREENECT2API int freenect2_set_ir_mode(freenect2_device* dev, freenect2_frame_mode mode);

/**
 * Get the number of depth camera modes supported by the driver.
 *
 * @return Number of depth modes supported by the driver
 */
FREENECT2API int freenect2_get_depth_mode_count();

/**
 * Get the frame descriptor of the nth supported depth mode for the
 * depth camera.
 *
 * @param mode_num Which of the supported modes to return information about
 *
 * @return A freenect2_frame_mode describing the nth depth mode
 */
FREENECT2API freenect2_frame_mode freenect2_get_depth_mode(int mode_num);

/**
 * Get the frame descriptor of the current depth mode for the specified
 * freenect2 device.
 *
 * @param dev Which device to return the currently-set depth mode for
 *
 * @return A freenect2_frame_mode describing the current depth mode of the specified device
 */
FREENECT2API freenect2_frame_mode freenect2_get_current_depth_mode(freenect2_device *dev);

/**
 * Convenience function to return a mode descriptor matching the
 * specified resolution and depth camera pixel format, if one exists.
 *
 * @param res Resolution desired
 * @param fmt Pixel format desired
 *
 * @return A freenect2_frame_mode that matches the arguments specified, if such a valid mode exists; otherwise, an invalid freenect2_frame_mode.
 */
FREENECT2API freenect2_frame_mode freenect2_find_depth_mode(freenect2_resolution res, freenect2_depth_format fmt);

/**
 * Sets the current depth mode for the specified device. The mode
 * cannot be changed while streaming is active.
 *
 * @param dev Device for which to set the depth mode
 * @param mode Frame mode to set
 *
 * @return 0 on success, < 0 if error
 */
FREENECT2API int freenect2_set_depth_mode(freenect2_device* dev, const freenect2_frame_mode mode);

/**
 * Enables or disables the specified flag.
 * 
 * @param flag Feature to set
 * @param value `FREENECT2_OFF` or `FREENECT2_ON`
 * 
 * @return 0 on success, < 0 if error
 */
FREENECT2API int freenect2_set_flag(freenect2_device *dev, freenect2_flag flag, freenect2_flag_value value);

#ifdef __cplusplus
}
#endif
