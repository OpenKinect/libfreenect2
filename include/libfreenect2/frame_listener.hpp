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

/** @file frame_listener.hpp Classes for frame listeners. */

#ifndef FRAME_LISTENER_HPP_
#define FRAME_LISTENER_HPP_

#include <cstddef>
#include <stdint.h>
#include <libfreenect2/config.h>

namespace libfreenect2
{

/** @defgroup frame Frame Listeners
 * Receive decoded image frames, and the frame format.
 */

/** Frame format and metadata. @ingroup frame */
class LIBFREENECT2_API Frame
{
  public:
  /** Available types of frames. */
  enum Type
  {
    Color = 1, ///< 1920x1080. BGRX or RGBX.
    Ir = 2,    ///< 512x424 float. Range is [0.0, 65535.0].
    Depth = 4  ///< 512x424 float, unit: millimeter. Non-positive, NaN, and infinity are invalid or missing data.
  };

  /** Pixel format. */
  enum Format
  {
    Invalid = 0, ///< Invalid format.
    Raw = 1, ///< Raw bitstream. 'bytes_per_pixel' defines the number of bytes
    Float = 2, ///< A 4-byte float per pixel
    BGRX = 4, ///< 4 bytes of B, G, R, and unused per pixel
    RGBX = 5, ///< 4 bytes of R, G, B, and unused per pixel
    Gray = 6, ///< 1 byte of gray per pixel
  };

  size_t width;           ///< Length of a line (in pixels).
  size_t height;          ///< Number of lines in the frame.
  size_t bytes_per_pixel; ///< Number of bytes in a pixel. If frame format is 'Raw' this is the buffer size.
  unsigned char* data;    ///< Data of the frame (aligned). @see See Frame::Type for pixel format.
  uint32_t timestamp;     ///< Unit: roughly or exactly 0.1 millisecond
  uint32_t sequence;      ///< Increasing frame sequence number
  float exposure;         ///< From 0.5 (very bright) to ~60.0 (fully covered)
  float gain;             ///< From 1.0 (bright) to 1.5 (covered)
  float gamma;            ///< From 1.0 (bright) to 6.4 (covered)
  uint32_t status;        ///< zero if ok; non-zero for errors.
  Format format;          ///< Byte format. Informative only, doesn't indicate errors.

  /** Construct a new frame.
   * @param width Width in pixel
   * @param height Height in pixel
   * @param bytes_per_pixel Bytes per pixel
   * @param data_ Memory to store frame data. If `NULL`, new memory is allocated.
   */
  Frame(size_t width, size_t height, size_t bytes_per_pixel, unsigned char *data_ = NULL);
  virtual ~Frame();

  protected:
  unsigned char* rawdata; ///< Unaligned start of #data.
};

/** Callback interface to receive new frames. @ingroup frame
 * You can inherit from FrameListener and define your own listener.
 */
class LIBFREENECT2_API FrameListener
{
public:
  virtual ~FrameListener();

  /**
   * libfreenect2 calls this function when a new frame is decoded.
   * @param type Type of the new frame.
   * @param frame Data of the frame.
   * @return true if you want to take ownership of the frame, i.e. reuse/delete it. Will be reused/deleted by caller otherwise.
   */
  virtual bool onNewFrame(Frame::Type type, Frame *frame) = 0;
};

} /* namespace libfreenect2 */
#endif /* FRAME_LISTENER_HPP_ */
