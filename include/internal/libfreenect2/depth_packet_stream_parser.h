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

/** @file depth_packet_stream_parser.h Parser processor definitions of depth packets. */

#ifndef DEPTH_PACKET_STREAM_PARSER_H_
#define DEPTH_PACKET_STREAM_PARSER_H_

#include <stddef.h>
#include <stdint.h>

#include <libfreenect2/config.h>

#include <libfreenect2/depth_packet_processor.h>

#include <libfreenect2/data_callback.h>

namespace libfreenect2
{

/** Footer of a depth packet. */
LIBFREENECT2_PACK(struct DepthSubPacketFooter
{
  uint32_t magic0;
  uint32_t magic1;
  uint32_t timestamp;
  uint32_t sequence;
  uint32_t subsequence;
  uint32_t length;
  uint32_t fields[32];
});

/**
 * Parser of th depth stream, recognizes valid depth packets in the stream, and
 * passes them on for further processing.
 */
class DepthPacketStreamParser : public DataCallback
{
public:
  DepthPacketStreamParser();
  virtual ~DepthPacketStreamParser();

  void setPacketProcessor(libfreenect2::BaseDepthPacketProcessor *processor);

  virtual void onDataReceived(unsigned char* buffer, size_t length);
private:
  libfreenect2::BaseDepthPacketProcessor *processor_;

  size_t buffer_size_;
  DepthPacket packet_;
  libfreenect2::Buffer work_buffer_;

  uint32_t processed_packets_;
  uint32_t current_sequence_;
  uint32_t current_subsequence_;
};

} /* namespace libfreenect2 */
#endif /* DEPTH_PACKET_STREAM_PARSER_H_ */
