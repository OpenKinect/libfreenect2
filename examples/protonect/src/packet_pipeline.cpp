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

#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/async_packet_processor.h>

namespace libfreenect2
{

PacketPipeline::~PacketPipeline()
{
}

DefaultPacketPipeline::DefaultPacketPipeline()
{
  rgb_parser_ = new RgbPacketStreamParser();
  depth_parser_ = new DepthPacketStreamParser();

  rgb_processor_ = new TurboJpegRgbPacketProcessor();
  OpenGLDepthPacketProcessor *depth_processor = new OpenGLDepthPacketProcessor(0);
  depth_processor->load11To16LutFromFile("");
  depth_processor->loadXTableFromFile("");
  depth_processor->loadZTableFromFile("");
  depth_processor_ = depth_processor;

  async_rgb_processor_ = new AsyncPacketProcessor<RgbPacket>(rgb_processor_);
  async_depth_processor_ = new AsyncPacketProcessor<DepthPacket>(depth_processor_);

  rgb_parser_->setPacketProcessor(async_rgb_processor_);
  depth_parser_->setPacketProcessor(async_depth_processor_);
}

DefaultPacketPipeline::~DefaultPacketPipeline()
{
  delete async_rgb_processor_;
  delete async_depth_processor_;
  delete rgb_processor_;
  delete depth_processor_;
  delete rgb_parser_;
  delete depth_parser_;
}

DefaultPacketPipeline::PacketParser *DefaultPacketPipeline::getRgbPacketParser() const
{
  return rgb_parser_;
}

DefaultPacketPipeline::PacketParser *DefaultPacketPipeline::getIrPacketParser() const
{
  return depth_parser_;
}

RgbPacketProcessor *DefaultPacketPipeline::getRgbPacketProcessor() const
{
  return rgb_processor_;
}

DepthPacketProcessor *DefaultPacketPipeline::getDepthPacketProcessor() const
{
  return depth_processor_;
}

} /* namespace libfreenect2 */
