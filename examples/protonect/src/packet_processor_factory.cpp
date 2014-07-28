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

#include <libfreenect2/packet_processor_factory.h>
#include <libfreenect2/rgb_packet_stream_parser.h>
#include <libfreenect2/depth_packet_stream_parser.h>

#include <libfreenect2/async_packet_processor.h>
#include <libfreenect2/rgb_packet_processor.h>
#include <libfreenect2/depth_packet_processor.h>

namespace libfreenect2
{

PacketProcessorFactory::~PacketProcessorFactory()
{
}

DefaultPacketProcessorFactory *DefaultPacketProcessorFactory::instance()
{
  static DefaultPacketProcessorFactory factory;
  return &factory;
}

DefaultPacketProcessorFactory::DefaultPacketProcessorFactory() {}

DefaultPacketProcessorFactory::~DefaultPacketProcessorFactory() {}

void DefaultPacketProcessorFactory::create(PacketStreamParser **rgb_packet_stream_parser, PacketStreamParser **depth_packet_stream_parser, RgbPacketProcessor **rgb_packet_processor, DepthPacketProcessor **depth_packet_processor)
{
  RgbPacketStreamParser *rgb_parser = new RgbPacketStreamParser();
  DepthPacketStreamParser *depth_parser = new DepthPacketStreamParser();

  TurboJpegRgbPacketProcessor *rgb_processor = new TurboJpegRgbPacketProcessor();
  OpenGLDepthPacketProcessor *depth_processor = new OpenGLDepthPacketProcessor(0);
  depth_processor->load11To16LutFromFile("");
  depth_processor->loadXTableFromFile("");
  depth_processor->loadZTableFromFile("");

  rgb_parser->setPacketProcessor(rgb_processor->makeAsync());
  depth_parser->setPacketProcessor(depth_processor->makeAsync());

  *rgb_packet_stream_parser = rgb_parser;
  *depth_packet_stream_parser = depth_parser;

  *rgb_packet_processor = rgb_processor;
  *depth_packet_processor = depth_processor;
}

} /* namespace libfreenect2 */
