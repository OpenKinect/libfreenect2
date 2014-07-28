/*
 * packet_processor_factory.h
 *
 *  Created on: Jul 28, 2014
 *      Author: christiankerl
 */

#ifndef PACKET_PROCESSOR_FACTORY_H_
#define PACKET_PROCESSOR_FACTORY_H_

#include <libfreenect2/usb/transfer_pool.h>
#include <libfreenect2/depth_packet_processor.h>
#include <libfreenect2/rgb_packet_processor.h>

namespace libfreenect2
{

class PacketProcessorFactory
{
public:
  typedef libfreenect2::usb::TransferPool::DataReceivedCallback PacketStreamParser;
  virtual ~PacketProcessorFactory();

  virtual void create(PacketStreamParser **rgb_packet_stream_parser, PacketStreamParser **depth_packet_stream_parser, RgbPacketProcessor **rgb_packet_processor, DepthPacketProcessor **depth_packet_processor) = 0;
};

class DefaultPacketProcessorFactory : public PacketProcessorFactory
{
public:
  static DefaultPacketProcessorFactory *instance();

  DefaultPacketProcessorFactory();
  virtual ~DefaultPacketProcessorFactory();
  virtual void create(PacketStreamParser **rgb_packet_stream_parser, PacketStreamParser **depth_packet_stream_parser, RgbPacketProcessor **rgb_packet_processor, DepthPacketProcessor **depth_packet_processor);
};

} /* namespace libfreenect2 */
#endif /* PACKET_PROCESSOR_FACTORY_H_ */
