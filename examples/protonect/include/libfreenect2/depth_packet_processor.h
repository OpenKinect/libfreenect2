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

#ifndef DEPTH_PACKET_PROCESSOR_H_
#define DEPTH_PACKET_PROCESSOR_H_

#include <stddef.h>
#include <stdint.h>

#include <libfreenect2/frame_listener.h>

namespace libfreenect2
{

struct DepthPacket
{
  uint32_t sequence;
  unsigned char *buffer;
  size_t buffer_length;
};

class DepthPacketProcessor
{
public:
  DepthPacketProcessor();
  virtual ~DepthPacketProcessor();

  virtual void setFrameListener(libfreenect2::FrameListener *listener);
  virtual void process(const DepthPacket &packet) = 0;

  virtual void loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length) = 0;

protected:
  libfreenect2::FrameListener *listener_;
};

// TODO: push this to some internal namespace
// use pimpl to hide opencv dependency
class CpuDepthPacketProcessorImpl;

class CpuDepthPacketProcessor : public DepthPacketProcessor
{
public:
  CpuDepthPacketProcessor();
  virtual ~CpuDepthPacketProcessor();

  virtual void loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length);

  /**
   * GUESS: the x and z table follow some polynomial, until we know the exact polynom formula and its coefficients
   * just load them from a memory dump - although they probably vary per camera
   */
  void loadXTableFromFile(const char* filename);

  void loadZTableFromFile(const char* filename);

  void load11To16LutFromFile(const char* filename);

  virtual void process(const DepthPacket &packet);
private:
  CpuDepthPacketProcessorImpl *impl_;
};

} /* namespace libfreenect2 */
#endif /* DEPTH_PACKET_PROCESSOR_H_ */
