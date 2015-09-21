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

#include <iostream>
#include <fstream>

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/depth_packet_processor.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/protocol/response.h>
#include <libfreenect2/threading.h>

static bool data_dumped = false;

class LIBFREENECT2_API DumpPacketProcessor : public libfreenect2::DepthPacketProcessor
{
public:
  DumpPacketProcessor() {}

  virtual void loadP0TablesFromCommandResponse(unsigned char* buffer, size_t buffer_length)
  {
    libfreenect2::protocol::P0TablesResponse* p0table = (libfreenect2::protocol::P0TablesResponse*)buffer;

    if(buffer_length < sizeof(libfreenect2::protocol::P0TablesResponse))
    {
      std::cerr << "[CpuDepthPacketProcessor::loadP0TablesFromCommandResponse] P0Table response too short!" << std::endl;
      return;
    }

    std::cerr << "[DumpPacketProcessor::loadP0TablesFromCommandResponse] Exporting p0 tables" << std::endl;
    std::ofstream p00out("p00.bin", std::ios::out | std::ios::binary);
    p00out.write(reinterpret_cast<char*>(p0table->p0table0), 512*424*sizeof(uint16_t));
    p00out.close();

    std::ofstream p01out("p01.bin", std::ios::out | std::ios::binary);
    p01out.write(reinterpret_cast<char*>(p0table->p0table1), 512*424*sizeof(uint16_t));
    p01out.close();

    std::ofstream p02out("p02.bin", std::ios::out | std::ios::binary);
    p02out.write(reinterpret_cast<char*>(p0table->p0table2), 512*424*sizeof(uint16_t));
    p02out.close();
  }

  virtual void process(const libfreenect2::DepthPacket &packet)
  {
    if(!::data_dumped && (packet.sequence > 16))
    {
      std::cerr << "[DumpPacketProcessor::process] Exporting depth packet " << packet.sequence << std::endl;
      std::ofstream rawIrOut("rawir.bin", std::ios::out | std::ios::binary);
      rawIrOut.write(reinterpret_cast<char*>(packet.buffer), packet.buffer_length);
      rawIrOut.close();
      ::data_dumped = true;
    }
  }
};

class LIBFREENECT2_API DumpPacketPipeline : public libfreenect2::BasePacketPipeline
{
protected:
  virtual libfreenect2::DepthPacketProcessor *createDepthPacketProcessor()
  {
    DumpPacketProcessor *depth_processor = new DumpPacketProcessor();
    return depth_processor;
  }

public:
  DumpPacketPipeline()
  {
    initialize();
  }
};

int main(int argc, char **argv)
{
  std::string program_path(argv[0]);
  size_t executable_name_idx = program_path.rfind("test_opengl");
  std::string binpath = "./";

  if(executable_name_idx != std::string::npos)
  {
    binpath = program_path.substr(0, executable_name_idx);
  }

  libfreenect2::Freenect2 freenect2;
  libfreenect2::Freenect2Device *dev;
  dev = freenect2.openDefaultDevice(new DumpPacketPipeline());

  if(dev == 0)
  {
    std::cout << "no device connected or failure opening the default one!" << std::endl;
    return -1;
  }

  dev->start();

  while(!data_dumped)
  {
    std::cerr << ".";
    libfreenect2::this_thread::sleep_for(libfreenect2::chrono::milliseconds(100));
  }
  std::cerr << std::endl;

  dev->stop();
  dev->close();

  return 0;
}
