#ifndef STREAMER_H
#define STREAMER_H

#include <libfreenect2/frame_listener.hpp>

#include "PracticalSocket.h"
#include <opencv2/opencv.hpp>
#include "config.h"

class Streamer
{
public:
  // methods
  void initialize();
  void stream(libfreenect2::Frame* frame);

private:
  // frame related parameters
  int jpegqual; // Compression Parameter
  vector < int > compression_params;
  vector < uchar > encoded;
  int total_pack;
  int ibuf[1];

  // udp related parameters
  string servAddress; // Server IP adress
  unsigned short servPort; // Server port
  UDPSocket sock;
};

#endif
