/*
 * This file is part of the OpenKinect Project. http://www.openkinect.org
 *
 * Copyright (c) 2017 individual OpenKinect contributors. See the CONTRIB file
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

#include "streamer.h"
#include <cstdlib>

void Streamer::initialize()
{
  std::cout << "Initialize Streamer." << std::endl;

  jpegqual =  ENCODE_QUALITY; // Compression Parameter

  servAddress = SERVER_ADDRESS;
  servPort = Socket::resolveService(SERVER_PORT, "udp"); // Server port

  compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
  compression_params.push_back(jpegqual);
}

void Streamer::stream(libfreenect2::Frame* frame)
{
  try
  {
    // int total_pack = 1 + (encoded.size() - 1) / PACK_SIZE;
    cv::Mat frame_depth = cv::Mat(frame->height, frame->width, CV_32FC1, frame->data) / 10;
    cv::imencode(".jpg", frame_depth, encoded, compression_params);

    // resize image
    // resize(frame, encoded, Size(FRAME_WIDTH, FRAME_HEIGHT), 0, 0, INTER_LINEAR);

    // show encoded frame
    // cv::namedWindow( "streamed frame", CV_WINDOW_AUTOSIZE);
    // cv::imshow("streamed frame", encoded);
    // cv::waitKey(0);

    total_pack = 1 + (encoded.size() - 1) / PACK_SIZE;

    // send pre-info
    ibuf[0] = total_pack;
    sock.sendTo(ibuf, sizeof(int), servAddress, servPort);

    // send image data packet
    for(int i = 0; i < total_pack; i++)
      sock.sendTo( & encoded[i * PACK_SIZE], PACK_SIZE, servAddress, servPort);
  }
  catch (SocketException & e)
  {
    std::cerr << e.what() << std::endl;
    // exit(1);
  }
}
