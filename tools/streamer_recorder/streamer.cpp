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

try {

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
    for (int i = 0; i < total_pack; i++)
        sock.sendTo( & encoded[i * PACK_SIZE], PACK_SIZE, servAddress, servPort);


    } catch (SocketException & e) {
        cerr << e.what() << endl;
        // exit(1);
    }
}
