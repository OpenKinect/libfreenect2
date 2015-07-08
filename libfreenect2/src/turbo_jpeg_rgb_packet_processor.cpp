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

#include <libfreenect2/rgb_packet_processor.h>

#include <opencv2/opencv.hpp>
#include <turbojpeg.h>

namespace libfreenect2
{

class TurboJpegRgbPacketProcessorImpl
{
public:

  tjhandle decompressor;

  Frame *frame;

  double timing_acc;
  double timing_acc_n;

  double timing_current_start;

  TurboJpegRgbPacketProcessorImpl()
  {
    decompressor = tjInitDecompress();
    if(decompressor == 0)
    {
      std::cerr << "[TurboJpegRgbPacketProcessorImpl] Failed to initialize TurboJPEG decompressor! TurboJPEG error: '" << tjGetErrorStr() << "'" << std::endl;
    }

    newFrame();

    timing_acc = 0.0;
    timing_acc_n = 0.0;
    timing_current_start = 0.0;
  }

  ~TurboJpegRgbPacketProcessorImpl()
  {
    if(decompressor != 0)
    {
      if(tjDestroy(decompressor) == -1)
      {
        std::cerr << "[~TurboJpegRgbPacketProcessorImpl] Failed to destroy TurboJPEG decompressor! TurboJPEG error: '" << tjGetErrorStr() << "'" << std::endl;
      }
    }
  }

  void newFrame()
  {
    frame = new Frame(1920, 1080, tjPixelSize[TJPF_BGRX]);
  }

  void startTiming()
  {
    timing_current_start = cv::getTickCount();
  }

  void stopTiming()
  {
    timing_acc += (cv::getTickCount() - timing_current_start) / cv::getTickFrequency();
    timing_acc_n += 1.0;

    if(timing_acc_n >= 100.0)
    {
      double avg = (timing_acc / timing_acc_n);
      std::cout << "[TurboJpegRgbPacketProcessor] avg. time: " << (avg * 1000) << "ms -> ~" << (1.0/avg) << "Hz" << std::endl;
      timing_acc = 0.0;
      timing_acc_n = 0.0;
    }
  }
};

TurboJpegRgbPacketProcessor::TurboJpegRgbPacketProcessor() :
    impl_(new TurboJpegRgbPacketProcessorImpl())
{
}

TurboJpegRgbPacketProcessor::~TurboJpegRgbPacketProcessor()
{
  delete impl_;
}

void TurboJpegRgbPacketProcessor::process(const RgbPacket &packet)
{
  if(impl_->decompressor != 0 && listener_ != 0)
  {
    impl_->startTiming();

    impl_->frame->timestamp = packet.timestamp;
    impl_->frame->sequence = packet.sequence;

    int r = tjDecompress2(impl_->decompressor, packet.jpeg_buffer, packet.jpeg_buffer_length, impl_->frame->data, 1920, 1920 * tjPixelSize[TJPF_BGRX], 1080, TJPF_BGRX, 0);

    if(r == 0)
    {
      if(listener_->onNewFrame(Frame::Color, impl_->frame))
      {
        impl_->newFrame();
      }
    }
    else
    {
      std::cerr << "[TurboJpegRgbPacketProcessor::doProcess] Failed to decompress rgb image! TurboJPEG error: '" << tjGetErrorStr() << "'" << std::endl;
    }

    impl_->stopTiming();
  }
}

} /* namespace libfreenect2 */
