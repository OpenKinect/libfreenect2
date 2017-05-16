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
#include <libfreenect2/logging.h>

#include <VideoToolbox/VideoToolbox.h>

namespace libfreenect2 {

class VTFrame: public Frame
{
 public:
  VTFrame(size_t width, size_t height, size_t bytes_per_pixel, CVPixelBufferRef pixelBuffer) :
      Frame(width,
            height,
            bytes_per_pixel,
            reinterpret_cast<unsigned char *>(CVPixelBufferGetBaseAddress(lockPixelBuffer(pixelBuffer)))),
      pixelBuffer(pixelBuffer) {
  }

  ~VTFrame() {
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
    CVPixelBufferRelease(pixelBuffer);
  }

 protected:
  CVPixelBufferRef lockPixelBuffer(CVPixelBufferRef _pixelBuffer) {
    CVPixelBufferLockBaseAddress(_pixelBuffer, 0);

    return _pixelBuffer;
  }

  CVPixelBufferRef pixelBuffer;
};

class VTRgbPacketProcessorImpl: public WithPerfLogging
{
 public:
  CMFormatDescriptionRef format;
  VTDecompressionSessionRef decoder;

  VTRgbPacketProcessorImpl() {
    int32_t width = 1920, height = 1080;

    CMVideoFormatDescriptionCreate(NULL, kCMVideoCodecType_JPEG, width, height, NULL, &format);

    int32_t pixelFormat = kCVPixelFormatType_32BGRA;
    const void *outputKeys[] = {kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferWidthKey, kCVPixelBufferHeightKey};
    const void *outputValues[] =
        {CFNumberCreate(NULL, kCFNumberSInt32Type, &pixelFormat), CFNumberCreate(NULL, kCFNumberSInt32Type, &width),
         CFNumberCreate(NULL, kCFNumberSInt32Type, &height)};

    CFDictionaryRef outputConfiguration = CFDictionaryCreate(NULL,
                                                             outputKeys,
                                                             outputValues,
                                                             3,
                                                             &kCFTypeDictionaryKeyCallBacks,
                                                             &kCFTypeDictionaryValueCallBacks);

    VTDecompressionOutputCallbackRecord callback = {&VTRgbPacketProcessorImpl::decodeFrame, NULL};

    VTDecompressionSessionCreate(NULL, format, NULL, outputConfiguration, &callback, &decoder);

    CFRelease(outputConfiguration);
  }

  ~VTRgbPacketProcessorImpl() {
    VTDecompressionSessionInvalidate(decoder);
    CFRelease(decoder);
    CFRelease(format);
  }

  static void decodeFrame(void *decompressionOutputRefCon,
                          void *sourceFrameRefCon,
                          OSStatus status,
                          VTDecodeInfoFlags infoFlags,
                          CVImageBufferRef pixelBuffer,
                          CMTime presentationTimeStamp,
                          CMTime presentationDuration) {
    CVPixelBufferRef *outputPixelBuffer = (CVPixelBufferRef *) sourceFrameRefCon;
    *outputPixelBuffer = CVPixelBufferRetain(pixelBuffer);
  }
};

VTRgbPacketProcessor::VTRgbPacketProcessor()
    : RgbPacketProcessor ()
    , impl_(new VTRgbPacketProcessorImpl())
{
}

VTRgbPacketProcessor::~VTRgbPacketProcessor()
{
  delete impl_;
}

void VTRgbPacketProcessor::process(const RgbPacket &packet)
{
  if (listener_ != 0) {
    impl_->startTiming();

    CMBlockBufferRef blockBuffer;
    CMBlockBufferCreateWithMemoryBlock(
        NULL,
        packet.jpeg_buffer,
        packet.jpeg_buffer_length,
        kCFAllocatorNull,
        NULL,
        0,
        packet.jpeg_buffer_length,
        0,
        &blockBuffer
    );

    CMSampleBufferRef sampleBuffer;
    CMSampleBufferCreate(
        NULL,
        blockBuffer,
        true,
        NULL,
        NULL,
        impl_->format,
        1,
        0,
        NULL,
        0,
        NULL,
        &sampleBuffer
    );

    CVPixelBufferRef pixelBuffer = NULL;
    VTDecompressionSessionDecodeFrame(impl_->decoder, sampleBuffer, 0, &pixelBuffer, NULL);

    Frame *frame = new VTFrame(1920, 1080, 4, pixelBuffer);
    frame->format = Frame::BGRX;

    frame->timestamp = packet.timestamp;
    frame->sequence = packet.sequence;
    frame->exposure = packet.exposure;
    frame->gain = packet.gain;
    frame->gamma = packet.gamma;

    listener_->onNewFrame(Frame::Color, frame);

    CFRelease(sampleBuffer);
    CFRelease(blockBuffer);

    impl_->stopTiming(LOG_INFO);
  }
}

} /* namespace libfreenect2 */
