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
#include "libfreenect2/logging.h"

namespace nv_headers {
#define DONT_USE_EXTERN_C
#include "nv_headers/jpeglib.h"
}

#include <dlfcn.h>

#ifdef LIBFREENECT2_WITH_CXX11_SUPPORT
#define TYPEOF(expr) decltype(expr)
#else
#define TYPEOF(expr) __typeof__(expr)
#endif

#define FOR_ALL(MACRO) \
  MACRO(jpeg_std_error) \
  MACRO(jpeg_CreateDecompress) \
  MACRO(jpeg_mem_src) \
  MACRO(jpeg_read_header) \
  MACRO(jpeg_start_decompress) \
  MACRO(jpeg_read_scanlines) \
  MACRO(jpeg_finish_decompress) \
  MACRO(jpeg_abort_decompress) \
  MACRO(jpeg_destroy_decompress)

class libjpeg_handle
{
private:
  void *handle;
public:
  #define DECLARE(func) TYPEOF(&nv_headers::func) func;
  FOR_ALL(DECLARE)
  bool good;

  #define INIT(func) func(0),
  libjpeg_handle(): FOR_ALL(INIT) good(false)
  {
    handle = dlopen(LIBFREENECT2_TEGRAJPEG_LIBRARY, RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
    const char *err;
    err = dlerror();
    if (handle == NULL) {
      LOG_ERROR << "dlopen: " << err;
      return;
    }
    #define IMPORT(func) func = reinterpret_cast<TYPEOF(&nv_headers::func)>(dlsym(handle, #func)); if ((err = dlerror())) { LOG_ERROR << "dlsym: " << err; dlclose(handle); handle = NULL; return; }
    FOR_ALL(IMPORT)
    good = true;
  }

  ~libjpeg_handle()
  {
    if (handle)
      dlclose(handle);
  }
};

using namespace nv_headers;

namespace libfreenect2
{

class TegraImage: public Buffer
{
public:
  struct jpeg_decompress_struct dinfo;
};

class TegraImageAllocator: public Allocator
{
public:
  void *owner;
  struct jpeg_error_mgr *jerr;
  libjpeg_handle &libjpeg;

  TegraImageAllocator(void *owner, struct jpeg_error_mgr *jerr, libjpeg_handle &libjpeg):
    owner(owner), jerr(jerr), libjpeg(libjpeg) {}

  virtual Buffer *allocate(size_t size)
  {
    TegraImage *ti = new TegraImage();
    ti->allocator = this;
    ti->dinfo.client_data = owner;
    ti->dinfo.err = jerr;
    libjpeg.jpeg_create_decompress(&ti->dinfo);
    return ti;
  }

  virtual void free(Buffer *b)
  {
    if (b == NULL)
      return;
    TegraImage *ti = static_cast<TegraImage *>(b);
    libjpeg.jpeg_destroy_decompress(&ti->dinfo);
    delete ti;
  }
};

class TegraFrame: public Frame
{
public:
  TegraFrame(size_t width, size_t height, size_t bpp, TegraImage *ti):
    Frame(width, height, bpp, (unsigned char*)-1)
  {
    data = NULL;
    rawdata = reinterpret_cast<unsigned char*>(ti);
  }

  TegraImage *image()
  {
    return reinterpret_cast<TegraImage *>(rawdata);
  }

  virtual ~TegraFrame()
  {
    image()->allocator->free(image());
    rawdata = NULL;
  }

  void fill()
  {
    data = image()->dinfo.jpegTegraMgr->buff[0];
  }
};

class TegraJpegRgbPacketProcessorImpl: public WithPerfLogging
{
public:
  libjpeg_handle libjpeg;
  struct jpeg_error_mgr jerr;
  size_t real_bib;

  static const size_t WIDTH = 1920;
  static const size_t HEIGHT = 1080;
  static const size_t BPP = 4;

  bool good;

  TegraFrame *frame;

  Allocator *image_allocator;

  TegraJpegRgbPacketProcessorImpl():
    libjpeg(),
    good(true),
    frame(NULL),
    image_allocator(NULL)
  {
    if (!libjpeg.good) {
      good = false;
      return;
    }
    libjpeg.jpeg_std_error(&jerr);
    jerr.error_exit = TegraJpegRgbPacketProcessorImpl::my_error_exit;

    image_allocator = new PoolAllocator(new TegraImageAllocator(reinterpret_cast<void*>(this), &jerr, libjpeg));

    newFrame();
  }

  ~TegraJpegRgbPacketProcessorImpl()
  {
    delete frame;
    delete image_allocator;
  }

  void newFrame()
  {
    frame = new TegraFrame(WIDTH, HEIGHT, BPP, static_cast<TegraImage *>(image_allocator->allocate(0)));
    frame->format = Frame::RGBX;
  }

  static inline TegraJpegRgbPacketProcessorImpl *owner(j_decompress_ptr dinfo)
  {
    return static_cast<TegraJpegRgbPacketProcessorImpl*>(dinfo->client_data);
  }

  static int fill_input_buffer(j_decompress_ptr dinfo)
  {
    dinfo->src->bytes_in_buffer = owner(dinfo)->real_bib;
    return 1;
  }

  static void abort_jpeg_error(j_decompress_ptr dinfo, const char *msg)
  {
    owner(dinfo)->libjpeg.jpeg_abort_decompress(dinfo);
    LOG_ERROR << msg;
    owner(dinfo)->good = false;
  }

  static void my_error_exit(j_common_ptr info)
  {
    char buffer[JMSG_LENGTH_MAX];
    info->err->format_message(info, buffer);
    abort_jpeg_error((j_decompress_ptr)info, buffer);
  }

  void decompress(unsigned char *buf, size_t len)
  {
    j_decompress_ptr dinfo = &frame->image()->dinfo;
    libjpeg.jpeg_mem_src(dinfo, buf, len);

    // This hack prevents an extra memcpy in jpeg_read_header
    real_bib = len;
    dinfo->src->bytes_in_buffer = 0;
    dinfo->src->fill_input_buffer = TegraJpegRgbPacketProcessorImpl::fill_input_buffer;
    libjpeg.jpeg_read_header(dinfo, true);

    if (dinfo->progressive_mode)
      abort_jpeg_error(dinfo, "Tegra HW doesn't support progressive JPEG; use TurboJPEG");

    if (!dinfo->tegra_acceleration)
      abort_jpeg_error(dinfo, "Tegra HW acceleration is disabled unexpectedly");

    if (dinfo->image_width != WIDTH || dinfo->image_height != HEIGHT)
      abort_jpeg_error(dinfo, "image dimensions does not match preset");

    dinfo->out_color_space = JCS_RGBA_8888;

    libjpeg.jpeg_start_decompress(dinfo);

    // Hardware acceleration returns the entire surface in one go.
    // The normal way with software decoding uses jpeg_read_scanlines with loop.
    if (libjpeg.jpeg_read_scanlines(dinfo, NULL, 0) != dinfo->output_height || dinfo->output_height != HEIGHT)
      abort_jpeg_error(dinfo, "Incomplete decoding result");

    /* Empirically: 1 surface for RGBA; 3 surfaces for YUV */
    //size_t pitch = dinfo->jpegTegraMgr->pitch[0];
    //unsigned char *surface = dinfo->jpegTegraMgr->buff[0];
    //if (pitch == 0 || surface == NULL)
    //  abort_jpeg_error(dinfo, "Empty result buffer");

    frame->fill();

    libjpeg.jpeg_finish_decompress(dinfo);
  }
};

TegraJpegRgbPacketProcessor::TegraJpegRgbPacketProcessor() :
    impl_(new TegraJpegRgbPacketProcessorImpl())
{
}

TegraJpegRgbPacketProcessor::~TegraJpegRgbPacketProcessor()
{
  delete impl_;
}

bool TegraJpegRgbPacketProcessor::good()
{
  return impl_->good;
}

void TegraJpegRgbPacketProcessor::process(const RgbPacket &packet)
{
  if (listener_ == NULL)
    return;

  impl_->startTiming();

  impl_->frame->timestamp = packet.timestamp;
  impl_->frame->sequence = packet.sequence;
  impl_->frame->exposure = packet.exposure;
  impl_->frame->gain = packet.gain;
  impl_->frame->gamma = packet.gamma;

  impl_->decompress(packet.jpeg_buffer, packet.jpeg_buffer_length);

  impl_->stopTiming(LOG_INFO);

  if (!impl_->good)
    impl_->frame->status = 1;

  if (listener_->onNewFrame(Frame::Color, impl_->frame))
    impl_->newFrame();
}
} /* namespace libfreenect2 */
