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
#include <jpeglib.h>
#include <stdexcept>
#include <unistd.h>
#include <fcntl.h>
#include <va/va.h>
#include <va/va_drm.h>

static inline VAStatus vaSafeCall(VAStatus err)
{
  if (err != VA_STATUS_SUCCESS)
    throw std::runtime_error(vaErrorStr(err));
  return err;
}

namespace libfreenect2
{

struct VaapiFrame: Frame
{
  VADisplay display;
  VAImage image;

  VaapiFrame(VADisplay display, size_t width, size_t height, size_t bytes_per_pixel):
    Frame(width, height, bytes_per_pixel, false),
    display(display)
  {
    /* Create image */
    VAImageFormat format = {0};
    format.fourcc = VA_FOURCC_BGRX;
    format.byte_order = VA_LSB_FIRST;
    format.bits_per_pixel = bytes_per_pixel;
    format.depth = 8;

    vaSafeCall(vaCreateImage(display, &format, width, height, &image));
  }

  void draw(VASurfaceID surface)
  {
    if (data != NULL)
      vaSafeCall(vaUnmapBuffer(display, image.buf));
    vaSafeCall(vaGetImage(display, surface, 0, 0, width, height, image.image_id));
    vaSafeCall(vaMapBuffer(display, image.buf, (void**)&data));
  }

  ~VaapiFrame()
  {
    if (data != NULL)
    {
      vaSafeCall(vaUnmapBuffer(display, image.buf));
      data = NULL;
    }
    vaSafeCall(vaDestroyImage(display, image.image_id));
  }
};

class VaapiDoubleBuffer: public DoubleBuffer
{
private:
  VADisplay display;
  VAContextID context;
  VABufferID buffers[2];

public:
  VaapiDoubleBuffer(VADisplay display, VAContextID context):
    DoubleBuffer(),
    display(display), context(context)
  {
    buffers[0] = VA_INVALID_ID;
    buffers[1] = VA_INVALID_ID;
  }

  virtual ~VaapiDoubleBuffer()
  {
    if(buffers[0] != VA_INVALID_ID)
    {
      vaSafeCall(vaUnmapBuffer(display, buffers[0]));
      vaSafeCall(vaDestroyBuffer(display, buffers[0]));
    }
    if(buffers[1] != VA_INVALID_ID)
    {
      vaSafeCall(vaUnmapBuffer(display, buffers[1]));
      vaSafeCall(vaDestroyBuffer(display, buffers[1]));
    }
  }

  virtual void allocate(size_t size)
  {
    vaSafeCall(vaCreateBuffer(display, context, VASliceDataBufferType, size, 1, NULL, &buffers[0]));
    vaSafeCall(vaCreateBuffer(display, context, VASliceDataBufferType, size, 1, NULL, &buffers[1]));

    buffer_[0].capacity = size;
    buffer_[0].length = 0;
    vaSafeCall(vaMapBuffer(display, buffers[0], (void**)&buffer_[0].data));

    buffer_[1].capacity = size;
    buffer_[1].length = 0;
    vaSafeCall(vaMapBuffer(display, buffers[1], (void**)&buffer_[1].data));
  }

  VABufferID back_id()
  {
    return buffers[(front_buffer_index_ + 1) & 1];
  }
};

class VaapiJpegRgbPacketProcessorImpl
{
public:
  int drm_fd;
  VADisplay display;
  VAConfigID config;
  VASurfaceID surface;
  VAContextID context;
  VABufferID pic_param_buf;
  VABufferID iq_buf;
  VABufferID huff_buf;
  VABufferID slice_param_buf;
  bool jpeg_first_packet;
  size_t jpeg_header_size;

  struct jpeg_decompress_struct dinfo;
  struct jpeg_error_mgr jerr;

  static const int WIDTH = 1920;
  static const int HEIGHT = 1080;

  VaapiFrame *frame;

  VaapiDoubleBuffer *packet_buffer;

  double timing_acc;
  double timing_acc_n;

  double timing_current_start;

  VaapiJpegRgbPacketProcessorImpl()
  {
    dinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&dinfo);

    initializeVaapi();

    newFrame();

    packet_buffer = new VaapiDoubleBuffer(display, context);

    timing_acc = 0.0;
    timing_acc_n = 0.0;
    timing_current_start = 0.0;

    jpeg_first_packet = true;
  }

  ~VaapiJpegRgbPacketProcessorImpl()
  {
    delete packet_buffer;
    delete frame;
    if (!jpeg_first_packet)
    {
      vaSafeCall(vaDestroyBuffer(display, pic_param_buf));
      vaSafeCall(vaDestroyBuffer(display, iq_buf));
      vaSafeCall(vaDestroyBuffer(display, huff_buf));
      vaSafeCall(vaDestroyBuffer(display, slice_param_buf));
    }
    vaSafeCall(vaDestroyContext(display, context));
    vaSafeCall(vaDestroySurfaces(display, &surface, 1));
    vaSafeCall(vaDestroyConfig(display, config));
    vaSafeCall(vaTerminate(display));
    if (drm_fd >= 0)
      close(drm_fd);
    jpeg_destroy_decompress(&dinfo);
  }

  void newFrame()
  {
    frame = new VaapiFrame(display, 1920, 1080, 4);
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
      std::cout << "[VaapiJpegRgbPacketProcessor] avg. time: " << (avg * 1000) << "ms -> ~" << (1.0/avg) << "Hz" << std::endl;
      timing_acc = 0.0;
      timing_acc_n = 0.0;
    }
  }

  void initializeVaapi()
  {
    /* Open display */
    static const char *drm_devices[] = {
      "/dev/dri/renderD128",
      "/dev/dri/card0",
      NULL,
    };
    for (int i = 0; drm_devices[i]; i++) {
      drm_fd = open(drm_devices[i], O_RDWR);
      if (drm_fd < 0)
        continue;
      display = vaGetDisplayDRM(drm_fd);
      if (vaDisplayIsValid(display))
        break;
      close(drm_fd);
      drm_fd = -1;
      display = NULL;
    }
    if (!vaDisplayIsValid(display))
      throw std::runtime_error("display not found");

    /* Initialize and create config */
    int major_ver, minor_ver;
    vaSafeCall(vaInitialize(display, &major_ver, &minor_ver));

    const char *driver = vaQueryVendorString(display);
    std::cerr << "[VaapiJpegRgbPacketProcessor::initializeVaapi] driver: " << driver << std::endl;

    int max_entrypoints = vaMaxNumEntrypoints(display);
    if (max_entrypoints <= 0)
      throw std::runtime_error("invalid MaxNumEntrypoints");

    VAEntrypoint entrypoints[max_entrypoints];
    int num_entrypoints;
    vaSafeCall(vaQueryConfigEntrypoints(display, VAProfileJPEGBaseline, entrypoints, &num_entrypoints));
    if (num_entrypoints < 0 || num_entrypoints > max_entrypoints)
      throw std::runtime_error("invalid number of entrypoints");

    int vld_entrypoint;
    for  (vld_entrypoint = 0; vld_entrypoint < num_entrypoints; vld_entrypoint++) {
      if (entrypoints[vld_entrypoint] == VAEntrypointVLD)
        break;
    }
    if (vld_entrypoint == num_entrypoints)
      throw std::runtime_error("did not find VLD");

    VAConfigAttrib attr;
    attr.type = VAConfigAttribRTFormat;
    vaSafeCall(vaGetConfigAttributes(display, VAProfileJPEGBaseline, VAEntrypointVLD, &attr, 1));
	unsigned int rtformat = VA_RT_FORMAT_YUV444;
    if ((attr.value & rtformat) == 0) {
      std::cerr << "[VaapiJpegRgbPacketProcessor::initializeVaapi] warning: YUV444 not supported by libva, chroma will be halved" << std::endl;
      rtformat = VA_RT_FORMAT_YUV420;
    }
    if ((attr.value & rtformat) == 0)
      throw std::runtime_error("does not support YUV420");

    vaSafeCall(vaCreateConfig(display, VAProfileJPEGBaseline, VAEntrypointVLD, &attr, 1, &config));

    /* Create surface and context */
    vaSafeCall(vaCreateSurfaces(display, rtformat, WIDTH, HEIGHT, &surface, 1, NULL, 0));

    vaSafeCall(vaCreateContext(display, config, WIDTH, HEIGHT, 0, &surface, 1, &context));
  }

  VABufferID createBuffer(VABufferType type, unsigned int size, void *data)
  {
    VABufferID buffer;
    vaSafeCall(vaCreateBuffer(display, context, type, size, 1, data, &buffer));
    if (buffer == VA_INVALID_ID)
      throw std::runtime_error("created invalid buffer");
    return buffer;
  }

  void createParameters(struct jpeg_decompress_struct &dinfo)
  {
    /* Picture Parameter */
    VAPictureParameterBufferJPEGBaseline pic = {0};
    pic.picture_width = dinfo.image_width;
    pic.picture_height = dinfo.image_height;
    for (int i = 0; i< dinfo.num_components; i++) {
      pic.components[i].component_id = dinfo.comp_info[i].component_id;
      pic.components[i].h_sampling_factor = dinfo.comp_info[i].h_samp_factor;
      pic.components[i].v_sampling_factor = dinfo.comp_info[i].v_samp_factor;
      pic.components[i].quantiser_table_selector = dinfo.comp_info[i].quant_tbl_no;
    }
    pic.num_components = dinfo.num_components;
    pic_param_buf = createBuffer(VAPictureParameterBufferType, sizeof(pic), &pic);

    /* IQ Matrix */
    VAIQMatrixBufferJPEGBaseline iq = {0};
    for (int i = 0; i < NUM_QUANT_TBLS; i++) {
      if (!dinfo.quant_tbl_ptrs[i])
        continue;
      iq.load_quantiser_table[i] = 1;
      /* Assuming dinfo.data_precision == 8 */
      const int natural_order[DCTSIZE2] = {
        0,  1,  8, 16,  9,  2,  3, 10,
        17, 24, 32, 25, 18, 11,  4,  5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13,  6,  7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
      };

      for (int j = 0; j < DCTSIZE2; j++)
        iq.quantiser_table[i][j] = dinfo.quant_tbl_ptrs[i]->quantval[natural_order[j]];
    }
    iq_buf = createBuffer(VAIQMatrixBufferType, sizeof(iq), &iq);

    /* Huffman Table */
    VAHuffmanTableBufferJPEGBaseline huff = {0};
    const int num_huffman_tables = 2;
    for (int i = 0; i < num_huffman_tables; i++) {
      if (!dinfo.dc_huff_tbl_ptrs[i] || !dinfo.ac_huff_tbl_ptrs[i])
        continue;
      huff.load_huffman_table[i] = 1;
      memcpy(huff.huffman_table[i].num_dc_codes, &dinfo.dc_huff_tbl_ptrs[i]->bits[1],
           sizeof(huff.huffman_table[i].num_dc_codes));
      memcpy(huff.huffman_table[i].dc_values, dinfo.dc_huff_tbl_ptrs[i]->huffval,
           sizeof(huff.huffman_table[i].dc_values));
      memcpy(huff.huffman_table[i].num_ac_codes, &dinfo.ac_huff_tbl_ptrs[i]->bits[1],
           sizeof(huff.huffman_table[i].num_ac_codes));
      memcpy(huff.huffman_table[i].ac_values, dinfo.ac_huff_tbl_ptrs[i]->huffval,
           sizeof(huff.huffman_table[i].ac_values));
    }
    huff_buf = createBuffer(VAHuffmanTableBufferType, sizeof(huff), &huff);

    /* Slice Parameter */
    VASliceParameterBufferJPEGBaseline *pslice;
    slice_param_buf = createBuffer(VASliceParameterBufferType, sizeof(*pslice), NULL);
    vaSafeCall(vaMapBuffer(display, slice_param_buf, (void**)&pslice));
    VASliceParameterBufferJPEGBaseline &slice = *pslice;

    slice.slice_data_offset = dinfo.src->next_input_byte - packet_buffer->back().data;
    slice.slice_data_flag = VA_SLICE_DATA_FLAG_ALL;
    for (int i = 0; i < dinfo.comps_in_scan; i++) {
      slice.components[i].component_selector = dinfo.cur_comp_info[i]->component_id;
      slice.components[i].dc_table_selector = dinfo.cur_comp_info[i]->dc_tbl_no;
      slice.components[i].ac_table_selector = dinfo.cur_comp_info[i]->ac_tbl_no;
    }
    slice.num_components = dinfo.comps_in_scan;
    slice.restart_interval = dinfo.restart_interval;
    unsigned int mcu_h_size = dinfo.max_h_samp_factor * DCTSIZE;
    unsigned int mcu_v_size = dinfo.max_v_samp_factor * DCTSIZE;
    unsigned int mcus_per_row = (WIDTH + mcu_h_size - 1) / mcu_h_size;
    unsigned int mcu_rows_in_scan = (HEIGHT + mcu_v_size - 1) / mcu_v_size;
    slice.num_mcus = mcus_per_row * mcu_rows_in_scan;

    vaSafeCall(vaUnmapBuffer(display, slice_param_buf));
  }

  void decompress(unsigned char *buf, size_t len)
  {
    if (jpeg_first_packet)
    {
      jpeg_mem_src(&dinfo, buf, len);
      int header_status = jpeg_read_header(&dinfo, true);
      if (header_status != JPEG_HEADER_OK)
        throw std::runtime_error("jpeg not ready for decompression");

      if (dinfo.image_width != WIDTH || dinfo.image_height != HEIGHT)
        throw std::runtime_error("image dimensions do not match preset");

      jpeg_first_packet = false;

      createParameters(dinfo);

      jpeg_header_size = len - dinfo.src->bytes_in_buffer;
      jpeg_abort_decompress(&dinfo);
    }
    /* Grab packet buffer for server */
    vaSafeCall(vaUnmapBuffer(display, packet_buffer->back_id()));

    /* The only parameter that changes after the first packet */
    VASliceParameterBufferJPEGBaseline *slice;
    vaSafeCall(vaMapBuffer(display, slice_param_buf, (void**)&slice));
    slice->slice_data_size = len - jpeg_header_size;
    vaSafeCall(vaUnmapBuffer(display, slice_param_buf));

    /* Commit buffers */
    vaSafeCall(vaBeginPicture(display, context, surface));
    VABufferID va_bufs[5] = {
      pic_param_buf, iq_buf, huff_buf, slice_param_buf, packet_buffer->back_id()
    };
    vaSafeCall(vaRenderPicture(display, context, va_bufs, 5));
    vaSafeCall(vaEndPicture(display, context));

    /* Sync surface */
    vaSafeCall(vaSyncSurface(display, surface));

    frame->draw(surface);

    /* Release packet buffer back to parser */
    vaSafeCall(vaMapBuffer(display, packet_buffer->back_id(), (void**)&packet_buffer->back().data));
  }
};

VaapiJpegRgbPacketProcessor::VaapiJpegRgbPacketProcessor() :
    impl_(new VaapiJpegRgbPacketProcessorImpl())
{
}

VaapiJpegRgbPacketProcessor::~VaapiJpegRgbPacketProcessor()
{
  delete impl_;
}

void VaapiJpegRgbPacketProcessor::process(const RgbPacket &packet)
{
  if (listener_ == 0)
    return;

  impl_->startTiming();

  impl_->frame->timestamp = packet.timestamp;
  impl_->frame->sequence = packet.sequence;

  try
  {
    impl_->decompress(packet.jpeg_buffer, packet.jpeg_buffer_length);
    if (listener_->onNewFrame(Frame::Color, impl_->frame))
    {
      impl_->newFrame();
    }
  }
  catch (const std::runtime_error &err)
  {
    std::cerr << "[VaapiJpegRgbPacketProcessor::doProcess] Failed to decompress: " << err.what() << std::endl;
  }

  impl_->stopTiming();
}

libfreenect2::DoubleBuffer *VaapiJpegRgbPacketProcessor::getPacketBuffer()
{
  return impl_->packet_buffer;
}

} /* namespace libfreenect2 */
