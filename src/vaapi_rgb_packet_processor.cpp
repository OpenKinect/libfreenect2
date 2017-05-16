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

#include <cstring>
#include <cstdio> //jpeglib.h does not include stdio.h
#include <jpeglib.h>
#include <unistd.h>
#include <fcntl.h>
#include <va/va.h>
#include <va/va_drm.h>
#include "libfreenect2/logging.h"
#include "libfreenect2/allocator.h"

#define CHECK_COND(cond) do { if (!(cond)) { LOG_ERROR << #cond " failed"; return false; } } while(0)
#define CHECK_VA(expr) do { VAStatus err = (expr); if (err != VA_STATUS_SUCCESS) { LOG_ERROR << #expr ": " << vaErrorStr(err); return false; } } while(0)
#define CALL_VA(expr) do { VAStatus err = (expr); if (err != VA_STATUS_SUCCESS) { LOG_ERROR << #expr ": " << vaErrorStr(err); } } while(0)

namespace libfreenect2
{

class VaapiImage: public Buffer
{
public:
  VAImage image;
};

class VaapiImageAllocator: public Allocator
{
public:
  VADisplay display;
  unsigned short width;
  unsigned short height;
  VAImageFormat format;
  Allocator *pool;

  VaapiImageAllocator(VADisplay display, unsigned short width, unsigned short height, const VAImageFormat &format):
    display(display), width(width), height(height), format(format)
  {
  }

  virtual Buffer *allocate(size_t size)
  {
    VaapiImage *vi = new VaapiImage();
    vi->allocator = this;
    CALL_VA(vaCreateImage(display, &format, width, height, &vi->image));
    return vi;
  }

  virtual void free(Buffer *b)
  {
    if (b == NULL)
      return;
    VaapiImage *vi = static_cast<VaapiImage *>(b);
    if (vi->data) {
      CALL_VA(vaUnmapBuffer(display, vi->image.buf));
      vi->data = NULL;
    }
    CALL_VA(vaDestroyImage(display, vi->image.image_id));
    delete vi;
  }
};

class VaapiFrame: public Frame
{
public:
  VaapiFrame(VaapiImage *vi):
    Frame(vi->image.width, vi->image.height, vi->image.format.bits_per_pixel/8, (unsigned char*)-1)
  {
    data = NULL;
    rawdata = reinterpret_cast<unsigned char*>(vi);
  }

  virtual ~VaapiFrame()
  {
    VaapiImage *vi = reinterpret_cast<VaapiImage *>(rawdata);
    vi->allocator->free(vi);
    rawdata = NULL;
  }

  bool draw(VADisplay display, VASurfaceID surface)
  {
    VaapiImage *vi = reinterpret_cast<VaapiImage *>(rawdata);
    VAImage &image = vi->image;
    data = NULL;
    if (vi->data != NULL) {
      vi->data = NULL;
      CHECK_VA(vaUnmapBuffer(display, image.buf));
    }
    CHECK_VA(vaGetImage(display, surface, 0, 0, image.width, image.height, image.image_id));
    CHECK_VA(vaMapBuffer(display, image.buf, (void**)&vi->data));
    data = vi->data;
    return true;
  }
};

class VaapiBuffer: public Buffer
{
public:
  VABufferID id;

  VaapiBuffer(): Buffer(), id(VA_INVALID_ID) {}
};

class VaapiAllocator: public Allocator
{
private:
  VADisplay display;
  VAContextID context;

  bool allocate_va(VaapiBuffer *b, size_t size)
  {
    CHECK_VA(vaCreateBuffer(display, context, VASliceDataBufferType, size, 1, NULL, &b->id));
    CHECK_VA(vaMapBuffer(display, b->id, (void**)&b->data));
    b->capacity = size;
    return true;
  }

public:
  VaapiAllocator(VADisplay display, VAContextID context):
    display(display), context(context) {}

  virtual Buffer *allocate(size_t size)
  {
    VaapiBuffer *vb = new VaapiBuffer();
    if (!allocate_va(vb, size))
      vb->data = NULL;
    return vb;
  }

  virtual void free(Buffer *b)
  {
    if (b == NULL)
      return;
    VaapiBuffer *vb = static_cast<VaapiBuffer *>(b);
    if (vb->data) {
      CALL_VA(vaUnmapBuffer(display, vb->id));
      CALL_VA(vaDestroyBuffer(display, vb->id));
    }
    delete vb;
  }
};

class VaapiRgbPacketProcessorImpl: public WithPerfLogging
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

  bool good;

  static const int WIDTH = 1920;
  static const int HEIGHT = 1080;

  VaapiFrame *frame;

  Allocator *buffer_allocator;
  Allocator *image_allocator;

  VaapiRgbPacketProcessorImpl():
    frame(NULL),
    buffer_allocator(NULL),
    image_allocator(NULL)
  {
    dinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&dinfo);

    good = initializeVaapi();
    if (!good)
      return;

    buffer_allocator = new PoolAllocator(new VaapiAllocator(display, context));

    VAImageFormat format = {0};
    format.fourcc = VA_FOURCC_BGRX;
    format.byte_order = VA_LSB_FIRST;
    format.bits_per_pixel = 4*8;
    format.depth = 8;

    image_allocator = new PoolAllocator(new VaapiImageAllocator(display, WIDTH, HEIGHT, format));

    newFrame();

    jpeg_first_packet = true;
  }

  ~VaapiRgbPacketProcessorImpl()
  {
    delete frame;
    delete buffer_allocator;
    delete image_allocator;
    if (good && !jpeg_first_packet) {
      CALL_VA(vaDestroyBuffer(display, pic_param_buf));
      CALL_VA(vaDestroyBuffer(display, iq_buf));
      CALL_VA(vaDestroyBuffer(display, huff_buf));
      CALL_VA(vaDestroyBuffer(display, slice_param_buf));
    }
    if (good) {
      CALL_VA(vaDestroyContext(display, context));
      CALL_VA(vaDestroySurfaces(display, &surface, 1));
      CALL_VA(vaDestroyConfig(display, config));
      CALL_VA(vaTerminate(display));
    }
    if (drm_fd >= 0)
      close(drm_fd);
    jpeg_destroy_decompress(&dinfo);
  }

  void newFrame()
  {
    frame = new VaapiFrame(static_cast<VaapiImage *>(image_allocator->allocate(0)));
    frame->format = Frame::BGRX;
  }

  bool initializeVaapi()
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
    CHECK_COND(vaDisplayIsValid(display));

    /* Initialize and create config */
    int major_ver, minor_ver;
    CHECK_VA(vaInitialize(display, &major_ver, &minor_ver));

    LOG_INFO << "driver: " << vaQueryVendorString(display);

    int max_entrypoints = vaMaxNumEntrypoints(display);
    CHECK_COND(max_entrypoints >= 1);

    VAEntrypoint entrypoints[max_entrypoints];
    int num_entrypoints;
    CHECK_VA(vaQueryConfigEntrypoints(display, VAProfileJPEGBaseline, entrypoints, &num_entrypoints));
    CHECK_COND(num_entrypoints >= 1 && num_entrypoints <= max_entrypoints);

    int vld_entrypoint;
    for  (vld_entrypoint = 0; vld_entrypoint < num_entrypoints; vld_entrypoint++) {
      if (entrypoints[vld_entrypoint] == VAEntrypointVLD)
        break;
    }
    CHECK_COND(vld_entrypoint < num_entrypoints);

    VAConfigAttrib attr;
    attr.type = VAConfigAttribRTFormat;
    CHECK_VA(vaGetConfigAttributes(display, VAProfileJPEGBaseline, VAEntrypointVLD, &attr, 1));
	unsigned int rtformat = VA_RT_FORMAT_YUV444;
    if ((attr.value & rtformat) == 0) {
      LOG_WARNING << "YUV444 not supported by libva, chroma will be halved";
      rtformat = VA_RT_FORMAT_YUV420;
    }
    CHECK_COND((attr.value & rtformat) != 0);

    CHECK_VA(vaCreateConfig(display, VAProfileJPEGBaseline, VAEntrypointVLD, &attr, 1, &config));

    /* Create surface and context */
    CHECK_VA(vaCreateSurfaces(display, rtformat, WIDTH, HEIGHT, &surface, 1, NULL, 0));

    CHECK_VA(vaCreateContext(display, config, WIDTH, HEIGHT, 0, &surface, 1, &context));

    return true;
  }

  VABufferID createBuffer(VABufferType type, unsigned int size, void *data)
  {
    VABufferID buffer = VA_INVALID_ID;
    CALL_VA(vaCreateBuffer(display, context, type, size, 1, data, &buffer));
    if (buffer == VA_INVALID_ID)
      LOG_ERROR << "failed to create valid buffer";
    return buffer;
  }

  bool createParameters(struct jpeg_decompress_struct &dinfo, const unsigned char *vb_start)
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
    CHECK_VA(vaMapBuffer(display, slice_param_buf, (void**)&pslice));
    VASliceParameterBufferJPEGBaseline &slice = *pslice;

    slice.slice_data_offset = dinfo.src->next_input_byte - vb_start;
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

    CHECK_VA(vaUnmapBuffer(display, slice_param_buf));
    return true;
  }

  bool decompress(unsigned char *buf, size_t len, VaapiBuffer *vb)
  {
    if (jpeg_first_packet) {
      jpeg_mem_src(&dinfo, buf, len);
      int header_status = jpeg_read_header(&dinfo, true);
      CHECK_COND(header_status == JPEG_HEADER_OK);
      CHECK_COND(dinfo.image_width == WIDTH && dinfo.image_height == HEIGHT);

      jpeg_first_packet = false;
      if (!createParameters(dinfo, vb->data))
        return false;

      jpeg_header_size = len - dinfo.src->bytes_in_buffer;
      jpeg_abort_decompress(&dinfo);
    }
    /* Grab the packet buffer for VAAPI backend */
    CHECK_VA(vaUnmapBuffer(display, vb->id));

    /* The only parameter that changes after the first packet */
    VASliceParameterBufferJPEGBaseline *slice;
    CHECK_VA(vaMapBuffer(display, slice_param_buf, (void**)&slice));
    slice->slice_data_size = len - jpeg_header_size;
    CHECK_VA(vaUnmapBuffer(display, slice_param_buf));

    /* Commit buffers */
    CHECK_VA(vaBeginPicture(display, context, surface));
    VABufferID va_bufs[5] = {pic_param_buf, iq_buf, huff_buf, slice_param_buf, vb->id};
    CHECK_VA(vaRenderPicture(display, context, va_bufs, 5));
    CHECK_VA(vaEndPicture(display, context));

    /* Sync surface */
    CHECK_VA(vaSyncSurface(display, surface));

    if (!frame->draw(display, surface))
      return false;

    CHECK_VA(vaMapBuffer(display, vb->id, (void**)&vb->data));

    return true;
  }
};

VaapiRgbPacketProcessor::VaapiRgbPacketProcessor() :
    impl_(new VaapiRgbPacketProcessorImpl())
{
}

VaapiRgbPacketProcessor::~VaapiRgbPacketProcessor()
{
  delete impl_;
}

bool VaapiRgbPacketProcessor::good()
{
  return impl_->good;
}

void VaapiRgbPacketProcessor::process(const RgbPacket &packet)
{
  if (listener_ == 0)
    return;

  impl_->startTiming();

  impl_->frame->timestamp = packet.timestamp;
  impl_->frame->sequence = packet.sequence;
  impl_->frame->exposure = packet.exposure;
  impl_->frame->gain = packet.gain;
  impl_->frame->gamma = packet.gamma;

  unsigned char *buf = packet.jpeg_buffer;
  size_t len = packet.jpeg_buffer_length;
  VaapiBuffer *vb = static_cast<VaapiBuffer *>(packet.memory);
  impl_->good = impl_->decompress(buf, len, vb);

  impl_->stopTiming(LOG_INFO);

  if (!impl_->good)
    impl_->frame->status = 1;

  if (listener_->onNewFrame(Frame::Color, impl_->frame))
    impl_->newFrame();
}

Allocator *VaapiRgbPacketProcessor::getAllocator()
{
  return impl_->buffer_allocator;
}
} /* namespace libfreenect2 */
