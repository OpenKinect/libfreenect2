
LIBFREENECT2_ROOT := ../../..

include $(CLEAR_VARS)
LOCAL_MODULE := libusb
LOCAL_SRC_FILES := $(LIBUSB_ROOT)/$(LIBUSB_SHARED_REL)
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libturbojpeg
LOCAL_SRC_FILES := $(LIBTURBOJPEG_ROOT)/$(LIBTURBOJPEG_SHARED_REL)
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LIBFREENECT2_SRC := $(LIBFREENECT2_ROOT)/src

LOCAL_C_INCLUDES += \
  $(LIBUSB_ROOT)/libusb \
  $(LIBTURBOJPEG_ROOT) \
  $(LIBFREENECT2_ROOT)/include \
  $(LIBFREENECT2_ROOT)/include/internal \
  $(LIBFREENECT2_SRC)/tinythread \
  $(LIBFREENECT2_ROOT)/platform/android

LOCAL_SRC_FILES := \
  $(LIBFREENECT2_SRC)/tinythread/tinythread.cpp \
  $(LIBFREENECT2_SRC)/allocator.cpp \
  $(LIBFREENECT2_SRC)/command_transaction.cpp \
  $(LIBFREENECT2_SRC)/cpu_depth_packet_processor.cpp \
  $(LIBFREENECT2_SRC)/depth_packet_processor.cpp \
  $(LIBFREENECT2_SRC)/depth_packet_stream_parser.cpp \
  $(LIBFREENECT2_SRC)/event_loop.cpp \
  $(LIBFREENECT2_SRC)/frame_listener_impl.cpp \
  $(LIBFREENECT2_SRC)/libfreenect2.cpp \
  $(LIBFREENECT2_SRC)/logging.cpp \
  $(LIBFREENECT2_SRC)/packet_pipeline.cpp \
  $(LIBFREENECT2_SRC)/registration.cpp \
  $(LIBFREENECT2_SRC)/resource.cpp \
  $(LIBFREENECT2_SRC)/rgb_packet_processor.cpp \
  $(LIBFREENECT2_SRC)/rgb_packet_stream_parser.cpp \
  $(LIBFREENECT2_SRC)/transfer_pool.cpp \
  $(LIBFREENECT2_SRC)/turbo_jpeg_rgb_packet_processor.cpp \
  $(LIBFREENECT2_SRC)/usb_control.cpp 

LOCAL_SHARED_LIBRARIES += libusb libturbojpeg

LOCAL_MODULE := libfreenect2

include $(BUILD_SHARED_LIBRARY)
