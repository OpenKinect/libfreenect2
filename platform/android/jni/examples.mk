
include $(CLEAR_VARS)

LIBFREENECT2_ROOT := ../../..

LOCAL_SRC_FILES := $(LIBFREENECT2_ROOT)/examples/Protonect.cpp

LOCAL_C_INCLUDES += \
  $(LIBFREENECT2_ROOT)/include \
  $(LIBFREENECT2_ROOT)/platform/android

LOCAL_SHARED_LIBRARIES += libfreenect2

LOCAL_MODULE := Protonect

include $(BUILD_EXECUTABLE)
