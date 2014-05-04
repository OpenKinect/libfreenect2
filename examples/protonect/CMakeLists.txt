CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(libfreenect2)
SET(CMAKE_BUILD_TYPE Release)

SET(MY_DIR ${libfreenect2_SOURCE_DIR})

# additional cmake modules
LIST(APPEND CMAKE_MODULE_PATH ${MY_DIR}/cmake_modules)

# setup threading
INCLUDE(SetupLibfreenect2Threading)
INCLUDE_DIRECTORIES(${LIBFREENECT2_THREADING_INCLUDE_DIR})

#set the default path for built executables to the "bin" directory
SET(EXECUTABLE_OUTPUT_PATH ${MY_DIR}/bin)

# dependencies
FIND_PACKAGE(OpenCV REQUIRED)

# OpenCV
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})

# LibUSB
INCLUDE_DIRECTORIES("${MY_DIR}/../../depends/libusbx/include/libusb-1.0/")
LINK_DIRECTORIES("${MY_DIR}/../../depends/libusbx/lib/")

INCLUDE_DIRECTORIES("${MY_DIR}/include")

ADD_EXECUTABLE(Protonect
  src/transfer_pool.cpp
  src/event_loop.cpp
  
  src/double_buffer.cpp
  src/frame_listener.cpp
  
  src/rgb_packet_stream_parser.cpp
  src/rgb_packet_processor.cpp
  src/turbo_jpeg_rgb_packet_processor.cpp
  
  src/depth_packet_stream_parser.cpp
  src/depth_packet_processor.cpp
  src/cpu_depth_packet_processor.cpp
  
  Protonect.cpp
  
  ${LIBFREENECT2_THREADING_SOURCE}
)

TARGET_LINK_LIBRARIES(Protonect
  usb-1.0
  ${OpenCV_LIBS}
  turbojpeg
  ${LIBFREENECT2_THREADING_LIBRARIES}
)