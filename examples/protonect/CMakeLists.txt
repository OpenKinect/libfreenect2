CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(libfreenect2)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

SET(MY_DIR ${libfreenect2_SOURCE_DIR})

# additional cmake modules
LIST(APPEND CMAKE_MODULE_PATH ${MY_DIR}/cmake_modules)

# setup threading
INCLUDE(SetupLibfreenect2Threading)
INCLUDE_DIRECTORIES(${LIBFREENECT2_THREADING_INCLUDE_DIR})

INCLUDE(GenerateResources)

#set the default path for built executables to the "bin" directory
SET(EXECUTABLE_OUTPUT_PATH ${MY_DIR}/bin)

#set the default path for built libraries to the "lib" directory
SET(LIBRARY_OUTPUT_PATH ${MY_DIR}/lib)

# dependencies
FIND_PACKAGE(OpenCV REQUIRED)
find_library(OPENCL_LIBRARIES OpenCL)

# OpenCV
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})

# LibUSB
INCLUDE_DIRECTORIES("${MY_DIR}/../../depends/libusb/include/libusb-1.0/")
LINK_DIRECTORIES("${MY_DIR}/../../depends/libusb/lib/")

# GLEW
INCLUDE_DIRECTORIES("${MY_DIR}/../../depends/glew/include/")
if (APPLE)
  LINK_DIRECTORIES("${MY_DIR}/../../depends/glew/lib/")
else()
  LINK_DIRECTORIES("${MY_DIR}/../../depends/glew/lib64/")
endif()
ADD_DEFINITIONS(-DGLEW_MX -DGLEW_STATIC)

# GLFW
SET(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")
SET(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs")
SET(GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs")
SET(GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation")

ADD_SUBDIRECTORY(${MY_DIR}/../../depends/glfw_src/ ${MY_DIR}/../../depends/glfw)
INCLUDE_DIRECTORIES(${MY_DIR}/../../depends/glfw_src/include/)

if(APPLE)
  # libjpeg-turbo
  INCLUDE_DIRECTORIES("${MY_DIR}/../../depends/libjpeg_turbo/include/")
  LINK_DIRECTORIES("${MY_DIR}/../../depends/libjpeg_turbo/lib/")
endif()

SET(RESOURCES_INC_FILE "${MY_DIR}/src/resources.inc")
GENERATE_RESOURCES(${RESOURCES_INC_FILE} ${MY_DIR} 
  11to16.bin 
  xTable.bin 
  zTable.bin 
  src/shader/debug.fs
  src/shader/default.vs
  src/shader/filter1.fs
  src/shader/filter2.fs
  src/shader/stage1.fs
  src/shader/stage2.fs
  src/opencl_depth_packet_processor.cl
)
add_definitions(-DREG_OPENCL_FILE="src/opencl_depth_packet_processor.cl")

INCLUDE_DIRECTORIES("${MY_DIR}/include")

ADD_DEFINITIONS(-DRESOURCES_INC)
ADD_LIBRARY(freenect2 SHARED
  src/opengl.cpp
  src/transfer_pool.cpp
  src/event_loop.cpp
  
  src/double_buffer.cpp
  src/frame_listener_impl.cpp
  
  src/rgb_packet_stream_parser.cpp
  src/rgb_packet_processor.cpp
  src/turbo_jpeg_rgb_packet_processor.cpp
  
  src/depth_packet_stream_parser.cpp
  src/depth_packet_processor.cpp
  src/cpu_depth_packet_processor.cpp
  src/opengl_depth_packet_processor.cpp
  src/opencl_depth_packet_processor.cpp
  src/resource.cpp
  
  src/usb_control.cpp
  src/command_transaction.cpp
  src/libfreenect2.cpp
  ${LIBFREENECT2_THREADING_SOURCE}
  ${RESOURCES_INC_FILE}
)

TARGET_LINK_LIBRARIES(freenect2
  usb-1.0
  ${OpenCV_LIBS}
  turbojpeg
  ${LIBFREENECT2_THREADING_LIBRARIES}
  GLEWmx
  glfw
  ${GLFW_LIBRARIES}
  OpenCL
)

ADD_EXECUTABLE(test_opengl
  src/test_opengl_depth_packet_processor.cpp
)

TARGET_LINK_LIBRARIES(test_opengl
  freenect2
)
  
ADD_EXECUTABLE(Protonect
  Protonect.cpp
)
  
TARGET_LINK_LIBRARIES(Protonect
  freenect2
)

set_target_properties(Protonect PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS Protonect freenect2
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install_files(/include/${PROJECT_NAME} FILES include/${PROJECT_NAME}/libfreenect2.hpp include/${PROJECT_NAME}/frame_listener.hpp)
install_files(/lib FILES "${MY_DIR}/../../depends/libusb/lib/")
if(APPLE)
  install_files(/lib FILES "${MY_DIR}/../../depends/glew/lib/")
  install_files(/lib FILES "${MY_DIR}/../../depends/libjpeg_turbo/lib/")
else()
  install_files(/lib FILES "${MY_DIR}/../../depends/glew/lib64/")
endif()
