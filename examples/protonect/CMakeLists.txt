CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(libfreenect2)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

SET(MY_DIR ${libfreenect2_SOURCE_DIR})

OPTION(ENABLE_CXX11 "Enable C++11 support" OFF)
OPTION(ENABLE_OPENCL "Enable OpenCL support" ON)
OPTION(ENABLE_OPENGL "Enable OpenGL support" ON)

IF(ENABLE_CXX11)
  INCLUDE(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  IF(COMPILER_SUPPORTS_CXX11)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  ELSEIF(COMPILER_SUPPORTS_CXX0X)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  ELSE()
    MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  ENDIF()
ENDIF(ENABLE_CXX11)

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

FIND_PACKAGE(OpenCL)

# dependencies
FIND_PACKAGE(OpenCV REQUIRED)

# OpenCV
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})

# LibUSB
INCLUDE_DIRECTORIES("${MY_DIR}/../../depends/libusb/include/libusb-1.0/")
LINK_DIRECTORIES("${MY_DIR}/../../depends/libusb/lib/")

IF(ENABLE_OPENGL)
  # GLFW
  SET(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries")
  SET(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs")
  SET(GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs")
  SET(GLFW_BUILD_DOCS OFF CACHE BOOL "Build the GLFW documentation")

  ADD_SUBDIRECTORY(${MY_DIR}/../../depends/glfw_src/ ${MY_DIR}/../../depends/glfw)
  INCLUDE_DIRECTORIES(${MY_DIR}/../../depends/glfw_src/include/)
ENDIF(ENABLE_OPENGL)

if(APPLE OR WIN32)
  # libjpeg-turbo
  INCLUDE_DIRECTORIES("${MY_DIR}/../../depends/libjpeg_turbo/include/")
  LINK_DIRECTORIES("${MY_DIR}/../../depends/libjpeg_turbo/lib/")
endif()

SET(RESOURCES_INC_FILE "${MY_DIR}/src/resources.inc")

SET(SOURCES
  src/transfer_pool.cpp
  src/event_loop.cpp
  
  src/double_buffer.cpp
  src/frame_listener_impl.cpp
  src/packet_pipeline.cpp
  
  src/rgb_packet_stream_parser.cpp
  src/rgb_packet_processor.cpp
  src/turbo_jpeg_rgb_packet_processor.cpp
  
  src/depth_packet_stream_parser.cpp
  src/depth_packet_processor.cpp
  src/cpu_depth_packet_processor.cpp
  src/resource.cpp
  
  src/usb_control.cpp
  src/command_transaction.cpp
  src/registration.cpp
  src/libfreenect2.cpp
  ${LIBFREENECT2_THREADING_SOURCE}
  ${RESOURCES_INC_FILE}
)

SET(LIBUSB usb-1.0)
IF(MSVC)
  SET(LIBUSB libusb-1.0)
ENDIF(MSVC)

SET(LIBRARIES
  ${LIBUSB}
  ${OpenCV_LIBS}
  turbojpeg
  ${LIBFREENECT2_THREADING_LIBRARIES}
)

SET(RESOURCES
  11to16.bin 
  xTable.bin 
  zTable.bin
)


IF(ENABLE_OPENGL)
  SET(LIBFREENECT2_WITH_OPENGL_SUPPORT 1)
  LIST(APPEND SOURCES
    src/flextGL.c
    src/opengl_depth_packet_processor.cpp
  )

  LIST(APPEND LIBRARIES
    glfw
    ${GLFW_LIBRARIES}
  )

  LIST(APPEND RESOURCES
    src/shader/debug.fs
    src/shader/default.vs
    src/shader/filter1.fs
    src/shader/filter2.fs
    src/shader/stage1.fs
    src/shader/stage2.fs
  )
ENDIF(ENABLE_OPENGL)

IF(ENABLE_OPENCL AND OPENCL_FOUND)
  SET(LIBFREENECT2_WITH_OPENCL_SUPPORT 1)
  INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIRS})
  
  LIST(APPEND SOURCES
    src/opencl_depth_packet_processor.cpp
  )

  LIST(APPEND LIBRARIES
    ${OPENCL_LIBRARIES}
  )
  
  LIST(APPEND RESOURCES
    src/opencl_depth_packet_processor.cl
  )
ENDIF(ENABLE_OPENCL AND OPENCL_FOUND)

CONFIGURE_FILE("${MY_DIR}/include/libfreenect2/config.h.in" "${MY_DIR}/include/libfreenect2/config.h" @ONLY)

GENERATE_RESOURCES(${RESOURCES_INC_FILE} ${MY_DIR} ${RESOURCES})

INCLUDE_DIRECTORIES("${MY_DIR}/include")

ADD_DEFINITIONS(-DRESOURCES_INC)
ADD_LIBRARY(freenect2 SHARED ${SOURCES})
TARGET_LINK_LIBRARIES(freenect2 ${LIBRARIES})
  
ADD_EXECUTABLE(Protonect
  Protonect.cpp
)
  
TARGET_LINK_LIBRARIES(Protonect
  freenect2
)

CONFIGURE_FILE(freenect2.cmake.in "${PROJECT_BINARY_DIR}/freenect2Config.cmake" @ONLY)

INSTALL(TARGETS Protonect DESTINATION bin)
INSTALL(TARGETS freenect2 DESTINATION lib)
INSTALL(DIRECTORY "${MY_DIR}/include/" DESTINATION include PATTERN "*.in" EXCLUDE)
IF(LIBFREENECT2_THREADING_TINYTHREAD)
  INSTALL(FILES  "${MY_DIR}/src/tinythread/tinythread.h" DESTINATION include/${PROJECT_NAME}/tinythread/)
ENDIF(LIBFREENECT2_THREADING_TINYTHREAD)
INSTALL(FILES "${PROJECT_BINARY_DIR}/freenect2Config.cmake" DESTINATION lib/cmake/freenect2/)
  
