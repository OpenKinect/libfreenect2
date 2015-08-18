CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

if(WIN32 AND NOT MINGW)
  if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
  endif()
endif()

PROJECT(libfreenect2)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

SET(MY_DIR ${libfreenect2_SOURCE_DIR})
SET(DEPENDS_DIR "${MY_DIR}/../../depends" CACHE STRING "dependency directory must be set to 'false' if external deps are used")

OPTION(ENABLE_CXX11 "Enable C++11 support" OFF)
OPTION(ENABLE_OPENCL "Enable OpenCL support" ON)
OPTION(ENABLE_OPENGL "Enable OpenGL support" ON)

IF(ENABLE_CXX11)
  INCLUDE(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  IF(COMPILER_SUPPORTS_CXX11)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    SET(LIBFREENECT2_WITH_CXX11_SUPPORT 1)
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

INCLUDE(GenerateResources)

#set the default path for built executables to the "bin" directory
SET(EXECUTABLE_OUTPUT_PATH ${MY_DIR}/bin)

#set the default path for built libraries to the "lib" directory
SET(LIBRARY_OUTPUT_PATH ${MY_DIR}/lib)

# dependencies
FIND_PACKAGE(PkgConfig)    # try find PKGConfig as it will be used if found
FIND_PACKAGE(LibUSB REQUIRED)
FIND_PACKAGE(TurboJPEG REQUIRED) #does not provide a package-config file

# Add includes
INCLUDE_DIRECTORIES(
  "${MY_DIR}/include"
  ${LIBFREENECT2_THREADING_INCLUDE_DIR}
  ${LibUSB_INCLUDE_DIRS}
  ${TurboJPEG_INCLUDE_DIRS}
)

LINK_DIRECTORIES(${LibUSB_LIBRARY_DIRS})

SET(RESOURCES_INC_FILE "${MY_DIR}/src/resources.inc.h")

SET(SOURCES
  include/libfreenect2/protocol/command.h
  include/libfreenect2/protocol/command_transaction.h
  include/libfreenect2/protocol/response.h
  include/libfreenect2/protocol/usb_control.h
  
  include/libfreenect2/usb/event_loop.h
  include/libfreenect2/usb/transfer_pool.h

  include/libfreenect2/async_packet_processor.h  
  include/libfreenect2/depth_packet_processor.h
  include/libfreenect2/depth_packet_stream_parser.h
  include/libfreenect2/double_buffer.h
  include/libfreenect2/frame_listener.hpp
  include/libfreenect2/frame_listener_impl.h
  include/libfreenect2/config.h
  include/libfreenect2/libfreenect2.hpp
  include/libfreenect2/packet_pipeline.h
  include/libfreenect2/packet_processor.h
  include/libfreenect2/registration.h
  include/libfreenect2/resource.h
  include/libfreenect2/rgb_packet_processor.h
  include/libfreenect2/rgb_packet_stream_parser.h
  include/libfreenect2/threading.h
  
  src/transfer_pool.cpp
  src/event_loop.cpp
  src/usb_control.cpp
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
  src/command_transaction.cpp
  src/registration.cpp
  src/timer.cpp
  src/libfreenect2.cpp
  
  ${LIBFREENECT2_THREADING_SOURCE}
  ${RESOURCES_INC_FILE}
)

SET(LIBRARIES
  ${LibUSB_LIBRARIES}
  ${TurboJPEG_LIBRARIES}
  ${LIBFREENECT2_THREADING_LIBRARIES}
)

SET(RESOURCES
  11to16.bin 
  xTable.bin 
  zTable.bin
)

IF(ENABLE_OPENGL)
  FIND_PACKAGE(GLFW3)
  FIND_PACKAGE(OpenGL)
  IF(GLFW3_FOUND)
    INCLUDE_DIRECTORIES(${GLFW3_INCLUDE_DIRS})

    LINK_DIRECTORIES(${GLFW3_LIBRARY_DIRS})
    LIST(APPEND LIBRARIES
      ${GLFW3_LIBRARIES}
      ${OPENGL_gl_LIBRARY}
    )
  SET(LIBFREENECT2_WITH_OPENGL_SUPPORT 1)
  LIST(APPEND SOURCES
    src/flextGL.c
    src/opengl_depth_packet_processor.cpp
  )
  
  LIST(APPEND RESOURCES
    src/shader/debug.fs
    src/shader/default.vs
    src/shader/filter1.fs
    src/shader/filter2.fs
    src/shader/stage1.fs
    src/shader/stage2.fs
  )
  ENDIF()
ENDIF(ENABLE_OPENGL)

IF(ENABLE_OPENCL)
  FIND_PACKAGE(OpenCL)

  IF(OPENCL_FOUND)      
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
  ENDIF(OPENCL_FOUND)
ENDIF(ENABLE_OPENCL)

SET(CMAKE_INSTALL_RPATH ${LibUSB_LIBDIR})

CONFIGURE_FILE("${MY_DIR}/include/libfreenect2/config.h.in" "${MY_DIR}/include/libfreenect2/config.h" @ONLY)
GENERATE_RESOURCES(${RESOURCES_INC_FILE} ${MY_DIR} ${RESOURCES})

ADD_DEFINITIONS(-DRESOURCES_INC)
ADD_LIBRARY(freenect2 OBJECT ${SOURCES})
set_target_properties(freenect2 PROPERTIES POSITION_INDEPENDENT_CODE 1)
ADD_LIBRARY(freenect2shared SHARED $<TARGET_OBJECTS:freenect2>)
ADD_LIBRARY(freenect2static STATIC $<TARGET_OBJECTS:freenect2>)
set_target_properties(freenect2shared PROPERTIES OUTPUT_NAME freenect2)
set_target_properties(freenect2static PROPERTIES OUTPUT_NAME freenect2)
IF(MSVC)
  set_target_properties(freenect2static PROPERTIES OUTPUT_NAME freenect2static)
ENDIF()
MESSAGE("Linking with these libraries: ${LIBRARIES}")
TARGET_LINK_LIBRARIES(freenect2shared ${LIBRARIES})
  
SET(Protonect_src
    Protonect.cpp
    )
    
IF (ENABLE_OPENGL AND OPENGL_FOUND)
    LIST(APPEND Protonect_src
        viewer.h
        viewer.cpp
        src/flextGL.c
        )
ENDIF()
  
ADD_EXECUTABLE(Protonect
  ${Protonect_src}

)
  
TARGET_LINK_LIBRARIES(Protonect
  freenect2shared
)

CONFIGURE_FILE(freenect2.cmake.in "${PROJECT_BINARY_DIR}/freenect2Config.cmake" @ONLY)
CONFIGURE_FILE(freenect2.pc.in "${PROJECT_BINARY_DIR}/freenect2.pc" @ONLY)

INSTALL(TARGETS freenect2shared DESTINATION lib)
INSTALL(TARGETS freenect2static DESTINATION lib)
INSTALL(DIRECTORY "${MY_DIR}/include/" DESTINATION include PATTERN "*.in" EXCLUDE)
IF(LIBFREENECT2_THREADING_TINYTHREAD)
  INSTALL(FILES  "${MY_DIR}/src/tinythread/tinythread.h" DESTINATION include/${PROJECT_NAME}/tinythread/)
ENDIF(LIBFREENECT2_THREADING_TINYTHREAD)
INSTALL(FILES "${PROJECT_BINARY_DIR}/freenect2Config.cmake" DESTINATION lib/cmake/freenect2/)
INSTALL(FILES "${PROJECT_BINARY_DIR}/freenect2.pc" DESTINATION lib/pkgconfig/)
  
