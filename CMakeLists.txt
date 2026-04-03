CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

SET(PROJECT_VER_MAJOR 0)
SET(PROJECT_VER_MINOR 2)
SET(PROJECT_VER_PATCH 0)
SET(PROJECT_VER "${PROJECT_VER_MAJOR}.${PROJECT_VER_MINOR}.${PROJECT_VER_PATCH}")
SET(PROJECT_APIVER "${PROJECT_VER_MAJOR}.${PROJECT_VER_MINOR}")

if(WIN32 AND NOT MINGW)
  if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
  endif()
endif()

IF(WIN32)
  # no permission for the default install prefix %ProgramFiles%
  SET(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE STRING "installation path")
ENDIF()

IF(NOT DEFINED CMAKE_BUILD_TYPE)
  # No effect for multi-configuration generators (e.g. for Visual Studio)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose: RelWithDebInfo Release Debug MinSizeRel None")
ENDIF()

PROJECT(libfreenect2)

IF(POLICY CMP0042)
  cmake_policy(SET CMP0042 NEW)
ENDIF()

SET(MY_DIR ${libfreenect2_SOURCE_DIR})
SET(DEPENDS_DIR "${MY_DIR}/depends" CACHE STRING "dependency directory must be set to 'false' if external deps are used")

OPTION(BUILD_SHARED_LIBS "Build shared (ON) or static (OFF) libraries" ON)
OPTION(BUILD_EXAMPLES "Build examples" ON)
OPTION(BUILD_OPENNI2_DRIVER "Build OpenNI2 driver" ON)
OPTION(ENABLE_CXX11 "Enable C++11 support" OFF)
OPTION(ENABLE_OPENCL "Enable OpenCL support" ON)
OPTION(ENABLE_CUDA "Enable CUDA support" ON)
OPTION(ENABLE_OPENGL "Enable OpenGL support" ON)
OPTION(ENABLE_VAAPI "Enable VA-API support" ON)
OPTION(ENABLE_TEGRAJPEG "Enable Tegra HW JPEG support" ON)
OPTION(ENABLE_PROFILING "Collect profiling stats (memory consuming)" OFF)

IF(ENABLE_PROFILING)
  SET(LIBFREENECT2_WITH_PROFILING 1)
ENDIF()

IF(MSVC)
  # suppress several "possible loss of data" warnings, and
  # "zero-length array in struct" from libusb.h
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4244 /wd4200 /wd4305 /wd4146")
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS) # no warning for getenv()
ELSE()
  # Heed warnings from non-MSVC compilers
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
ENDIF()

SET(HAVE_CXX11 disabled)
IF(ENABLE_CXX11)
  INCLUDE(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  IF(COMPILER_SUPPORTS_CXX11)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    SET(LIBFREENECT2_WITH_CXX11_SUPPORT 1)
    SET(HAVE_CXX11 yes)
  ELSEIF(COMPILER_SUPPORTS_CXX0X)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    SET(HAVE_CXX11 c++0x)
  ELSE()
    SET(HAVE_CXX11 no)
    MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  ENDIF()
ENDIF(ENABLE_CXX11)

# additional cmake modules
LIST(APPEND CMAKE_MODULE_PATH ${MY_DIR}/cmake_modules)

# setup threading
INCLUDE(SetupLibfreenect2Threading)

INCLUDE(GenerateResources)

#set the default path for built executables to the "bin" directory
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

#set the default path for built libraries to the "lib" directory
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# dependencies
FIND_PACKAGE(PkgConfig)    # try find PKGConfig as it will be used if found
FIND_PACKAGE(LibUSB REQUIRED)

# Add includes
INCLUDE_DIRECTORIES(
  "${MY_DIR}/include"
  "${MY_DIR}/include/internal"
  ${PROJECT_BINARY_DIR} # for generated headers
  ${LIBFREENECT2_THREADING_INCLUDE_DIR}
  ${LibUSB_INCLUDE_DIRS}
)

SET(RESOURCES_INC_FILE "${PROJECT_BINARY_DIR}/resources.inc.h")

SET(SOURCES
  include/internal/libfreenect2/protocol/command.h
  include/internal/libfreenect2/protocol/command_transaction.h
  include/internal/libfreenect2/protocol/response.h
  include/internal/libfreenect2/protocol/usb_control.h

  include/internal/libfreenect2/usb/event_loop.h
  include/internal/libfreenect2/usb/transfer_pool.h

  include/libfreenect2/logger.h
  include/internal/libfreenect2/logging.h

  include/internal/libfreenect2/async_packet_processor.h
  include/internal/libfreenect2/depth_packet_processor.h
  include/internal/libfreenect2/depth_packet_stream_parser.h
  include/internal/libfreenect2/allocator.h
  include/libfreenect2/frame_listener.hpp
  include/libfreenect2/frame_listener_impl.h
  include/libfreenect2/libfreenect2.hpp
  include/libfreenect2/color_settings.h
  include/libfreenect2/led_settings.h
  include/libfreenect2/packet_pipeline.h
  include/internal/libfreenect2/packet_processor.h
  include/libfreenect2/registration.h
  include/internal/libfreenect2/resource.h
  include/internal/libfreenect2/rgb_packet_processor.h
  include/internal/libfreenect2/rgb_packet_stream_parser.h
  include/internal/libfreenect2/threading.h

  src/transfer_pool.cpp
  src/event_loop.cpp
  src/usb_control.cpp
  src/allocator.cpp
  src/frame_listener_impl.cpp
  src/packet_pipeline.cpp
  src/rgb_packet_stream_parser.cpp
  src/rgb_packet_processor.cpp
  src/depth_packet_stream_parser.cpp
  src/depth_packet_processor.cpp
  src/cpu_depth_packet_processor.cpp
  src/resource.cpp
  src/command_transaction.cpp
  src/registration.cpp
  src/logging.cpp
  src/libfreenect2.cpp

  ${LIBFREENECT2_THREADING_SOURCE}
  ${RESOURCES_INC_FILE}
  "${PROJECT_BINARY_DIR}/libfreenect2/config.h"
  "${PROJECT_BINARY_DIR}/libfreenect2/export.h"
)

SET(LIBRARIES
  ${LibUSB_LIBRARIES}
  ${LIBFREENECT2_THREADING_LIBRARIES}
)

SET(LIBFREENECT2_DLLS
  ${LibUSB_DLL}
)

SET(HAVE_VideoToolbox "no (Apple only)")
IF(APPLE)
  FIND_LIBRARY(VIDEOTOOLBOX_LIBRARY VideoToolbox)

  SET(HAVE_VideoToolbox no)
  IF(VIDEOTOOLBOX_LIBRARY)
    SET(LIBFREENECT2_WITH_VT_SUPPORT 1)
    SET(HAVE_VideoToolbox yes)

    FIND_LIBRARY(COREFOUNDATION_LIBRARY CoreFoundation REQUIRED)
    FIND_LIBRARY(COREMEDIA_LIBRARY CoreMedia REQUIRED)
    FIND_LIBRARY(COREVIDEO_LIBRARY CoreVideo REQUIRED)

    LIST(APPEND SOURCES
      src/vt_rgb_packet_processor.cpp
    )

    LIST(APPEND LIBRARIES
      ${VIDEOTOOLBOX_LIBRARY}
      ${COREFOUNDATION_LIBRARY}
      ${COREMEDIA_LIBRARY}
      ${COREVIDEO_LIBRARY}
    )
  ENDIF(VIDEOTOOLBOX_LIBRARY)
ENDIF(APPLE)

SET(HAVE_VAAPI disabled)
IF(ENABLE_VAAPI)
  IF(PKG_CONFIG_FOUND)
    PKG_CHECK_MODULES(VAAPI libva libva-drm)
  ENDIF()
  FIND_PACKAGE(JPEG)

  SET(HAVE_VAAPI no)
  IF(VAAPI_FOUND AND JPEG_FOUND)
    SET(LIBFREENECT2_WITH_VAAPI_SUPPORT 1)
    SET(HAVE_VAAPI yes)

    INCLUDE_DIRECTORIES(${VAAPI_INCLUDE_DIRS})

    LIST(APPEND SOURCES
      src/vaapi_rgb_packet_processor.cpp
    )
    LIST(APPEND LIBRARIES
      ${VAAPI_LIBRARIES}
      ${JPEG_LIBRARY}
    )
  ENDIF()
ENDIF(ENABLE_VAAPI)

SET(HAVE_TegraJPEG disabled)
IF(ENABLE_TEGRAJPEG)
  FIND_PACKAGE(TegraJPEG)

  SET(HAVE_TegraJPEG no)
  IF(TegraJPEG_FOUND)
    SET(LIBFREENECT2_WITH_TEGRAJPEG_SUPPORT 1)
    SET(HAVE_TegraJPEG yes)

    INCLUDE_DIRECTORIES(${TegraJPEG_INCLUDE_DIRS})

    LIST(APPEND SOURCES
      src/tegra_jpeg_rgb_packet_processor.cpp
    )

    LIST(APPEND LIBRARIES
      ${CMAKE_DL_LIBS}
    )
  ENDIF()
ENDIF()

IF(LIBFREENECT2_WITH_VT_SUPPORT)
  FIND_PACKAGE(TurboJPEG)
ELSE()
  # VAAPI can fail to start at runtime. It must have a fallback.
  FIND_PACKAGE(TurboJPEG REQUIRED)
ENDIF()

SET(HAVE_TurboJPEG no)
IF(TurboJPEG_FOUND)
  SET(LIBFREENECT2_WITH_TURBOJPEG_SUPPORT 1)
  SET(HAVE_TurboJPEG yes)

  INCLUDE_DIRECTORIES(${TurboJPEG_INCLUDE_DIRS})

  LIST(APPEND SOURCES
    src/turbo_jpeg_rgb_packet_processor.cpp
  )

  LIST(APPEND LIBRARIES
    ${TurboJPEG_LIBRARIES}
  )

  LIST(APPEND LIBFREENECT2_DLLS
   ${TurboJPEG_DLL}
  )
ENDIF()

SET(HAVE_OpenGL disabled)
IF(ENABLE_OPENGL)
  FIND_PACKAGE(GLFW3)
  FIND_PACKAGE(OpenGL)
  SET(HAVE_OpenGL no)
  IF(GLFW3_FOUND AND OPENGL_FOUND)
    SET(LIBFREENECT2_WITH_OPENGL_SUPPORT 1)
    SET(HAVE_OpenGL yes)

    INCLUDE_DIRECTORIES(${GLFW3_INCLUDE_DIRS})

    LIST(APPEND LIBFREENECT2_DLLS ${GLFW3_DLL})
    LIST(APPEND LIBRARIES
      ${GLFW3_LIBRARIES}
      ${OPENGL_gl_LIBRARY}
    )
    LIST(APPEND SOURCES
      src/flextGL.cpp
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

SET(HAVE_OpenCL disabled)
IF(ENABLE_OPENCL)
  FIND_PACKAGE(OpenCL)

  SET(HAVE_OpenCL no)
  IF(OpenCL_FOUND)
    SET(LIBFREENECT2_WITH_OPENCL_SUPPORT 1)
    SET(HAVE_OpenCL yes)

    IF(UNIX AND NOT APPLE)
      INCLUDE(CheckOpenCLICDLoader)
      IF(OpenCL_C_WORKS AND NOT OpenCL_CXX_WORKS)
        SET(LIBFREENECT2_OPENCL_ICD_LOADER_IS_OLD 1)
        SET(HAVE_OpenCL "yes but buggy")
        MESSAGE(WARNING "Your libOpenCL.so is incompatible with CL/cl.h. Install ocl-icd-opencl-dev to update libOpenCL.so?")
      ENDIF()
    ENDIF()
    INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS})

    LIST(APPEND SOURCES
      src/opencl_depth_packet_processor.cpp
      src/opencl_kde_depth_packet_processor.cpp
    )

    LIST(APPEND LIBRARIES
      ${OpenCL_LIBRARIES}
    )

    LIST(APPEND RESOURCES
      src/opencl_depth_packet_processor.cl
      src/opencl_kde_depth_packet_processor.cl
    )

    # Major Linux distro stable releases have buggy OpenCL ICD loader.
    # The workaround of disabling exceptions can only be set up during compile time.
    # Diabling it for all should be harmless. The flag is the same for GCC/Clang/ICC.
    IF(UNIX AND NOT APPLE)
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions")
    ENDIF()
  ENDIF(OpenCL_FOUND)
ENDIF(ENABLE_OPENCL)

SET(HAVE_CUDA disabled)
IF(ENABLE_CUDA)
  FIND_PACKAGE(CUDA)
  SET(HAVE_CUDA no)
  IF(CUDA_FOUND AND MSVC14 AND CUDA_VERSION VERSION_LESS 8.0)
    SET(HAVE_CUDA "no (VS2015 not supported)")
  ELSEIF(CUDA_FOUND)
    SET(LIBFREENECT2_WITH_CUDA_SUPPORT 1)
    SET(HAVE_CUDA yes)

    STRING(REPLACE "\\" "/" NVCUDASAMPLES_ROOT "$ENV{NVCUDASAMPLES_ROOT}")
    STRING(REPLACE "\\" "/" NVCUDASAMPLES8_0_ROOT "$ENV{NVCUDASAMPLES8_0_ROOT}")
    CUDA_INCLUDE_DIRECTORIES(
      "${MY_DIR}/include/"
      "${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc"
      "${NVCUDASAMPLES_ROOT}/common/inc"
      "${NVCUDASAMPLES8_0_ROOT}/common/inc"
    )
    SET(CUDA_FLAGS -use_fast_math)
    IF(NOT MSVC)
      SET(CUDA_FLAGS "${CUDA_FLAGS} -Xcompiler -fPIC")
    ENDIF()
    IF(HAVE_CXX11 STREQUAL yes AND CUDA_VERSION VERSION_GREATER 7.0)
      SET(CUDA_FLAGS "${CUDA_FLAGS} -std=c++11")
    ENDIF()

    SET(OLD_CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    STRING(REGEX REPLACE "-std=c\\+\\+.." "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    CUDA_COMPILE(CUDA_OBJECTS
      src/cuda_depth_packet_processor.cu
      src/cuda_kde_depth_packet_processor.cu
      OPTIONS ${CUDA_FLAGS}
    )
    SET(CMAKE_CXX_FLAGS "${OLD_CMAKE_CXX_FLAGS}")

    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

    LIST(APPEND SOURCES
      ${CUDA_OBJECTS}
    )

    LIST(APPEND LIBRARIES
      ${CUDA_LIBRARIES}
    )
  ENDIF()
ENDIF(ENABLE_CUDA)

# RPATH handling for CUDA 8.0 libOpenCL.so conflict. See #804.
IF(HAVE_OpenCL STREQUAL yes AND UNIX AND NOT APPLE)
  FILE(GLOB CUDA_ld_so_conf /etc/ld.so.conf.d/cuda*.conf)
  IF(CUDA_ld_so_conf)
    MESSAGE(WARNING "Your CUDA installation overrides OpenCL system library path.")
    GET_FILENAME_COMPONENT(OpenCL_LIB_PATH ${OpenCL_LIBRARY} DIRECTORY)
    SET(CMAKE_INSTALL_RPATH ${OpenCL_LIB_PATH} CACHE STRING "Set RPATH for system OpenCL")
  ENDIF()
ENDIF()

# RPATH handling for private libusb copies
# Users have two options:
# 1. Build libusb in depends/ and leave it there:
#      Do NOT set CMAKE_INSTALL_RPATH. It works by default.
# 2. Build libusb and install it somewhere:
#      Set CMAKE_INSTALL_RPATH to the libusb.so installation directory before compiling.
# Both command line -DCMAKE_INSTALL_RPATH=... and CMake GUI settings are accepted.
#
# Anyway if wrong versions of libusb is used, errors will be reported explicitly.
IF(NOT DEFINED CMAKE_INSTALL_RPATH AND NOT ${LibUSB_LIBDIR} MATCHES "^/usr/lib")
  SET(CMAKE_INSTALL_RPATH ${LibUSB_LIBDIR} CACHE STRING "Set RPATH for a private libusb")
ELSEIF(DEFINED CMAKE_INSTALL_RPATH)
  SET(CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} CACHE STRING "Set RPATH for a private libusb")
ENDIF()
IF(DEFINED CMAKE_INSTALL_RPATH)
  MESSAGE(STATUS "RPATH set to ${CMAKE_INSTALL_RPATH}")
ENDIF()

CONFIGURE_FILE("${MY_DIR}/include/libfreenect2/config.h.in" "${PROJECT_BINARY_DIR}/libfreenect2/config.h" @ONLY)
GENERATE_RESOURCES(${RESOURCES_INC_FILE} ${MY_DIR} ${RESOURCES})

ADD_DEFINITIONS(-DRESOURCES_INC)
ADD_LIBRARY(freenect2 ${SOURCES})
SET_TARGET_PROPERTIES(freenect2 PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
  VERSION ${PROJECT_VER}
  SOVERSION ${PROJECT_APIVER}
)
INCLUDE(GenerateExportHeader)
GENERATE_EXPORT_HEADER(freenect2
  BASE_NAME libfreenect2
  EXPORT_FILE_NAME libfreenect2/export.h
)

IF(MSVC AND NOT BUILD_SHARED_LIBS)
  # MSVC creates freenect2.lib for both dynamic and static by default
  set_target_properties(freenect2 PROPERTIES SUFFIX "static.lib")
ENDIF()
STRING(REPLACE ";" "\n " LIBRARIES_STRING "${LIBRARIES}")
MESSAGE(STATUS "Linking with these libraries: \n ${LIBRARIES_STRING}")
TARGET_LINK_LIBRARIES(freenect2 ${LIBRARIES})

CONFIGURE_FILE(freenect2.cmake.in "${PROJECT_BINARY_DIR}/freenect2Config.cmake" @ONLY)
CONFIGURE_FILE(freenect2Version.cmake.in "${PROJECT_BINARY_DIR}/freenect2ConfigVersion.cmake" @ONLY)
CONFIGURE_FILE(freenect2.pc.in "${PROJECT_BINARY_DIR}/freenect2.pc" @ONLY)

INSTALL(TARGETS freenect2 DESTINATION lib RUNTIME DESTINATION bin)
INSTALL(DIRECTORY "${MY_DIR}/include/${PROJECT_NAME}" DESTINATION include PATTERN "*.in" EXCLUDE)
INSTALL(DIRECTORY "${PROJECT_BINARY_DIR}/${PROJECT_NAME}" DESTINATION include)
INSTALL(FILES "${PROJECT_BINARY_DIR}/freenect2Config.cmake" DESTINATION lib/cmake/freenect2/)
INSTALL(FILES "${PROJECT_BINARY_DIR}/freenect2ConfigVersion.cmake" DESTINATION lib/cmake/freenect2/)
INSTALL(FILES "${PROJECT_BINARY_DIR}/freenect2.pc" DESTINATION lib/pkgconfig/)

ADD_SUBDIRECTORY(${MY_DIR}/doc)

SET(HAVE_Examples disabled)
IF(BUILD_EXAMPLES)
  SET(HAVE_Examples yes)
  MESSAGE(STATUS "Configurating examples")
  ADD_SUBDIRECTORY(${MY_DIR}/examples)
ENDIF()

SET(HAVE_OpenNI2 disabled)
IF(BUILD_OPENNI2_DRIVER)
  FIND_PACKAGE(OpenNI2)
  SET(HAVE_OpenNI2 no)
  IF(OpenNI2_FOUND)
    SET(HAVE_OpenNI2 yes)
    FILE(GLOB OPENNI2_DRIVER_SOURCES src/openni2/*.cpp)
    ADD_LIBRARY(freenect2-openni2 ${OPENNI2_DRIVER_SOURCES} ${LIBFREENECT2_THREADING_SOURCE})
    TARGET_INCLUDE_DIRECTORIES(freenect2-openni2 PRIVATE ${OpenNI2_INCLUDE_DIRS})
    TARGET_LINK_LIBRARIES(freenect2-openni2 freenect2 ${LIBFREENECT2_THREADING_LIBRARIES})
    SET_TARGET_PROPERTIES(freenect2-openni2 PROPERTIES SOVERSION 0)
    IF(NOT ${CMAKE_INSTALL_PREFIX} MATCHES "^/usr")
      SET_TARGET_PROPERTIES(freenect2-openni2 PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
    ENDIF()
    INSTALL(TARGETS freenect2-openni2 DESTINATION lib/OpenNI2/Drivers RUNTIME DESTINATION bin)
    ADD_CUSTOM_TARGET(install-openni2
      DEPENDS freenect2-openni2
      COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_INSTALL_PREFIX}/lib/OpenNI2/Drivers/" "${OpenNI2_LIBRARY_DIR}/OpenNI2/Drivers/"
      VERBATIM
    )
  ENDIF()
ENDIF()

OPTION(BUILD_STREAMER_RECORDER "Build streamer_recorder" OFF)
SET(HAVE_streamer_recorder disabled)
IF(BUILD_STREAMER_RECORDER)
  SET(HAVE_streamer_recorder yes)
  MESSAGE(STATUS "Configurating streamer_recorder")
  ADD_SUBDIRECTORY(${MY_DIR}/tools/streamer_recorder)
ENDIF()

GET_CMAKE_PROPERTY(vars VARIABLES)
MESSAGE(STATUS "Feature list:")
FOREACH(var ${vars})
  IF(var MATCHES ^HAVE_)
    STRING(REPLACE HAVE_ "" feature ${var})
    MESSAGE(STATUS "  ${feature}    ${${var}}")
  ENDIF()
ENDFOREACH()
