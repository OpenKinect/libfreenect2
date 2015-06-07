include(CheckCSourceCompiles)

find_path(TurboJPEG_INCLUDE_DIR turbojpeg.h DOC "Found TurboJPEG include directory)" PATHS "${CMAKE_SOURCE_DIR}/../../depends/libjpeg_turbo/include" "C:/libjpeg-turbo64/include" "/opt/libjpeg-turbo/include" "$ENV{TurboJPEG_ROOT}/include")

if(TurboJPEG_INCLUDE_DIR STREQUAL "TurboJPEG_INCLUDE_DIR-NOTFOUND")
  message(FATAL_ERROR "Could not find turbojpeg.h - Try define TurboJPEG_ROOT as a system variable.")
else()
  message(STATUS "TurboJPEG_INCLUDE_DIR = ${TurboJPEG_INCLUDE_DIR}")
endif()

find_library(TurboJPEG_LIBRARY NAMES turbojpeg.lib libturbojpeg.so libturbojpeg.so.0 libturbojpeg.a DOC "Found TurboJPEG library path" PATHS "${CMAKE_SOURCE_DIR}/../../depends/libjpeg_turbo/lib" "C:/libjpeg-turbo64/lib" "$ENV{TurboJPEG_ROOT}/lib" "/opt/libjpeg-turbo/lib64" "/opt/libjpeg-turbo/lib")

if(WIN32)
  set(CMAKE_REQUIRED_DEFINITIONS -MT)
endif()
set(CMAKE_REQUIRED_INCLUDES ${TurboJPEG_INCLUDE_DIR})
set(CMAKE_REQUIRED_LIBRARIES ${TurboJPEG_LIBRARY})
check_c_source_compiles("#include <turbojpeg.h>\nint main(void) { tjhandle h=tjInitCompress(); return 0; }" TURBOJPEG_WORKS)
set(CMAKE_REQUIRED_DEFINITIONS)
set(CMAKE_REQUIRED_INCLUDES)
set(CMAKE_REQUIRED_LIBRARIES)
if(NOT TURBOJPEG_WORKS)
  message(FATAL_ERROR "Could not link with TurboJPEG library ${TurboJPEG_LIBRARY}.  If it is installed in a different place, then set TurboJPEG_LIBRARY accordingly.")
endif()

message(STATUS "TurboJPEG_LIBRARY = ${TurboJPEG_LIBRARY}")
