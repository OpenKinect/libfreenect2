# FindTurboJPEG.cmake
# - TurboJPEG_FOUND
# - TurboJPEG_INCLUDE_DIRS
# - TurboJPEG_LIBRARIES

INCLUDE(CheckCSourceCompiles)

FIND_PATH(TurboJPEG_INCLUDE_DIRS
  turbojpeg.h
  DOC "Found TurboJPEG include directory"
  PATHS
    "${CMAKE_SOURCE_DIR}/../../depends/libjpeg_turbo/include"
    "C:/libjpeg-turbo64/include"
    "/opt/libjpeg-turbo/include"
    "$ENV{TurboJPEG_ROOT}/include"
)

FIND_LIBRARY(TurboJPEG_LIBRARIES
  NAMES turbojpeg.lib libturbojpeg.so libturbojpeg.so.0 libturbojpeg.a
  DOC "Found TurboJPEG library path"
  PATHS
    "${CMAKE_SOURCE_DIR}/../../depends/libjpeg_turbo/lib"
    "C:/libjpeg-turbo64/lib"
    "$ENV{TurboJPEG_ROOT}/lib"
    "/opt/libjpeg-turbo/lib64"
    "/opt/libjpeg-turbo/lib"
)

if(MSVC)
  set(CMAKE_REQUIRED_DEFINITIONS -MT)
endif()
set(CMAKE_REQUIRED_INCLUDES ${TurboJPEG_INCLUDE_DIRS})
set(CMAKE_REQUIRED_LIBRARIES ${TurboJPEG_LIBRARIES})
check_c_source_compiles("#include <turbojpeg.h>\nint main(void) { tjhandle h=tjInitCompress(); return 0; }" TURBOJPEG_WORKS)
set(CMAKE_REQUIRED_DEFINITIONS)
set(CMAKE_REQUIRED_INCLUDES)
set(CMAKE_REQUIRED_LIBRARIES)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TurboJPEG DEFAULT_MSG TurboJPEG_LIBRARIES TurboJPEG_INCLUDE_DIRS TURBOJPEG_WORKS)
