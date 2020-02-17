# FindTegraJPEG.cmake
# - TegraJPEG_FOUND
# - TegraJPEG_INCLUDE_DIRS
# - TegraJPEG_LIBRARIES

FIND_PATH(TegraJPEG_INCLUDE_DIRS
  libjpeg-8b/jpeglib.h
  DOC "Found TegraJPEG include directory"
  PATHS /usr/src/tegra_multimedia_api/include /usr/src/jetson_multimedia_api/include
  NO_DEFAULT_PATH
)

FIND_LIBRARY(TegraJPEG_LIBRARIES
  NAMES nvjpeg
  DOC "Found TegraJPEG library (libnvjpeg.so)"
  PATH_SUFFIXES tegra
)

IF(TegraJPEG_INCLUDE_DIRS AND TegraJPEG_LIBRARIES)
  INCLUDE(CheckCSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${TegraJPEG_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${TegraJPEG_LIBRARIES})
  check_c_source_compiles("#include <stdio.h>\n#include <libjpeg-8b/jpeglib.h>\nint main() { struct jpeg_decompress_struct d; jpeg_create_decompress(&d); d.jpegTegraMgr = 0; d.input_frame_buf = 0; return 0; }" TegraJPEG_WORKS)
  set(CMAKE_REQUIRED_INCLUDES)
  set(CMAKE_REQUIRED_LIBRARIES)
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TegraJPEG FOUND_VAR TegraJPEG_FOUND
  REQUIRED_VARS TegraJPEG_LIBRARIES TegraJPEG_INCLUDE_DIRS TegraJPEG_WORKS)
