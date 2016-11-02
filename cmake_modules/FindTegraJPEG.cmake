# FindTegraJPEG.cmake
# - TegraJPEG_FOUND
# - TegraJPEG_INCLUDE_DIRS
# - TegraJPEG_LIBRARIES

# Detect Linux4Tegra distribution
SET(L4T_RELEASE_FILE /etc/nv_tegra_release)
IF(EXISTS ${L4T_RELEASE_FILE})
  SET(TegraJPEG_IS_L4T TRUE)
  EXECUTE_PROCESS(
    COMMAND sha1sum --quiet -c /etc/nv_tegra_release
    RESULT_VARIABLE TegraJPEG_DRIVER_ERROR
    OUTPUT_VARIABLE TegraJPEG_DRIVER_OUTPUT
    ERROR_QUIET
  )
  IF(TegraJPEG_DRIVER_ERROR)
    MESSAGE(WARNING "Tegra drivers have wrong checksum:\n${TegraJPEG_DRIVER_OUTPUT}")
  ELSE()
    SET(TegraJPEG_DRIVER_OK TRUE)
  ENDIF()
ENDIF()

# Detect L4T version
IF(TegraJPEG_IS_L4T)
  FILE(READ ${L4T_RELEASE_FILE} L4T_RELEASE_CONTENT LIMIT 64 OFFSET 2)
  STRING(REGEX REPLACE "^R([0-9]*)[^,]*, REVISION: ([0-9.]*).*" "\\1" L4T_VER_MAJOR "${L4T_RELEASE_CONTENT}")
  STRING(REGEX REPLACE "^R([0-9]*)[^,]*, REVISION: ([0-9.]*).*" "\\2" L4T_VER_MINOR "${L4T_RELEASE_CONTENT}")
  SET(L4T_VER "${L4T_VER_MAJOR}.${L4T_VER_MINOR}")
  MESSAGE(STATUS "Found Linux4Tegra ${L4T_VER}")
  IF(L4T_VER VERSION_LESS 21.3.0)
    MESSAGE(WARNING "Linux4Tegra version (${L4T_VER}) less than minimum requirement (21.3)")
  ELSE()
    SET(TegraJPEG_L4T_OK TRUE)
  ENDIF()

  IF(L4T_VER MATCHES ^21.3)
    SET(L4T_GSTJPEG_URL_PART r21_Release_v3.0/sources)
  ELSEIF(L4T_VER MATCHES ^21.4)
    SET(L4T_GSTJPEG_URL_PART r21_Release_v4.0/source)
  ELSEIF(L4T_VER MATCHES ^21.5)
    SET(L4T_GSTJPEG_URL_PART r21_Release_v5.0/source)  
  ELSEIF(L4T_VER MATCHES ^23.1)
    SET(L4T_GSTJPEG_URL_PART r23_Release_v1.0/source)
  ELSEIF(L4T_VER MATCHES ^23.2)
    SET(L4T_GSTJPEG_URL_PART r23_Release_v2.0/source)
  ELSEIF(L4T_VER MATCHES ^24.1)
    SET(L4T_GSTJPEG_URL_PART r24_Release_v1.0/24.1_64bit/source)
  ELSEIF(L4T_VER MATCHES ^24.2)
    SET(L4T_GSTJPEG_URL_PART r24_Release_v2.0/BSP)
  ELSE()
    MESSAGE(WARNING "Linux4Tegra version (${L4T_VER}) is not recognized.")
    SET(TegraJPEG_L4T_OK FALSE)
  ENDIF()
ENDIF()

# Download gstjpeg source
IF(TegraJPEG_L4T_OK)
  IF(L4T_VER VERSION_LESS 24.2.0)
    SET(L4T_SRC_FILE "gstjpeg_src.tbz2")
  ELSE()
    SET(L4T_SRC_FILE "sources.tbz2")
  ENDIF()
  SET(L4T_GSTJPEG_URL "http://developer.download.nvidia.com/embedded/L4T/${L4T_GSTJPEG_URL_PART}/${L4T_SRC_FILE}")
  SET(L4T_GSTJPEG_DEST ${DEPENDS_DIR}/source/gstjpeg_src.tbz2)
  SET(L4T_SRC_DEST ${DEPENDS_DIR}/source/${L4T_SRC_FILE})
  IF(NOT EXISTS ${L4T_SRC_DEST})
    MESSAGE(STATUS "Downloading ${L4T_SRC_FILE}...")
    # Do we want checksum for the download?
    FILE(DOWNLOAD ${L4T_GSTJPEG_URL} ${L4T_SRC_DEST} STATUS L4T_SRC_STATUS)
    LIST(GET L4T_SRC_STATUS 0 L4T_SRC_ERROR)
    LIST(GET L4T_SRC_STATUS 1 L4T_SRC_MSG)
    IF(L4T_SRC_ERROR)
      MESSAGE(WARNING "Failed to download ${L4T_SRC_FILE}: ${L4T_SRC_MSG}")
      FILE(REMOVE ${L4T_SRC_FILE})
    ENDIF()
  ENDIF()
  IF(NOT EXISTS ${L4T_GSTJPEG_DEST} AND EXISTS ${L4T_SRC_DEST})
    MESSAGE(STATUS "Extracting ${L4T_SRC_FILE}...")
    EXECUTE_PROCESS(
      COMMAND ${CMAKE_COMMAND} -E tar xjf ${L4T_SRC_DEST}
      WORKING_DIRECTORY ${DEPENDS_DIR}
    )
  ENDIF()
  EXECUTE_PROCESS(
    COMMAND ${CMAKE_COMMAND} -E tar xjf ${L4T_GSTJPEG_DEST}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    RESULT_VARIABLE L4T_HEADERS_ERROR
    ERROR_VARIABLE L4T_HEADERS_MSG
  )
  IF(L4T_HEADERS_ERROR)
    MESSAGE(WARNING "Failed to unpack gstjpeg_src.tbz2: ${L4T_HEADERS_MSG}")
  ENDIF()
ENDIF()

FIND_PATH(TegraJPEG_INCLUDE_DIRS
  nv_headers/jpeglib.h
  DOC "Found TegraJPEG include directory"
  PATHS ${CMAKE_BINARY_DIR}/gstjpeg_src
  NO_DEFAULT_PATH
)

FIND_LIBRARY(TegraJPEG_LIBRARIES
  NAMES nvjpeg
  DOC "Found TegraJPEG library (libnvjpeg.so)"
  PATH_SUFFIXES tegra
)

FIND_LIBRARY(TegraJPEG_LIBRARIES
  NAMES jpeg
  DOC "Found TegraJPEG library (libjpeg.so)"
  PATHS /usr/lib/arm-linux-gnueabihf/tegra
  NO_DEFAULT_PATH
)

IF(TegraJPEG_INCLUDE_DIRS AND TegraJPEG_LIBRARIES)
  INCLUDE(CheckCSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${TegraJPEG_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${TegraJPEG_LIBRARIES})
  check_c_source_compiles("#include <stdio.h>\n#include <nv_headers/jpeglib.h>\nint main() { struct jpeg_decompress_struct d; jpeg_create_decompress(&d); d.jpegTegraMgr = 0; d.input_frame_buf = 0; return 0; }" TegraJPEG_WORKS)
  set(CMAKE_REQUIRED_INCLUDES)
  set(CMAKE_REQUIRED_LIBRARIES)
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TegraJPEG FOUND_VAR TegraJPEG_FOUND
  REQUIRED_VARS TegraJPEG_LIBRARIES TegraJPEG_INCLUDE_DIRS TegraJPEG_L4T_OK TegraJPEG_DRIVER_OK TegraJPEG_WORKS)
