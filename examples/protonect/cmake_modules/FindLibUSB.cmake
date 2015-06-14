# - Find libusb for portable USB support
# 
# If the LibUSB_ROOT environment variable
# is defined, it will be used as base path.
# The following standard variables get defined:
#  LibUSB_FOUND:    true if LibUSB was found
#  LibUSB_INCLUDE_DIR: the directory that contains the include file
#  LibUSB_LIBRARIES:  the libraries

IF(PKG_CONFIG_FOUND)
  SET(ENV{PKG_CONFIG_PATH} "${DEPENDS_DIR}/libusb/lib/pkgconfig")
  PKG_CHECK_MODULES(LibUSB libusb-1.0)
  RETURN()
ENDIF()

FIND_PATH(LibUSB_INCLUDE_DIRS
  NAMES libusb.h
  PATHS
    "${DEPENDS_DIR}/libusb"
    "${DEPENDS_DIR}/libusbx"
    ENV LibUSB_ROOT
  PATH_SUFFIXES
    include
    libusb
)

SET(SUFFIXES
  x64/Release/dll
  x64/Debug/dll
  Win32/Release/dll
  Win32/Debug/dll
  MS64
)
IF(USE_STATIC_LIBS)
  SET(SUFFIXES
    x64/Release/lib
    x64/Debug/lib
    Win32/Release/lib
    Win32/Debug/lib
  )
ENDIF()

FIND_LIBRARY(LibUSB_LIBRARIES
  NAMES libusb-1.0
  PATHS
    "${DEPENDS_DIR}/libusb"
    "${DEPENDS_DIR}/libusbx"
    ENV LibUSB_ROOT
  PATH_SUFFIXES
    ${SUFFIXES}
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibUSB DEFAULT_MSG LibUSB_LIBRARIES LibUSB_INCLUDE_DIRS)
