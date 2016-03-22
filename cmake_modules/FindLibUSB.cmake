# - Find libusb for portable USB support
# 
# If the LibUSB_ROOT environment variable
# is defined, it will be used as base path.
# The following standard variables get defined:
#  LibUSB_FOUND:    true if LibUSB was found
#  LibUSB_INCLUDE_DIR: the directory that contains the include file
#  LibUSB_LIBRARIES:  the libraries

IF(PKG_CONFIG_FOUND)
  IF(DEPENDS_DIR) #Otherwise use System pkg-config path
    SET(ENV{PKG_CONFIG_PATH} "${DEPENDS_DIR}/libusb/lib/pkgconfig")
  ENDIF()
  SET(MODULE "libusb-1.0")
  IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
    SET(MODULE "libusb-1.0>=1.0.20")
  ENDIF()
  IF(LibUSB_FIND_REQUIRED)
    SET(LibUSB_REQUIRED "REQUIRED")
  ENDIF()
  PKG_CHECK_MODULES(LibUSB ${LibUSB_REQUIRED} ${MODULE})

  FIND_LIBRARY(LibUSB_LIBRARY
    NAMES ${LibUSB_LIBRARIES}
    HINTS ${LibUSB_LIBRARY_DIRS}
  )
  SET(LibUSB_LIBRARIES ${LibUSB_LIBRARY})

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
    include/libusb-1.0
)

SET(LIBUSB_NAME libusb)
IF(WIN32)
  INCLUDE(CheckCSourceRuns)
  CHECK_C_SOURCE_RUNS("#include <windows.h>\nint main(){return !LoadLibraryA(\"libusbK\");}" LIBUSB_WITH_LIBUSBK)
  CHECK_C_SOURCE_RUNS("#include <windows.h>\nint main(){return !LoadLibraryA(\"UsbDkHelper\");}" LIBUSB_WITH_USBDK)

  IF(LIBUSB_USE_USBDK)
    SET(LIBUSB_NAME libusb-usbdk)
  ENDIF()

  IF(LIBUSB_NAME MATCHES ^libusb-usbdk$ AND NOT LIBUSB_WITH_USBDK)
    MESSAGE(WARNING "UsbDk device driver is not found. Fall back to libusbK.")
    SET(LIBUSB_NAME libusb)
  ENDIF()

  IF(LIBUSB_NAME MATCHES ^libusb$ AND NOT LIBUSB_WITH_LIBUSBK)
    MESSAGE(FATAL_ERROR "No USB device driver is installed.")
  ENDIF()
ENDIF()

FIND_LIBRARY(LibUSB_LIBRARIES
  NAMES ${LIBUSB_NAME}-1.0
  PATHS
    "${DEPENDS_DIR}/libusb"
    "${DEPENDS_DIR}/libusbx"
    ENV LibUSB_ROOT
  PATH_SUFFIXES
    x64/Release/dll
    x64/Debug/dll
    Win32/Release/dll
    Win32/Debug/dll
    MS64
    MS64/dll
)

IF(WIN32)
FIND_FILE(LibUSB_DLL
  ${LIBUSB_NAME}-1.0.dll
  PATHS
    "${DEPENDS_DIR}/libusb"
    "${DEPENDS_DIR}/libusbx"
    ENV LibUSB_ROOT
  PATH_SUFFIXES
    x64/Release/dll
    x64/Debug/dll
    Win32/Release/dll
    Win32/Debug/dll
    MS64
    MS64/dll
)
IF(LibUSB_DLL AND LIBUSB_USE_USBDK)
  FILE(COPY ${LibUSB_DLL} DESTINATION ${CMAKE_BINARY_DIR})
  SET(LibUSB_DLL ${CMAKE_BINARY_DIR}/libusb-1.0.dll)
  FILE(RENAME ${CMAKE_BINARY_DIR}/${LIBUSB_NAME}-1.0.dll ${LibUSB_DLL})
ENDIF()
ENDIF()

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LibUSB FOUND_VAR LibUSB_FOUND
  REQUIRED_VARS LibUSB_LIBRARIES LibUSB_INCLUDE_DIRS)
