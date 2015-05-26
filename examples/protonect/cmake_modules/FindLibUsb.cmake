# - Find libusb for portable USB support
# This module will find libusb as published by
#  http://libusb.sf.net and
#  http://libusb-win32.sf.net
# 
# It will use PkgConfig if present and supported, else search
# it on its own. If the LibUSB_ROOT_DIR environment variable
# is defined, it will be used as base path.
# The following standard variables get defined:
#  LibUSB_FOUND:    true if LibUSB was found
#  LibUSB_INCLUDE_DIR: the directory that contains the include file
#  LibUSB_LIBRARIES:  the library

include ( CheckLibraryExists )
include ( CheckIncludeFile )

find_path ( LibUSB_INCLUDE_DIR
  NAMES
    libusb.h
  PATHS
    $ENV{ProgramFiles}/LibUSB-Win32
    $ENV{LibUSB_ROOT_DIR}
  PATH_SUFFIXES
    libusb
  )

mark_as_advanced ( LibUSB_INCLUDE_DIR )

if ( ${CMAKE_SYSTEM_NAME} STREQUAL "Windows" )
  # LibUSB-Win32 binary distribution contains several libs.
  # Use the lib that got compiled with the same compiler.
  if ( MSVC )
  if ( ${CMAKE_SIZEOF_VOID_P} EQUAL 8 )
    set ( LibUSB_LIBRARY_PATH_SUFFIX_RELEASE x64/Release/dll )
    set ( LibUSB_LIBRARY_PATH_SUFFIX_DEBUG x64/Debug/dll )
  else ()
    set ( LibUSB_LIBRARY_PATH_SUFFIX_RELEASE win32/Release/dll )
    set ( LibUSB_LIBRARY_PATH_SUFFIX_DEBUG win32/Debug/dll )
  endif ()      
  endif ( MSVC )
endif ( ${CMAKE_SYSTEM_NAME} STREQUAL "Windows" )

find_library ( LibUSB_LIBRARY_RELEASE
  NAMES
    libusb libusb-1.0 usb
  PATHS
    $ENV{ProgramFiles}/LibUSB-Win32
    $ENV{LibUSB_ROOT_DIR}
  PATH_SUFFIXES
    ${LibUSB_LIBRARY_PATH_SUFFIX_RELEASE}
  )
  
find_library ( LibUSB_LIBRARY_DEBUG
  NAMES
    libusb libusb-1.0 libusb-1.0d usb
  PATHS
    $ENV{ProgramFiles}/LibUSB-Win32
    $ENV{LibUSB_ROOT_DIR}
  PATH_SUFFIXES
    ${LibUSB_LIBRARY_PATH_SUFFIX_DEBUG}
  )  
  
set (LibUSB_LIBRARIES
  debug ${LibUSB_LIBRARY_DEBUG}
  optimized ${LibUSB_LIBRARY_RELEASE}
  )

if ( LibUSB_INCLUDE_DIR AND LibUSB_LIBRARIES )
  set ( LibUSB_FOUND 1 )
endif ( LibUSB_INCLUDE_DIR AND LibUSB_LIBRARIES )

if ( LibUSB_FOUND )
  set ( CMAKE_REQUIRED_INCLUDES "${LibUSB_INCLUDE_DIR}" )
  check_include_file ( usb.h LibUSB_FOUND )
endif ( LibUSB_FOUND )

if ( LibUSB_FOUND )
  check_library_exists ( "${LibUSB_LIBRARIES}" usb_open "" LibUSB_FOUND )
endif ( LibUSB_FOUND )

if ( NOT LibUSB_FOUND )
  if ( NOT LibUSB_FIND_QUIETLY )
  message ( STATUS "LibUSB not found, try setting LibUSB_ROOT_DIR environment variable." )
  endif ( NOT LibUSB_FIND_QUIETLY )
  if ( LibUSB_FIND_REQUIRED )
  message ( FATAL_ERROR "LibUSB could not be found." )
  endif ( LibUSB_FIND_REQUIRED )
endif ( NOT LibUSB_FOUND )
