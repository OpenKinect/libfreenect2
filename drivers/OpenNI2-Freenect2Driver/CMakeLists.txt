CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12.1)

if(WIN32 AND NOT MINGW)
  if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
  endif()
endif()

IF(NOT DEFINED CMAKE_BUILD_TYPE)
  # No effect for multi-configuration generators (e.g. for Visual Studio)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose: RelWithDebInfo Release Debug MinSizeRel None")
ENDIF()

PROJECT(openni2_freenect2driver)

SET(MY_DIR ${openni2_freenect2driver_SOURCE_DIR})
SET(freenect2_ROOT_DIR ${MY_DIR}/../..)

FIND_PACKAGE(PkgConfig)    # try find PKGConfig as it will be used if found
SET(OPENNI2_INCLUDE $ENV{OPENNI2_INCLUDE})
SET(OPENNI2_LINRARY $ENV{OPENNI2_REDIST})

IF(TARGET freenect2)
  MESSAGE(STATUS "Using in-tree freenect2 target")
  SET(freenect2_LIBRARIES freenect2)
  SET(freenect2_DLLS ${LIBFREENECT2_DLLS})
ELSE()
  FIND_PACKAGE(freenect2 REQUIRED)
  # Out-of-tree build will have to have DLLs manually copied.
ENDIF()

INCLUDE_DIRECTORIES(
  ${freenect2_INCLUDE_DIR}
  ${OPENNI2_INCLUDE}
)

SET(OpenNI2-Freenect2Driver_src
  src/ColorStream.cpp
  src/DepthStream.cpp
  src/IrStream.cpp
  src/DeviceDriver.cpp
  src/Registration.cpp
  ${freenect2_ROOT_DIR}/${LIBFREENECT2_THREADING_SOURCE}
)

SET(OpenNI2-Freenect2Driver_LIBRARIES
  ${freenect2_LIBRARIES}
)

SET(OpenNI2-Freenect2Driver_DLLS
  ${freenect2_DLLS}
)

ADD_LIBRARY(OpenNI2-Freenect2Driver ${OpenNI2-Freenect2Driver_src})

TARGET_LINK_LIBRARIES(OpenNI2-Freenect2Driver
  ${OpenNI2-Freenect2Driver_LIBRARIES}
)

INSTALL(TARGETS OpenNI2-Freenect2Driver DESTINATION "${OPENNI2_LINRARY}/OpenNI2/Drivers")
