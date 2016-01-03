# - Find OpenNI2
# 
#  OpenNI2_FOUND:    true if OpenNI2 was found
#  OpenNI2_INCLUDE_DIRS: the directory that contains the include file
#  OpenNI2_INSTALL_DIR:  the directory that contains the librariy file

IF(PKG_CONFIG_FOUND)
  IF(CMAKE_SYSTEM_NAME MATCHES "Linux")
    SET(MODULE "libopenni2")
    IF(OpenNI2_FIND_REQUIRED)
      SET(OpenNI2_REQUIRED "REQUIRED")
    ENDIF()
    PKG_CHECK_MODULES(OpenNI2 ${OpenNI2_REQUIRED} ${MODULE})
    FIND_PATH(OpenNI2_INSTALL_DIR
      NAMES ${OpenNI2_LIBRARIES}
      HINTS ${OpenNI2_LIBRARY_DIRS}
    )
    SET(OpenNI2_INSTALL_DIR ${OpenNI2_LIBRARY})

    RETURN()
  ENDIF()
ENDIF()

FIND_PATH(OpenNI2_INCLUDE_DIRS
  NAMES Driver/OniDriverAPI.h
  PATHS
    "/opt/include"
    "/opt/local/include"
    "/usr/include"
    "/usr/local/include"
    ENV OPENNI2_INCLUDE
  PATH_SUFFIXES
    "ni2"
    "openni2"
)

FIND_PATH(OpenNI2_INSTALL_DIR
  NAMES
    libOniFile.so
    libOniFile.dylib
  PATHS
    "/opt/lib"
    "/opt/local/lib"
    "/usr/lib"
    "/usr/local/lib"
    ENV OPENNI2_REDIST
  PATH_SUFFIXES
    ni2/OpenNI2/Drivers
    OpenNI2/Drivers/lib
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenNI2 DEFAULT_MSG OpenNI2_INSTALL_DIR OpenNI2_INCLUDE_DIRS)
