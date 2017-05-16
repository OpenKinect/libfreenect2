# - Find OpenNI2
# 
# If the OPENNI2_INCLUDE and OPENNI2_REDIST environment variables
# are defined, they will be used as search path.
# The following standard variables get defined:
#  OpenNI2_FOUND:    true if found
#  OpenNI2_INCLUDE_DIRS: the directory that contains the include file
#  OpenNI2_LIBRARY_DIR: the directory that contains the library

IF(PKG_CONFIG_FOUND)
  PKG_CHECK_MODULES(OpenNI2 libopenni2)
ENDIF()

FIND_PATH(OpenNI2_INCLUDE_DIRS
  NAMES Driver/OniDriverAPI.h
  PATHS
    "/opt/include"
    "/opt/local/include"
    "/usr/include"
    "/usr/local/include"
    ENV OPENNI2_INCLUDE
    ENV PROGRAMFILES
    ENV ProgramW6432
  HINTS ${OpenNI2_INCLUDE_DIRS}
  PATH_SUFFIXES
    ni2
    openni2
    OpenNI2/Include
)

FIND_LIBRARY(OpenNI2_LIBRARY
  NAMES OpenNI2 ${OpenNI2_LIBRARIES}
  PATHS
   "/opt/lib"
    "/opt/local/lib"
    "/usr/lib"
    "/usr/local/lib"
    ENV OPENNI2_REDIST
    ENV PROGRAMFILES
    ENV ProgramW6432
  HINTS ${OpenNI2_LIBRARY_DIRS}
  PATH_SUFFIXES
    ni2/OpenNI2/Drivers
    OpenNI2/Drivers/lib
    OpenNI2/Lib
)

GET_FILENAME_COMPONENT(OpenNI2_LIBRARY_DIR ${OpenNI2_LIBRARY} DIRECTORY)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OpenNI2 FOUND_VAR OpenNI2_FOUND
  REQUIRED_VARS OpenNI2_LIBRARY_DIR OpenNI2_INCLUDE_DIRS)
