# - Try to find GLFW3
#
# If no pkgconfig, define GLFW_ROOT to installation tree
# Will define the following:
# GLFW3_FOUND
# GLFW3_INCLUDE_DIRS
# GLFW3_LIBRARIES

IF(PKG_CONFIG_FOUND)
  IF(APPLE)
    # homebrew or macports pkgconfig locations
    SET(ENV{PKG_CONFIG_PATH} "/usr/local/opt/glfw3/lib/pkgconfig:/opt/local/lib/pkgconfig")
  ENDIF()
  SET(ENV{PKG_CONFIG_PATH} "${DEPENDS_DIR}/glfw/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
  PKG_CHECK_MODULES(GLFW3 glfw3)

  RETURN()
ENDIF()

FIND_PATH(GLFW3_INCLUDE_DIRS
  GLFW/glfw3.h
  DOC "GLFW include directory "
  PATHS
    "${DEPENDS_DIR}/glfw"
    "$ENV{ProgramW6432}/glfw"
    ENV GLFW_ROOT
  PATH_SUFFIXES
    include
)

# directories in the official binary package
IF(MINGW)
  SET(_SUFFIX lib-mingw)
ELSEIF(MSVC11)
  SET(_SUFFIX lib-vc2012)
ELSEIF(MSVC)
  SET(_SUFFIX lib-vc2013)
ENDIF()

FIND_LIBRARY(GLFW3_LIBRARIES
  NAMES glfw3dll glfw3
  PATHS
    "${DEPENDS_DIR}/glfw"
    "$ENV{ProgramW6432}/glfw"
    ENV GLFW_ROOT
  PATH_SUFFIXES
    lib
    ${_SUFFIX}
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GLFW3 "Could NOT find GLFW3 - try adding GLFW_ROOT in enviroment variables." GLFW3_LIBRARIES GLFW3_INCLUDE_DIRS)
